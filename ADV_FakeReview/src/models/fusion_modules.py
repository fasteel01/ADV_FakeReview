"""
Self-MM / MISA / MMIM 아이디어를 경량화해서 구현한 fusion module들.

설계 원칙(자세한 근거는 reports/fusion_design.md 참고):
  - Self-MM: modality별 auxiliary classification head. 원 논문의 self-supervised
    label generation(감정 강도 회귀용 알고리즘)은 이진분류에 안 맞아서 빼고,
    "각 branch가 자기 정보만으로도 어느정도 분류할 수 있어야 한다"는
    핵심 아이디어만 auxiliary loss로 가져왔다.
  - MISA: modality-invariant/specific representation 분리. 원 논문의
    CMD(Central Moment Discrepancy) loss 대신, 구현이 훨씬 간단하고 안정적인
    "orthogonality loss(내적을 0에 가깝게)"로 shared/specific을 분리한다.
  - MMIM: 여기엔 없음. mutual information estimator 대신 CLIP cosine similarity를
    바로 feature로 쓰는 쪽을 택했다(src/baseline_textimage.py의 CLIP 관련 코드 참고).
  - MAG, MPLMM: 구현하지 않음 (이유는 reports/fusion_design.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchEncoder(nn.Module):
    """원본 modality feature -> 공통 차원(combined_dim)의 branch representation"""

    def __init__(self, in_dim, combined_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, combined_dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfMMAuxHead(nn.Module):
    """Self-MM에서 영감을 받은 modality-specific auxiliary classifier.
    branch representation만 보고 fake/real을 맞혀보게 해서, fusion 전에도
    각 branch가 유의미한 정보를 담도록 유도한다."""

    def __init__(self, combined_dim, num_classes=2):
        super().__init__()
        self.head = nn.Linear(combined_dim, num_classes)

    def forward(self, branch_repr):
        return self.head(branch_repr)


class MISALiteFusion(nn.Module):
    """MISA에서 영감을 받은 shared/specific 분리.

    - shared_proj: 모든 branch에 대해 '같은 가중치'를 공유하는 linear layer.
      서로 다른 modality raw representation을 같은 변환에 통과시켜서
      "modality-invariant" 정보만 남기도록 유도.
    - specific_proj: branch마다 별도의 linear layer. modality 고유 정보를 보존.
    - orthogonality loss: 같은 branch의 shared/specific 벡터가 서로 다른 정보를
      담도록(내적이 0에 가깝도록) 정규화. MISA 원 논문의 CMD loss를 대신하는
      훨씬 가벼운 대체재.
    """

    def __init__(self, combined_dim, shared_dim=32, specific_dim=32, num_branches=2):
        super().__init__()
        self.shared_proj = nn.Linear(combined_dim, shared_dim)  # 모든 branch가 공유
        self.specific_projs = nn.ModuleList([
            nn.Linear(combined_dim, specific_dim) for _ in range(num_branches)
        ])

    def forward(self, branch_reprs):
        """branch_reprs: list of [batch, combined_dim] tensors (branch 순서 고정)"""
        shared = [self.shared_proj(b) for b in branch_reprs]
        specific = [proj(b) for proj, b in zip(self.specific_projs, branch_reprs)]

        # orthogonality loss: 같은 branch 안에서 shared/specific이 다른 정보를 담도록
        ortho_loss = sum(
            F.cosine_similarity(s, sp, dim=-1).pow(2).mean()
            for s, sp in zip(shared, specific)
        ) / len(branch_reprs)

        fused = torch.cat(shared + specific, dim=-1)
        return fused, ortho_loss
