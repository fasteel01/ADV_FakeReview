"""
MPLMM-lite v2: 결측 프롬프트를 "분류기 직전"이 아니라 GNN "입력 단"에 주입.

behavior_gnn_missing_modality.py(v1)에서는 GNN이 계산을 다 끝낸 뒤 graph
representation을 통째로 대체했는데, 그 지점 바로 뒤가 단일 Linear classifier라서
0을 넣든 학습된 벡터를 넣든 편향(bias)으로 흡수돼 차이가 안 났다
(reports/experiment_b_missing_modality.md 참고).

v2는 그 문제를 구조적으로 고친다: 결측 플래그를 GNN에 "들어가기 전" 원본 feature에
이어붙여서(concat), RelationalGNN의 2개 layer(각각 Linear + ReLU + 이웃
aggregation)를 실제로 통과하게 만든다. 특히 이웃 aggregation(sparse.mm)이 있어서,
한 노드에 심은 결측 신호가 그래프를 타고 이웃 노드에게도 전파된다 - 이건 v1에는
없던, 원 MPLMM 논문의 "정보가 여러 레이어를 거치며 능동적으로 보완된다"는 취지에
더 가까운 메커니즘이다.

비교 대상:
  (1) reference: 결측 시뮬레이션 없음 (구조는 동일하게, 플래그 채널은 항상 0)
  (2) naive: 결측 노드 feature를 0으로 지움, 플래그 채널도 0
  (3) MPLMM (input-level prompt): 결측 노드 feature를 0으로 지움 + 학습된 플래그 채널 삽입
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from models.simple_gnn import RelationalGNN, normalize_adj
from behavior_gnn_fusion import evaluate, DATA_PATH, SEED, COMBINED_DIM
from models.fusion_modules import BranchEncoder

MISSING_RATE = 0.2
PROMPT_DIM = 8


class GraphBranchWithPrompt(nn.Module):
    """결측 플래그를 GNN 입력 단에서 concat하는 graph branch.

    strategy:
      "none"   - missing_mask 무시, feature/flag 둘 다 원본 그대로(flag는 항상 0)
      "naive"  - 결측 노드 feature를 0으로, flag는 항상 0 (신호 없음)
      "prompt" - 결측 노드 feature를 0으로, flag에 학습된 missing_prompt 삽입
    """

    def __init__(self, in_dim, hidden_dim, num_relations, prompt_dim=PROMPT_DIM,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.missing_prompt = nn.Parameter(torch.randn(prompt_dim) * 0.02)
        # GNN 입력 차원이 원본 feature + prompt_dim만큼 늘어남
        self.gnn = RelationalGNN(in_dim + prompt_dim, hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, adjs, missing_mask, strategy):
        num_nodes = x.shape[0]
        flag = torch.zeros(num_nodes, self.prompt_dim, device=x.device)
        x_in = x

        if strategy != "none" and missing_mask is not None:
            x_in = x.clone()
            x_in[missing_mask] = 0.0  # 결측 노드는 원본 정보를 진짜로 지움
            if strategy == "prompt":
                flag = flag.clone()
                flag[missing_mask] = self.missing_prompt  # 학습된 결측 신호 삽입

        x_aug = torch.cat([x_in, flag], dim=-1)
        return self.gnn.forward_repr(x_aug, adjs)


class MissingAwareModelV2(nn.Module):
    def __init__(self, in_dim, num_relations, strategy="none"):
        super().__init__()
        self.strategy = strategy
        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranchWithPrompt(in_dim, COMBINED_DIM, num_relations)
        self.classifier = nn.Linear(COMBINED_DIM * 2, 2)

    def forward(self, x, adjs, missing_mask=None):
        feat_repr = self.feature_branch(x)  # feature branch는 원본 feature 그대로 사용(비교 기준 유지)
        graph_repr = self.graph_branch(x, adjs, missing_mask, self.strategy)
        fused = torch.cat([feat_repr, graph_repr], dim=-1)
        return self.classifier(fused)


def train_and_eval(model, x, y, adjs, missing_mask, train_idx, val_idx, test_idx, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    n_pos = y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

    best_val_f1, best_test, best_test_missing_only = -1, None, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjs, missing_mask)
        loss = F.cross_entropy(logits[train_idx], y[train_idx], weight=weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adjs, missing_mask)
            val_metrics = evaluate(logits, y, val_idx)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_test = evaluate(logits, y, test_idx)
                if missing_mask is not None:
                    test_missing_idx = test_idx[missing_mask[test_idx]]
                    if len(test_missing_idx) > 5:
                        best_test_missing_only = evaluate(logits, y, test_missing_idx)

        if (epoch + 1) % 25 == 0:
            print(f"    epoch {epoch+1:3d}  loss={loss.item():.4f}  val_f1={val_metrics['f1']:.4f}")

    return best_test, best_test_missing_only


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = sio.loadmat(DATA_PATH)
    label = d["label"].flatten().astype("int64")
    features = np.asarray(d["features"].todense()).astype("float32")
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    num_nodes = features.shape[0]

    x = torch.from_numpy(features)
    y = torch.from_numpy(label)
    adjs = [normalize_adj(d[k], num_nodes) for k in ["net_rur", "net_rtr", "net_rsr"]]

    idx = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=label, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=label[temp_idx], random_state=SEED)
    train_idx, val_idx, test_idx = map(torch.from_numpy, (train_idx, val_idx, test_idx))

    rng = np.random.RandomState(SEED)
    missing_mask_np = np.zeros(num_nodes, dtype=bool)
    missing_mask_np[rng.choice(num_nodes, size=int(num_nodes * MISSING_RATE), replace=False)] = True
    missing_mask = torch.from_numpy(missing_mask_np)
    print(f"전체 노드 {num_nodes}개 중 {missing_mask_np.sum()}개({MISSING_RATE*100:.0f}%)를 "
          f"'그래프 정보 없음(신규 유저)'으로 시뮬레이션 (v2: GNN 입력 단 주입)\n")

    settings = [
        ("(1) reference: 결측 시뮬레이션 없음 (ceiling)", "none"),
        ("(2) naive: 결측 노드 feature=0, flag 없음", "naive"),
        ("(3) MPLMM v2: 결측 노드 feature=0 + 학습된 flag를 GNN 입력에 주입", "prompt"),
    ]

    results = []
    print("=== MPLMM v2: 결측 modality를 GNN 입력 단에서 처리 (YelpChi) ===\n")
    for name, strategy in settings:
        print(f"[{name}]")
        torch.manual_seed(SEED)
        model = MissingAwareModelV2(in_dim=features.shape[1], num_relations=3, strategy=strategy)
        test_metrics, test_missing_only = train_and_eval(
            model, x, y, adjs, missing_mask, train_idx, val_idx, test_idx
        )
        print(f"  -> 전체 test: acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")
        if test_missing_only:
            print(f"  -> 결측 노드만: acc={test_missing_only['acc']:.4f}  f1={test_missing_only['f1']:.4f}  auc={test_missing_only['auc']:.4f}")
        print()
        results.append((name, test_metrics, test_missing_only))

    print("=== 요약 (전체 test set 기준) ===")
    print(f"{'setting':55s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for name, m, _ in results:
        print(f"{name:55s} {m['acc']:8.4f} {m['f1']:8.4f} {m['auc']:8.4f}")

    print("\n=== 요약 (결측으로 시뮬레이션된 노드만) ===")
    print(f"{'setting':55s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for name, _, mo in results:
        if mo:
            print(f"{name:55s} {mo['acc']:8.4f} {mo['f1']:8.4f} {mo['auc']:8.4f}")
        else:
            print(f"{name:55s} {'(결측 없음)':>26s}")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_missing_modality_v2.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "acc", "f1", "auc", "missing_only_acc", "missing_only_f1", "missing_only_auc"])
        for name, m, mo in results:
            row = [name, m["acc"], m["f1"], m["auc"]]
            row += [mo["acc"], mo["f1"], mo["auc"]] if mo else ["", "", ""]
            w.writerow(row)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
