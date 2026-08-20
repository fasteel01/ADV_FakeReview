"""
MPLMM-lite v3: 결측 프롬프트를 고정 상수가 아니라 "가진 정보로 조건부 생성"하도록 확장.

v2에서 확인한 것: 위치(GNN 입력 단)는 맞게 고쳤지만, 모든 결측 노드에 완전히
동일한 학습된 벡터 하나(`missing_prompt`, 파라미터로 고정)를 꽂아 넣다 보니
"정보가 없다"는 사실은 알려줘도 "그 노드에 맞게 어떻게 보완할지"는 학습할 수
없었다. 그래서 naive 대비 개선폭이 작았다(reports/experiment_b_missing_modality_v2.csv).

v3는 이 부분을 고친다: 고정 벡터 대신, **결측되지 않은 다른 정보(feat_repr,
즉 리뷰 자체의 텍스트/feature 정보 - "그래프 정보는 없어도 이 리뷰 자체의 특징은
안다")를 입력으로 받아 매 노드마다 다른 prompt를 생성하는 작은 MLP**를 쓴다.

  prompt_i = MLP(feat_repr_i)   (모든 노드에 대해 계산되지만, 실제로 쓰이는 건
                                  missing_mask==True인 노드뿐)

주의: feat_repr는 원본 feature(x)를 그대로 쓰는 feature_branch에서 나온 것이라
"결측 시뮬레이션"의 영향을 받지 않는다(이 실험에서 결측 처리는 graph_branch
입력에만 적용됨). 즉 여기서 prompt를 생성하는 데 쓰는 정보는 진짜로 "아직
갖고 있는" 정보이지, 결측된 정보를 몰래 훔쳐보는 게 아니다.

reference/naive/v2(static prompt) 결과는 이미 experiment_b_missing_modality_v2.csv에
있으므로, 여기서는 cond_prompt 세팅 하나만 새로 학습해서 비교한다(같은 seed,
같은 split, 같은 결측 마스크 사용 - 공정 비교).
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


class GraphBranchConditionalPrompt(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, condition_dim,
                 prompt_dim=PROMPT_DIM, num_layers=2, dropout=0.3):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_generator = nn.Sequential(
            nn.Linear(condition_dim, 32), nn.ReLU(),
            nn.Linear(32, prompt_dim),
        )
        self.gnn = RelationalGNN(in_dim + prompt_dim, hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, adjs, missing_mask, condition):
        num_nodes = x.shape[0]
        x_in = x.clone()
        x_in[missing_mask] = 0.0  # 결측 노드는 진짜로 feature를 지움 (v2와 동일)

        generated = self.prompt_generator(condition)  # [num_nodes, prompt_dim], 노드별로 다름
        flag = torch.zeros(num_nodes, self.prompt_dim, device=x.device)
        flag[missing_mask] = generated[missing_mask]

        x_aug = torch.cat([x_in, flag], dim=-1)
        return self.gnn.forward_repr(x_aug, adjs)


class MissingAwareModelV3(nn.Module):
    def __init__(self, in_dim, num_relations):
        super().__init__()
        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranchConditionalPrompt(
            in_dim, COMBINED_DIM, num_relations, condition_dim=COMBINED_DIM
        )
        self.classifier = nn.Linear(COMBINED_DIM * 2, 2)

    def forward(self, x, adjs, missing_mask):
        feat_repr = self.feature_branch(x)  # 결측 마스크 영향 안 받음 - "아직 갖고 있는 정보"
        graph_repr = self.graph_branch(x, adjs, missing_mask, condition=feat_repr)
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

    # v2와 완전히 동일한 결측 마스크 (같은 SEED, 같은 rng 호출 순서) - 공정 비교 위해 필수
    rng = np.random.RandomState(SEED)
    missing_mask_np = np.zeros(num_nodes, dtype=bool)
    missing_mask_np[rng.choice(num_nodes, size=int(num_nodes * MISSING_RATE), replace=False)] = True
    missing_mask = torch.from_numpy(missing_mask_np)
    print(f"전체 노드 {num_nodes}개 중 {missing_mask_np.sum()}개({MISSING_RATE*100:.0f}%) 결측 "
          f"(v3: feat_repr 기반 conditional prompt)\n")

    print("=== (4) MPLMM v3: feat_repr 기반 conditional prompt ===")
    torch.manual_seed(SEED)
    model = MissingAwareModelV3(in_dim=features.shape[1], num_relations=3)
    test_metrics, test_missing_only = train_and_eval(
        model, x, y, adjs, missing_mask, train_idx, val_idx, test_idx
    )
    print(f"  -> 전체 test: acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")
    if test_missing_only:
        print(f"  -> 결측 노드만: acc={test_missing_only['acc']:.4f}  f1={test_missing_only['f1']:.4f}  auc={test_missing_only['auc']:.4f}")

    print("\n=== v1/v2/v3 전체 비교 (결측 노드만 기준, f1) ===")
    print("(1) reference (ceiling)               f1=0.6897")
    print("(2) naive (feature=0)                 f1=0.6147")
    print("(3) v2 static prompt (GNN 입력단)      f1=0.6154")
    print(f"(4) v3 conditional prompt (feat_repr 기반)  f1={test_missing_only['f1']:.4f}" if test_missing_only else "(4) v3: (결측 없음)")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_missing_modality_v3.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "acc", "f1", "auc", "missing_only_acc", "missing_only_f1", "missing_only_auc"])
        row = ["(4) v3 conditional prompt (feat_repr 기반)", test_metrics["acc"], test_metrics["f1"], test_metrics["auc"]]
        row += [test_missing_only["acc"], test_missing_only["f1"], test_missing_only["auc"]] if test_missing_only else ["", "", ""]
        w.writerow(row)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
