"""
MPLMM(Missing-modality Prompt Learning) 아이디어를 실험 B에 라이트하게 적용.

원래 MPLMM은 큰 멀티모달 트랜스포머에 학습형 prompt를 꽂아 modality 결측을
다루는 논문이라, 우리 스케일에서 그대로 구현하기엔 무겁다(architecture_rationale.md
4절에서 이미 "구현 안 함, 향후 연구"로 정리했었음). 대신 핵심 아이디어만 라이트하게
가져온다: modality가 없을 때 그냥 0으로 채우는 대신, "이 modality가 없다"는 것
자체를 나타내는 학습된 벡터(prompt)로 대체한다.

실제 시나리오: 리뷰 이력이 거의 없는 신규 유저는 reviewer-graph 상에서 유의미한
neighbor 정보가 없다 - 이건 behavior modality가 사실상 "결측"된 상황과 같다.
이걸 인위적으로 시뮬레이션해서(일부 노드의 graph representation을 강제로
결측 처리), 세 가지 방식을 비교한다:

  (1) reference: 결측 시뮬레이션 없음 (모든 노드가 graph 정보를 정상적으로 사용)
  (2) naive: 결측 노드의 graph representation을 0으로 채움
  (3) MPLMM-lite: 결측 노드의 graph representation을 학습된 prompt 벡터로 대체

fusion 메커니즘(Self-MM/MISA)은 이미 실험 B에서 도움이 안 된다는 게 확인됐으므로
(reports/experiment_b_hparam_sweep.md), 여기서는 plain concat(가장 성능이 좋았던
세팅)을 기반으로 결측 처리 방식만 따로 비교한다.
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from models.simple_gnn import normalize_adj
from behavior_gnn_fusion import GraphBranch, evaluate, DATA_PATH, SEED, COMBINED_DIM
from models.fusion_modules import BranchEncoder

MISSING_RATE = 0.2  # 전체 노드 중 "신규 유저(그래프 정보 없음)"로 시뮬레이션할 비율


class MissingAwareModel(nn.Module):
    """plain concat fusion + 결측 graph representation 처리 방식(none/zero/prompt) 실험용"""

    def __init__(self, in_dim, num_relations, missing_strategy="none"):
        super().__init__()
        self.missing_strategy = missing_strategy  # "none" | "zero" | "prompt"
        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranch(in_dim, COMBINED_DIM, num_relations)
        self.classifier = nn.Linear(COMBINED_DIM * 2, 2)
        if missing_strategy == "prompt":
            self.missing_prompt = nn.Parameter(torch.randn(COMBINED_DIM) * 0.02)

    def forward(self, x, adjs, missing_mask=None):
        feat_repr = self.feature_branch(x)
        graph_repr = self.graph_branch(x, adjs)

        if missing_mask is not None and self.missing_strategy != "none":
            graph_repr = graph_repr.clone()
            if self.missing_strategy == "zero":
                graph_repr[missing_mask] = 0.0
            elif self.missing_strategy == "prompt":
                graph_repr[missing_mask] = self.missing_prompt

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
                # 결측으로 시뮬레이션된 test 노드만 따로도 확인 (효과가 여기 집중돼야 함)
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
          f"'그래프 정보 없음(신규 유저)'으로 시뮬레이션\n")

    # missing_mask는 세 세팅 모두에 전달한다 - reference는 strategy="none"이라 실제로는
    # graph representation을 건드리지 않지만, 그래도 "나중에 결측 처리될 노드들"만 따로
    # 평가해서 "정보가 있었다면 어느 정도 성능이 나왔을지"(ceiling)를 (2)/(3)과 비교할 수 있게 한다.
    settings = [
        ("(1) reference: 결측 시뮬레이션 없음 (ceiling)", "none", missing_mask),
        ("(2) naive: 결측 노드 graph repr = 0", "zero", missing_mask),
        ("(3) MPLMM-lite: 결측 노드 graph repr = 학습된 prompt", "prompt", missing_mask),
    ]

    results = []
    print("=== MPLMM-lite: 결측 modality(신규 유저) 처리 방식 비교 (YelpChi) ===\n")
    for name, strategy, mask_for_train in settings:
        print(f"[{name}]")
        torch.manual_seed(SEED)
        model = MissingAwareModel(in_dim=features.shape[1], num_relations=3, missing_strategy=strategy)
        test_metrics, test_missing_only = train_and_eval(
            model, x, y, adjs, mask_for_train, train_idx, val_idx, test_idx
        )
        print(f"  -> 전체 test: acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")
        if test_missing_only:
            print(f"  -> 결측 노드만: acc={test_missing_only['acc']:.4f}  f1={test_missing_only['f1']:.4f}  auc={test_missing_only['auc']:.4f}")
        print()
        results.append((name, test_metrics, test_missing_only))

    print("=== 요약 (전체 test set 기준) ===")
    print(f"{'setting':45s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for name, m, _ in results:
        print(f"{name:45s} {m['acc']:8.4f} {m['f1']:8.4f} {m['auc']:8.4f}")

    print("\n=== 요약 (결측으로 시뮬레이션된 노드만) ===")
    print(f"{'setting':45s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for name, _, mo in results:
        if mo:
            print(f"{name:45s} {mo['acc']:8.4f} {mo['f1']:8.4f} {mo['auc']:8.4f}")
        else:
            print(f"{name:45s} {'(결측 없음)':>26s}")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_missing_modality.csv")
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
