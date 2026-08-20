"""
MPLMM-lite v4: (B) 더 현실적인 결측 시뮬레이션 + (A) multi-seed 재현성 검증.

v1~v3(reports/experiment_b_missing_modality_v3_findings.md)까지는 두 가지 한계가 있었다:

  (B) 결측 노드의 feature만 0으로 지우고, 그래프 엣지(R-U-R/R-T-R/R-S-R)는 그대로
      뒀다. 즉 "신규 유저"라 해놓고도 이웃으로부터는 정상적으로 정보를 받을 수
      있었다 - 결측 시뮬레이션이 실제보다 약했다.
  (A) 모든 비교가 seed 하나(=42)에서 나온 단일 실행 결과였다. naive/v2/v3 차이가
      작아서(f1 0.6147/0.6154/0.6234), 이게 진짜 효과인지 우연인지 확인이 안 됐다.

이 스크립트는 둘 다 고친다:

  (B) R-U-R(같은 유저) 관계만 결측 노드에 대해 끊는다. R-T-R(같은 상품+시기)/
      R-S-R(같은 상품+평점)은 유저 이력이 아니라 리뷰 자체의 상품/시기/평점에
      의존하므로, 신규 유저의 리뷰라도 여전히 유효하다 - 그대로 둔다.
      (FraudSquad류가 애초에 관계를 3개로 나눠서 쓰는 이유가 바로 이거라, 이번에
      실험으로 그 근거를 다시 확인하는 셈이다.)
  (A) seed 3개(42, 123, 2026)로 반복해서 mean±std를 낸다.

비교하는 4개 세팅은 v1~v3와 동일한 개념이지만, 이번엔 하나의 모델 클래스로
통일해서(strategy 파라미터로 구분) 아키텍처 파라미터 수를 동일하게 맞췄다:
  none   = reference (결측 시뮬레이션 없음, ceiling)
  naive  = feature 0 + R-U-R 제거, flag 없음
  static = feature 0 + R-U-R 제거, flag = 모든 결측 노드 동일한 학습된 벡터 (v2)
  cond   = feature 0 + R-U-R 제거, flag = feat_repr 기반으로 노드별 생성 (v3)
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from models.simple_gnn import RelationalGNN, normalize_adj
from behavior_gnn_fusion import evaluate, DATA_PATH, COMBINED_DIM
from models.fusion_modules import BranchEncoder

MISSING_RATE = 0.2
PROMPT_DIM = 8
SEEDS = [42, 123, 2026]
SPLIT_SEED = 42  # train/val/test split은 seed 간에 고정 (결측 시뮬레이션/모델 초기화 변동만 보기 위해)


def zero_rows_sparse(adj, mask):
    """sparse COO adjacency에서 mask==True인 행(row)의 엣지를 전부 제거."""
    indices = adj.indices()
    values = adj.values()
    keep = ~mask[indices[0]]
    new_indices = indices[:, keep]
    new_values = values[keep]
    return torch.sparse_coo_tensor(new_indices, new_values, adj.shape).coalesce()


class GraphBranchV4(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, condition_dim,
                 prompt_dim=PROMPT_DIM, num_layers=2, dropout=0.3):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.missing_prompt = nn.Parameter(torch.randn(prompt_dim) * 0.02)
        self.prompt_generator = nn.Sequential(
            nn.Linear(condition_dim, 32), nn.ReLU(),
            nn.Linear(32, prompt_dim),
        )
        self.gnn = RelationalGNN(in_dim + prompt_dim, hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, adjs, missing_mask, strategy, condition):
        num_nodes = x.shape[0]
        x_in = x
        flag = torch.zeros(num_nodes, self.prompt_dim, device=x.device)

        if strategy != "none":
            x_in = x.clone()
            x_in[missing_mask] = 0.0
            if strategy == "static":
                flag = flag.clone()
                flag[missing_mask] = self.missing_prompt
            elif strategy == "cond":
                generated = self.prompt_generator(condition)
                flag = flag.clone()
                flag[missing_mask] = generated[missing_mask]
            # strategy == "naive": flag는 0으로 둔 채 그대로

        x_aug = torch.cat([x_in, flag], dim=-1)
        return self.gnn.forward_repr(x_aug, adjs)


class MissingAwareModelV4(nn.Module):
    def __init__(self, in_dim, num_relations, strategy="none"):
        super().__init__()
        self.strategy = strategy
        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranchV4(in_dim, COMBINED_DIM, num_relations, condition_dim=COMBINED_DIM)
        self.classifier = nn.Linear(COMBINED_DIM * 2, 2)

    def forward(self, x, adjs, missing_mask):
        feat_repr = self.feature_branch(x)
        graph_repr = self.graph_branch(x, adjs, missing_mask, self.strategy, condition=feat_repr)
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
    return best_test, best_test_missing_only


def main():
    d = sio.loadmat(DATA_PATH)
    label = d["label"].flatten().astype("int64")
    features = np.asarray(d["features"].todense()).astype("float32")
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    num_nodes = features.shape[0]

    x = torch.from_numpy(features)
    y = torch.from_numpy(label)
    adjs_normal = [normalize_adj(d[k], num_nodes) for k in ["net_rur", "net_rtr", "net_rsr"]]

    idx = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=label, random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=label[temp_idx], random_state=SPLIT_SEED)
    train_idx, val_idx, test_idx = map(torch.from_numpy, (train_idx, val_idx, test_idx))

    settings = ["none", "naive", "static", "cond"]
    setting_names = {
        "none": "(1) reference (ceiling)",
        "naive": "(2) naive (feature=0, R-U-R 제거)",
        "static": "(3) static prompt (feature=0, R-U-R 제거 + 동일 벡터)",
        "cond": "(4) conditional prompt (feature=0, R-U-R 제거 + feat_repr 기반 생성)",
    }

    # {setting: {"acc": [...], "f1": [...], "auc": [...], "m_acc": [...], "m_f1": [...], "m_auc": [...]}}
    all_results = {s: {"acc": [], "f1": [], "auc": [], "m_acc": [], "m_f1": [], "m_auc": []} for s in settings}

    print(f"=== MPLMM v4: 현실적 결측(R-U-R만 제거) + seed {len(SEEDS)}개 반복 ===\n")

    for seed in SEEDS:
        print(f"\n########## SEED {seed} ##########")
        rng = np.random.RandomState(seed)
        missing_mask_np = np.zeros(num_nodes, dtype=bool)
        missing_mask_np[rng.choice(num_nodes, size=int(num_nodes * MISSING_RATE), replace=False)] = True
        missing_mask = torch.from_numpy(missing_mask_np)

        adjs_missing = [zero_rows_sparse(adjs_normal[0], missing_mask), adjs_normal[1], adjs_normal[2]]

        for s in settings:
            adjs_for_run = adjs_normal if s == "none" else adjs_missing
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MissingAwareModelV4(in_dim=features.shape[1], num_relations=3, strategy=s)
            test_m, test_m_only = train_and_eval(model, x, y, adjs_for_run, missing_mask, train_idx, val_idx, test_idx)
            all_results[s]["acc"].append(test_m["acc"])
            all_results[s]["f1"].append(test_m["f1"])
            all_results[s]["auc"].append(test_m["auc"])
            if test_m_only:
                all_results[s]["m_acc"].append(test_m_only["acc"])
                all_results[s]["m_f1"].append(test_m_only["f1"])
                all_results[s]["m_auc"].append(test_m_only["auc"])
            print(f"  [{setting_names[s]}] seed={seed}  "
                  f"전체 f1={test_m['f1']:.4f}  결측만 f1={test_m_only['f1'] if test_m_only else float('nan'):.4f}")

    print("\n\n=== 최종 요약 (mean ± std, seed 3개) ===")
    print(f"{'setting':60s} {'전체 f1':>16s} {'결측만 f1':>16s} {'결측만 acc':>16s} {'결측만 auc':>16s}")
    rows = []
    for s in settings:
        r = all_results[s]
        f1_mean, f1_std = np.mean(r["f1"]), np.std(r["f1"])
        mf1_mean, mf1_std = np.mean(r["m_f1"]), np.std(r["m_f1"])
        macc_mean, macc_std = np.mean(r["m_acc"]), np.std(r["m_acc"])
        mauc_mean, mauc_std = np.mean(r["m_auc"]), np.std(r["m_auc"])
        print(f"{setting_names[s]:60s} "
              f"{f1_mean:.4f}±{f1_std:.4f}  {mf1_mean:.4f}±{mf1_std:.4f}  "
              f"{macc_mean:.4f}±{macc_std:.4f}  {mauc_mean:.4f}±{mauc_std:.4f}")
        rows.append((s, f1_mean, f1_std, mf1_mean, mf1_std, macc_mean, macc_std, mauc_mean, mauc_std,
                     r["f1"], r["m_f1"]))

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_missing_modality_v4.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "f1_mean", "f1_std", "missing_f1_mean", "missing_f1_std",
                     "missing_acc_mean", "missing_acc_std", "missing_auc_mean", "missing_auc_std",
                     "f1_per_seed", "missing_f1_per_seed"])
        for row in rows:
            s, f1m, f1s, mf1m, mf1s, maccm, maccs, maucm, maucs, f1list, mf1list = row
            w.writerow([setting_names[s], f1m, f1s, mf1m, mf1s, maccm, maccs, maucm, maucs,
                        ";".join(f"{v:.4f}" for v in f1list), ";".join(f"{v:.4f}" for v in mf1list)])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
