"""
MPLMM-lite v5: conditional prompt의 용량(prompt_dim)을 키우면 ceiling에 더 가까워지는가?

v4에서 conditional prompt(prompt_dim=8)가 naive/static보다 평균적으로 나았지만
(결측만 f1 0.5189 vs 0.5039/0.5055), ceiling(0.7063)까지는 한참 못 미쳤다. 이 남은
격차가 "prompt 용량이 작아서"인지, 아니면 "애초에 결측된 정보 자체가 복구
불가능해서"인지 구분하기 위해, prompt_dim을 8→16→32로 키워가며 같은 3개 seed로
재검증한다.

- 용량을 키워도 격차가 안 줄면: 남은 격차는 모델 용량 문제가 아니라 원천적인
  정보 손실이라는 더 강한 결론을 낼 수 있다.
- 용량을 키우면 격차가 계속 줄면: 아직 우리가 쓴 8차원이 병목이었다는 뜻이고,
  더 큰 프롬프트가 여지가 있다는 뜻이다.

prompt_dim=8 결과는 v4에서 이미 seed 3개로 확보했으므로 재사용하고, 여기서는
16/32만 새로 돌린다. 나머지 설계(R-U-R만 제거하는 현실적 결측, seed 3개, train/val/test
split 고정)는 v4와 동일해서 직접 비교 가능하다.
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
from behavior_gnn_missing_modality_v4 import zero_rows_sparse, train_and_eval, MISSING_RATE, SEEDS, SPLIT_SEED

PROMPT_DIMS = [16, 32]  # 8은 v4에서 이미 확보

# v4 (prompt_dim=8) 결과 재사용 - reports/experiment_b_missing_modality_v4.csv 참고
V4_DIM8_MISSING_F1 = {42: 0.5137, 123: 0.5162, 2026: 0.5269}


class GraphBranchCapacity(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, condition_dim, prompt_dim,
                 num_layers=2, dropout=0.3):
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
        x_in[missing_mask] = 0.0
        generated = self.prompt_generator(condition)
        flag = torch.zeros(num_nodes, self.prompt_dim, device=x.device)
        flag[missing_mask] = generated[missing_mask]
        x_aug = torch.cat([x_in, flag], dim=-1)
        return self.gnn.forward_repr(x_aug, adjs)


class MissingAwareModelCapacity(nn.Module):
    def __init__(self, in_dim, num_relations, prompt_dim):
        super().__init__()
        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranchCapacity(
            in_dim, COMBINED_DIM, num_relations, condition_dim=COMBINED_DIM, prompt_dim=prompt_dim
        )
        self.classifier = nn.Linear(COMBINED_DIM * 2, 2)

    def forward(self, x, adjs, missing_mask):
        feat_repr = self.feature_branch(x)
        graph_repr = self.graph_branch(x, adjs, missing_mask, condition=feat_repr)
        fused = torch.cat([feat_repr, graph_repr], dim=-1)
        return self.classifier(fused)


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

    results = {8: [V4_DIM8_MISSING_F1[s] for s in SEEDS]}  # v4에서 재사용

    print(f"=== MPLMM v5: conditional prompt 용량(prompt_dim) 비교, seed {len(SEEDS)}개 ===\n")
    for prompt_dim in PROMPT_DIMS:
        print(f"\n########## prompt_dim = {prompt_dim} ##########")
        f1_list = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            missing_mask_np = np.zeros(num_nodes, dtype=bool)
            missing_mask_np[rng.choice(num_nodes, size=int(num_nodes * MISSING_RATE), replace=False)] = True
            missing_mask = torch.from_numpy(missing_mask_np)
            adjs_missing = [zero_rows_sparse(adjs_normal[0], missing_mask), adjs_normal[1], adjs_normal[2]]

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MissingAwareModelCapacity(in_dim=features.shape[1], num_relations=3, prompt_dim=prompt_dim)
            test_m, test_m_only = train_and_eval(model, x, y, adjs_missing, missing_mask, train_idx, val_idx, test_idx)
            f1_list.append(test_m_only["f1"] if test_m_only else float("nan"))
            print(f"  seed={seed}  결측만 f1={f1_list[-1]:.4f}")
        results[prompt_dim] = f1_list

    print("\n\n=== 최종 요약: prompt_dim별 결측만 f1 (mean ± std) ===")
    print("(참고) reference ceiling = 0.7063 ± 0.0048, naive = 0.5039 ± 0.0180 (v4에서)")
    for dim in [8] + PROMPT_DIMS:
        arr = np.array(results[dim])
        print(f"  prompt_dim={dim:3d}: {arr.mean():.4f} ± {arr.std():.4f}  (seed별: {', '.join(f'{v:.4f}' for v in arr)})")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_missing_modality_v5_capacity.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_dim", "missing_f1_mean", "missing_f1_std"] + [f"seed_{s}" for s in SEEDS])
        for dim in [8] + PROMPT_DIMS:
            arr = np.array(results[dim])
            w.writerow([dim, arr.mean(), arr.std()] + list(arr))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
