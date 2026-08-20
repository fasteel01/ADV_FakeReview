"""
실험 B의 "+ Self-MM + MISA-lite" 세팅이 하이퍼파라미터(aux_weight, ortho_weight)에
민감해서 우연히 잘 나온 건 아닌지 가볍게 확인하는 스크립트.

behavior_gnn_fusion.py의 기본값은 aux_weight=0.3, ortho_weight=0.1였다(별도로
튜닝한 값이 아니라 상식적인 기본값). 여기서는 그 근방의 다른 조합 몇 개로도
비슷한 성능이 나오는지만 확인한다 - 최적값을 찾는 그리드서치가 아니라, "결과가
특정 값 하나에만 우연히 맞아떨어진 게 아니다"를 보여주는 robustness check.
"""
import os
import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split

from behavior_gnn_fusion import FusionModel, train_model, DATA_PATH, SEED

# (aux_weight, ortho_weight) 조합. 기본값(0.3, 0.1) 포함 + 위아래로 벌어진 값 2개.
SWEEP = [
    (0.1, 0.1),
    (0.3, 0.1),   # behavior_gnn_fusion.py 기본값
    (0.5, 0.3),
]


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
    from models.simple_gnn import normalize_adj
    adjs = [normalize_adj(d[k], num_nodes) for k in ["net_rur", "net_rtr", "net_rsr"]]

    idx = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=label, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=label[temp_idx], random_state=SEED)
    train_idx, val_idx, test_idx = map(torch.from_numpy, (train_idx, val_idx, test_idx))

    print("=== 실험 B robustness check: aux_weight / ortho_weight 민감도 ===\n")
    results = []
    for aux_w, ortho_w in SWEEP:
        print(f"[aux_weight={aux_w}, ortho_weight={ortho_w}]")
        torch.manual_seed(SEED)
        model = FusionModel(in_dim=features.shape[1], num_relations=3, use_aux=True, use_misa=True)
        test_metrics = train_model(
            model, x, y, adjs, train_idx, val_idx, test_idx,
            aux_weight=aux_w, ortho_weight=ortho_w,
        )
        print(f"  -> test acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}\n")
        results.append((aux_w, ortho_w, test_metrics))

    print("=== 요약 ===")
    print(f"{'aux_weight':>10s} {'ortho_weight':>13s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for aux_w, ortho_w, m in results:
        print(f"{aux_w:10.2f} {ortho_w:13.2f} {m['acc']:8.4f} {m['f1']:8.4f} {m['auc']:8.4f}")

    accs = [m["acc"] for _, _, m in results]
    f1s = [m["f1"] for _, _, m in results]
    print(f"\nacc 범위: {min(accs):.4f} ~ {max(accs):.4f} (편차 {max(accs)-min(accs):.4f})")
    print(f"f1  범위: {min(f1s):.4f} ~ {max(f1s):.4f} (편차 {max(f1s)-min(f1s):.4f})")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_hparam_sweep.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aux_weight", "ortho_weight", "acc", "f1", "auc"])
        for aux_w, ortho_w, m in results:
            w.writerow([aux_w, ortho_w, m["acc"], m["f1"], m["auc"]])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
