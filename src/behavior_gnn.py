"""
실험 B: Text(feature) + Reviewer Behavior graph baseline (YelpChi)

비교:
  (1) feature-only  : YelpChi의 32-dim handcrafted feature(원 텍스트에서 추출된 것,
                       Rayana & Akoglu 2015)만 사용한 MLP -> "텍스트만 있을 때"에 해당
  (2) + graph (R-U-R/R-T-R/R-S-R) : FraudSquad/CARE-GNN 방식의 관계 그래프를
                       추가한 경량 GNN (src/models/simple_gnn.py)

핵심 질문: "리뷰어 행동 관계 그래프를 추가하면 text-only 대비 성능이 오르는가?"
이 오르면 Reviewer Behavior modality가 실제로 fake review 탐지에 신호를 담고
있다는 근거가 되고, 최종 3-modality 구조에서 graph branch를 유지할 논리적
정당성이 생긴다.
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from models.simple_gnn import RelationalGNN, normalize_adj

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "yelpchi", "YelpChi.mat")
SEED = 42


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(logits, y, mask):
    proba = F.softmax(logits[mask], dim=1)[:, 1].detach().numpy()
    pred = logits[mask].argmax(dim=1).detach().numpy()
    y_true = y[mask].numpy()
    return {
        "acc": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba),
    }


def train_model(model, x, y, adjs, train_idx, val_idx, test_idx, epochs=150, lr=0.01, use_graph=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # class imbalance -> weighted loss
    n_pos = y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

    best_val_f1, best_test = -1, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjs) if use_graph else model(x)
        loss = F.cross_entropy(logits[train_idx], y[train_idx], weight=weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adjs) if use_graph else model(x)
            val_metrics = evaluate(logits, y, val_idx)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_test = evaluate(logits, y, test_idx)

        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}  val_f1={val_metrics['f1']:.4f}")

    return best_test


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

    idx = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=label, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=label[temp_idx], random_state=SEED)
    train_idx, val_idx, test_idx = map(torch.from_numpy, (train_idx, val_idx, test_idx))

    print("=== 실험 B: Text(feature)-only vs +Behavior Graph (YelpChi) ===")
    print(f"nodes={num_nodes}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n")

    print("[1] feature-only MLP (그래프 없음, text-derived feature만)")
    mlp = MLP(in_dim=features.shape[1])
    mlp_test = train_model(mlp, x, y, adjs=None, train_idx=train_idx, val_idx=val_idx,
                            test_idx=test_idx, use_graph=False)
    print(f"  -> test acc={mlp_test['acc']:.4f}  f1={mlp_test['f1']:.4f}  auc={mlp_test['auc']:.4f}\n")

    print("[2] + Reviewer Behavior Graph (R-U-R / R-T-R / R-S-R, 경량 relational GNN)")
    adjs = [normalize_adj(d[k], num_nodes) for k in ["net_rur", "net_rtr", "net_rsr"]]
    gnn = RelationalGNN(in_dim=features.shape[1], hidden_dim=64, num_relations=3)
    gnn_test = train_model(gnn, x, y, adjs=adjs, train_idx=train_idx, val_idx=val_idx,
                            test_idx=test_idx, use_graph=True)
    print(f"  -> test acc={gnn_test['acc']:.4f}  f1={gnn_test['f1']:.4f}  auc={gnn_test['auc']:.4f}\n")

    print("=== 요약 ===")
    print(f"{'setting':30s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    print(f"{'feature-only (text)':30s} {mlp_test['acc']:8.4f} {mlp_test['f1']:8.4f} {mlp_test['auc']:8.4f}")
    print(f"{'+ behavior graph':30s} {gnn_test['acc']:8.4f} {gnn_test['f1']:8.4f} {gnn_test['auc']:8.4f}")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "acc", "f1", "auc"])
        w.writerow(["feature-only (text)", mlp_test["acc"], mlp_test["f1"], mlp_test["auc"]])
        w.writerow(["+ behavior graph", gnn_test["acc"], gnn_test["f1"], gnn_test["auc"]])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
