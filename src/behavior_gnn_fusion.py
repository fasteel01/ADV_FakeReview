"""
실험 B 확장판: Self-MM / MISA 아이디어를 실제로 얹어서 Day1 baseline과 비교.

Day1(src/behavior_gnn.py)은 "feature-only vs +graph" 두 세팅만 비교했다.
여기서는 "+graph" 세팅을 두 개의 pseudo-modality(branch)로 보고
Self-MM/MISA 아이디어를 적용한다:

  - branch 1 = feature representation (그래프 없이 raw feature만 encoding)
  - branch 2 = graph representation   (RelationalGNN으로 관계 정보까지 encoding)

주의: 이 둘은 원래 논문들이 상정하는 "완전히 독립적인 modality"(text/image/audio)와는
다르다 (branch 2가 branch 1 정보를 그래프로 전파한 것이라 완전히 독립적이지 않음).
그래서 이건 "최종 3-modality 모델에 넣을 fusion 메커니즘을 지금 확보 가능한 데이터로
미리 프로토타이핑/검증"하는 용도로 쓰는 것이고, 실제 text/image/behavior 3개 modality
버전은 src/baseline_textimage.py의 run_full()에서 Colab으로 검증한다.

비교 대상:
  (1) Day1 baseline: plain RelationalGNN (fusion 메커니즘 없음)
  (2) + Self-MM: 위 (1)에 branch별 auxiliary classification loss만 추가
  (3) + Self-MM + MISA-lite: branch representation을 shared/specific으로 나눠서 fusion
"""
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from models.simple_gnn import RelationalGNN, RelationalGNNLayer, normalize_adj
from models.fusion_modules import BranchEncoder, SelfMMAuxHead, MISALiteFusion

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "yelpchi", "YelpChi.mat")
SEED = 42
COMBINED_DIM = 64


class GraphBranch(nn.Module):
    """RelationalGNN을 hidden representation만 뽑도록 감싼 wrapper"""

    def __init__(self, in_dim, hidden_dim, num_relations, num_layers=2, dropout=0.3):
        super().__init__()
        self.gnn = RelationalGNN(in_dim, hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, adjs):
        return self.gnn.forward_repr(x, adjs)


class FusionModel(nn.Module):
    """use_aux / use_misa 플래그로 Self-MM / MISA-lite를 켜고 끌 수 있게 만든 통합 모델"""

    def __init__(self, in_dim, num_relations, use_aux=True, use_misa=True):
        super().__init__()
        self.use_aux = use_aux
        self.use_misa = use_misa

        self.feature_branch = BranchEncoder(in_dim, COMBINED_DIM)
        self.graph_branch = GraphBranch(in_dim, COMBINED_DIM, num_relations)

        if use_misa:
            self.fusion = MISALiteFusion(COMBINED_DIM, shared_dim=32, specific_dim=32, num_branches=2)
            fused_dim = 32 * 2 * 2  # shared(2 branch) + specific(2 branch)
        else:
            fused_dim = COMBINED_DIM * 2  # 그냥 concat

        self.classifier = nn.Linear(fused_dim, 2)

        if use_aux:
            self.aux_feature = SelfMMAuxHead(COMBINED_DIM)
            self.aux_graph = SelfMMAuxHead(COMBINED_DIM)

    def forward(self, x, adjs):
        feat_repr = self.feature_branch(x)
        graph_repr = self.graph_branch(x, adjs)

        ortho_loss = torch.tensor(0.0)
        if self.use_misa:
            fused, ortho_loss = self.fusion([feat_repr, graph_repr])
        else:
            fused = torch.cat([feat_repr, graph_repr], dim=-1)

        main_logits = self.classifier(fused)

        aux_logits = None
        if self.use_aux:
            aux_logits = (self.aux_feature(feat_repr), self.aux_graph(graph_repr))

        return main_logits, aux_logits, ortho_loss


def evaluate(logits, y, mask):
    proba = F.softmax(logits[mask], dim=1)[:, 1].detach().numpy()
    pred = logits[mask].argmax(dim=1).detach().numpy()
    y_true = y[mask].numpy()
    return {
        "acc": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba),
    }


def train_model(model, x, y, adjs, train_idx, val_idx, test_idx,
                 epochs=100, lr=0.01, aux_weight=0.3, ortho_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    n_pos = y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

    best_val_f1, best_test = -1, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        main_logits, aux_logits, ortho_loss = model(x, adjs)
        loss = F.cross_entropy(main_logits[train_idx], y[train_idx], weight=weight)
        if aux_logits is not None:
            for al in aux_logits:
                loss = loss + aux_weight * F.cross_entropy(al[train_idx], y[train_idx], weight=weight)
        loss = loss + ortho_weight * ortho_loss

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            main_logits, _, _ = model(x, adjs)
            val_metrics = evaluate(main_logits, y, val_idx)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_test = evaluate(main_logits, y, test_idx)

        if (epoch + 1) % 25 == 0:
            print(f"    epoch {epoch+1:3d}  loss={loss.item():.4f}  val_f1={val_metrics['f1']:.4f}")

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
    adjs = [normalize_adj(d[k], num_nodes) for k in ["net_rur", "net_rtr", "net_rsr"]]

    idx = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=label, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=label[temp_idx], random_state=SEED)
    train_idx, val_idx, test_idx = map(torch.from_numpy, (train_idx, val_idx, test_idx))

    settings = [
        ("(1) Day1 baseline: plain GNN (fusion 메커니즘 없음)", dict(use_aux=False, use_misa=False)),
        ("(2) + Self-MM (branch별 auxiliary loss)", dict(use_aux=True, use_misa=False)),
        ("(3) + Self-MM + MISA-lite (shared/specific 분리)", dict(use_aux=True, use_misa=True)),
    ]

    results = {}
    print("=== 실험 B 확장: Self-MM / MISA-lite fusion ablation (YelpChi) ===\n")
    for name, kwargs in settings:
        print(f"[{name}]")
        torch.manual_seed(SEED)
        model = FusionModel(in_dim=features.shape[1], num_relations=3, **kwargs)
        test_metrics = train_model(model, x, y, adjs, train_idx, val_idx, test_idx)
        results[name] = test_metrics
        print(f"  -> test acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}\n")

    print("=== 요약 ===")
    print(f"{'setting':55s} {'acc':>8s} {'f1':>8s} {'auc':>8s}")
    for name, m in results.items():
        print(f"{name:55s} {m['acc']:8.4f} {m['f1']:8.4f} {m['auc']:8.4f}")

    import csv
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_b_fusion_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "acc", "f1", "auc"])
        for name, m in results.items():
            w.writerow([name, m["acc"], m["f1"], m["auc"]])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
