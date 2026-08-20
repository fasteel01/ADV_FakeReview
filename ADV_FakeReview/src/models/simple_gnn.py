"""
PyTorch Geometric 없이 순수 PyTorch(sparse tensor)로 구현한 경량 GNN.

이 샌드박스에서는 torch-geometric의 확장 wheel(torch-scatter/torch-sparse)이
download.pytorch.org에서 받아지지 않아 설치가 안 되므로, 직접 sparse matmul로
message passing을 구현했다. FraudSquad가 쓰는 gated graph transformer보다는
훨씬 단순하지만(2-layer mean-aggregation GCN 스타일), "그래프 관계를 쓰면
text/feature-only보다 성능이 오르는가"라는 질문에 답하기엔 충분하다.

멀티 관계(R-U-R, R-T-R, R-S-R)는 relation별로 따로 aggregate한 뒤 concat하는
방식으로 처리한다 (FraudSquad/CARE-GNN이 여러 relation을 따로 다루는 것과 동일한 아이디어).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj_coo, num_nodes):
    """symmetric mean-normalized sparse adjacency (self-loop 포함) -> torch sparse tensor"""
    adj_coo = adj_coo.tocoo()
    row = torch.from_numpy(adj_coo.row.astype("int64"))
    col = torch.from_numpy(adj_coo.col.astype("int64"))
    # add self loops
    self_idx = torch.arange(num_nodes, dtype=torch.int64)
    row = torch.cat([row, self_idx])
    col = torch.cat([col, self_idx])
    values = torch.ones(row.shape[0])
    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    norm_values = deg_inv[adj.indices()[0]]
    adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_values, (num_nodes, num_nodes)).coalesce()
    return adj_norm


class RelationalGNNLayer(nn.Module):
    """여러 relation 그래프에 대해 각각 mean-aggregation 후 concat, 그다음 linear projection"""

    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_rel = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])

    def forward(self, x, adjs):
        out = self.lin_self(x)
        for adj, lin in zip(adjs, self.lin_rel):
            agg = torch.sparse.mm(adj, x)
            out = out + lin(agg)
        return F.relu(out)


class RelationalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, num_layers=2, dropout=0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            RelationalGNNLayer(dims[i], dims[i + 1], num_relations) for i in range(num_layers)
        ])
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward_repr(self, x, adjs):
        """최종 classifier 이전의 hidden representation을 반환.
        (fusion_modules.py의 MISA-lite/Self-MM aux head에서 사용)"""
        h = x
        for layer in self.layers:
            h = layer(h, adjs)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, adjs):
        return self.classifier(self.forward_repr(x, adjs))
