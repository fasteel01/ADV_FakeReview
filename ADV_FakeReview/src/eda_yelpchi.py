"""
YelpChi EDA
- 라벨 분포, feature 통계, 관계 그래프(R-U-R / R-T-R / R-S-R) 밀도 및
  "이웃 라벨이 자기 라벨과 얼마나 다른가"(homophily) 확인
  -> REAL/FraudSquad가 강조하는 "fraudster는 개별로 보면 정상처럼 보이지만
     관계로 보면 다르다(camouflage)"는 주장을 이 데이터에서도 재확인
"""
import os
import numpy as np
import scipy.io as sio

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "yelpchi", "YelpChi.mat")


def edge_label_homophily(adj, label):
    """인접한 두 노드가 같은 라벨을 가질 확률 (label homophily ratio)"""
    adj = adj.tocoo()
    same = (label[adj.row] == label[adj.col]).sum()
    total = len(adj.row)
    return same / total if total > 0 else float("nan")


def main():
    d = sio.loadmat(DATA_PATH)
    label = d["label"].flatten()
    features = d["features"]

    print("=== YelpChi EDA ===")
    print(f"nodes(reviews)={features.shape[0]}, feature_dim={features.shape[1]}")
    vals, counts = np.unique(label, return_counts=True)
    print("라벨 분포:", dict(zip(vals.tolist(), counts.tolist())))
    fraud_ratio = counts[vals == 1].sum() / counts.sum()
    print(f"fraud 비율: {fraud_ratio:.3%}")

    print("\n관계 그래프별 엣지 수 및 label homophily:")
    for key, desc in [
        ("net_rur", "R-U-R (같은 유저가 쓴 리뷰끼리)"),
        ("net_rtr", "R-T-R (같은 상품, 같은 달에 올라온 리뷰끼리)"),
        ("net_rsr", "R-S-R (같은 상품, 같은 별점 리뷰끼리)"),
        ("homo", "전체 관계 합집합"),
    ]:
        adj = d[key]
        homophily = edge_label_homophily(adj, label)
        print(f"  {key:10s} {desc:35s} edges={adj.nnz:>9,}  label_homophily={homophily:.3f}")

    print(
        "\n해석: homophily가 1에 가까우면 '연결된 리뷰끼리 같은 라벨(정상-정상, "
        "fraud-fraud)'일 확률이 높다는 뜻 -> 그래프 구조 자체가 라벨과 상관관계를 "
        "가진다 -> Reviewer Behavior(그래프) modality가 실제로 신호를 담고 있다는 "
        "1차 근거로 사용 가능."
    )


if __name__ == "__main__":
    main()
