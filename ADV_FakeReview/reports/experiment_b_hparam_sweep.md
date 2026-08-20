# 실험 B 추가 확인: Self-MM+MISA가 plain baseline보다 낮은 게 하이퍼파라미터 탓인가?

## 배경

`experiment_b_fusion_results.csv` 결과를 보면, 실험 A(full-mode, BERT+ResNet+CLIP)와
달리 실험 B에서는 fusion 메커니즘(Self-MM, MISA-lite)을 추가할수록 오히려 f1이
떨어졌다.

| 세팅 | acc | f1 | auc |
|---|---|---|---|
| (1) plain GNN (fusion 없음) | 0.9106 | **0.7189** | 0.9413 |
| (2) + Self-MM | 0.8904 | 0.6897 | 0.9449 |
| (3) + Self-MM + MISA-lite | 0.8988 | 0.7038 | 0.9453 |

`aux_weight=0.3`, `ortho_weight=0.1`은 튜닝한 값이 아니라 상식적인 기본값이었어서,
"하이퍼파라미터를 잘못 골라서 손해를 본 것 아닌가"를 확인해볼 필요가 있었다.

## 확인 방법

`src/behavior_gnn_fusion_sweep.py`: (3) + Self-MM + MISA-lite 세팅을 고정하고
`(aux_weight, ortho_weight)` 조합 3개(기본값 포함)로만 가볍게 비교했다. 그리드서치가
아니라 "결과가 특정 값 하나에만 우연히 맞아떨어진 게 아니다"를 보는 robustness
check.

## 결과

| aux_weight | ortho_weight | acc | f1 | auc |
|---|---|---|---|---|
| 0.1 | 0.1 | 0.9043 | 0.7111 | 0.9447 |
| 0.3 | 0.1 (기본값) | 0.8988 | 0.7038 | 0.9453 |
| 0.5 | 0.3 | 0.9019 | 0.7062 | 0.9426 |

acc 편차 0.0054, f1 편차 0.0073로 세 조합이 거의 차이가 없고, 셋 다 plain
baseline(f1 0.7189)보다는 낮다.

## 결론

**Self-MM+MISA가 실험 B에서 plain baseline보다 낮게 나오는 건 하이퍼파라미터를
잘못 골라서가 아니라 일관된 패턴이다.** 우연이 아니라는 게 확인됐으니, 오히려
이유를 설명하는 게 중요해졌다:

실험 B의 두 branch(`feature_branch`, `graph_branch`)는 실험 A의 text/image처럼
완전히 독립적인 modality가 아니다. `graph_branch`는 `feature_branch`가 보는 것과
같은 원본 feature를 그래프로 전파(propagate)한 것이라, 두 branch가 이미 상당히
겹치는 정보를 담고 있다(`behavior_gnn_fusion.py` 파일 상단 주석에도 이 한계를
미리 명시해뒀음). MISA의 shared/specific 분리나 Self-MM의 branch별 auxiliary
loss는 원래 "서로 다른 정보를 가진 modality를 어떻게 잘 합칠까"를 위한
장치인데, 애초에 두 branch가 비슷한 정보를 담고 있으면 이 분리 자체가 불필요한
제약이 되어 오히려 손해를 볼 수 있다.

## 발표 포인트

이건 "실패한 실험"이 아니라 오히려 더 설득력 있는 스토리다:

- 실험 A(진짜 독립적인 text/image modality) → Self-MM+MISA가 도움이 됨
- 실험 B(겹치는 정보를 가진 feature/graph pseudo-modality) → Self-MM+MISA가
  도움이 안 됨, 그리고 이게 하이퍼파라미터 탓이 아니라는 것도 확인함

**"fusion 메커니즘은 modality가 진짜로 서로 다른/보완적인 정보를 담고 있을 때
효과가 있고, 정보가 겹치는 경우엔 오히려 불필요한 제약이 된다"**는 일반화된
결론을 두 실험의 대비로 보여줄 수 있다. 이건 처음에 우리가 세운 최종 비전
구조(text/image/behavior 3-modality)를 정당화하는 데도 도움이 된다 — text,
image, reviewer behavior는 실제로 서로 다른 종류의 정보이므로, 이 결과에 따르면
fusion 메커니즘이 거기서는 효과를 낼 가능성이 높다는 근거가 되기 때문이다.
