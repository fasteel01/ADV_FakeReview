# Fusion 메커니즘 구현 결정 (Self-MM / MISA / MMIM / MAG / MPLMM)

Day1 이후 사용자 요청으로 "아이디어만 인용"에서 "가능한 만큼 실제 구현"으로 범위를 확장했다. 이 문서는 5개 논문 각각에 대해 무엇을 구현했고, 왜 그렇게 결정했는지 기록한다.

## 구현함

**Self-MM → `SelfMMAuxHead`** (`src/models/fusion_modules.py`)

원 논문의 self-supervised unimodal label generation(감정 강도 회귀용)은 이진분류(fake/real)에 그대로 옮기기 애매해서 제외했다. 대신 핵심 아이디어인 "각 modality branch가 자기 정보만으로도 어느 정도 분류할 수 있어야 한다"를 auxiliary classification head + auxiliary loss로 구현했다. 구현 비용이 거의 없고(Linear layer 하나), fusion 이전 단계에서 각 branch가 붕괴하지 않도록 정규화 효과도 있다.

**MISA → `MISALiteFusion`** (`src/models/fusion_modules.py`)

원 논문은 modality-invariant/specific representation을 CMD(Central Moment Discrepancy) loss + 여러 보조 loss로 분리한다. 여기서는 (1) 모든 branch가 공유하는 linear layer로 shared(invariant) representation을 만들고, (2) branch별 linear layer로 specific representation을 만든 뒤, (3) 같은 branch의 shared/specific이 서로 다른 정보를 담도록 orthogonality loss(코사인 유사도 제곱의 최소화)로 정규화하는 경량 버전을 구현했다. CMD보다 구현·학습이 훨씬 간단하고 안정적이다.

**MMIM → CLIP cosine similarity feature** (`src/models/multimodal_model.py`)

mutual information estimator(CLUB/NCE 계열)는 구현 난이도와 학습 불안정성이 높아 별도 구현하지 않았다. 대신 "modality 간 관계 자체를 feature로 쓴다"는 MMIM의 목적을, CLIP text-image cosine similarity를 명시적 입력 feature로 추가하는 방식으로 달성했다. 사전학습된 CLIP을 그대로 쓰기 때문에 추가 학습이 필요 없다.

## 구현하지 않음

**MAG (Multimodal Adaptation Gate)**

BERT/XLNet의 internal layer(특정 encoder layer의 hidden state)에 다른 modality 정보를 gate로 주입하는 구조라, HuggingFace의 기본 forward pass를 hook이나 커스텀 forward로 직접 수정해야 한다. 구현 리스크(디버깅 어려움, 학습 불안정 가능성)에 비해 이 프로젝트 스케일에서 fusion 성능 이득이 클 것으로 보이지 않아 제외했다. 발표에서는 "검토했으나 구현 리스크 대비 이득이 불확실해 제외, 향후 확장 후보"로 한 줄 언급하는 것을 추천한다.

**MPLMM (missing modality prompt learning)**

이번 실험(A: text+image, B: text-feature+graph) 모두 두 modality가 항상 존재하는 세팅이라, missing modality 상황 자체가 데이터에 없다. 즉 지금 가진 데이터로는 MPLMM을 적용/검증할 대상이 없다. 실서비스에서는 이미지 없는 리뷰, 이력이 부족한 신규 유저처럼 modality 결측이 흔히 발생하므로, "최종 3-modality 모델을 실서비스에 적용할 때는 MPLMM류의 missing-modality-robust 기법으로 확장 가능하다"는 문장으로 발표 마지막 "향후 연구" 슬라이드에 넣는 것을 추천한다.

## 검증 순서

1. `src/behavior_gnn_fusion.py`: "text-feature branch vs graph branch"라는 두 pseudo-modality에 Self-MM/MISA-lite를 먼저 적용해서, 사전학습 가중치 없이 지금 당장(샌드박스에서도) 검증. → 결과는 `reports/experiment_b_fusion_results.csv`
2. `src/baseline_textimage.py --mode full`: 실제 text/image 두 modality(BERT+ResNet embedding)에 동일한 메커니즘(+CLIP consistency)을 적용. 학교 서버/Colab에서 실행 필요. → 결과는 `reports/experiment_a_full_results.csv`

두 실험 모두 "concat만" vs "+Self-MM" vs "+Self-MM+MISA-lite" 순서로 쌓아가며 비교하도록 설계해서, 어떤 메커니즘이 실제로 기여하는지 발표에서 단계별로 보여줄 수 있다.
