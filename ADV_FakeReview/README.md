# ADV_FakeReview

Text + Image + Reviewer Behavior 멀티모달 가짜 리뷰 탐지 리서치 (2026 ADV Session)

## 프로젝트 개요

리뷰 텍스트, 리뷰 이미지, 작성자(reviewer)의 행동 패턴을 함께 사용해 가짜 리뷰를 탐지하는 모델을 연구한다. 최종 비전 구조와 논문별 설계 근거는 `reports/architecture_rationale.md`를 참고. 이 리포는 8/26 발표를 목표로 한 실증 실험 코드를 담는다.

3개 modality가 동시에 갖춰진 공개 데이터셋이 존재하지 않기 때문에(자세한 근거는 `reports/architecture_rationale.md`), 두 개의 독립적인 실험으로 나눠 각 modality의 기여도를 따로 검증한다.

- **실험 A (Text + Image)**: [AiGen-FoodReview](https://github.com/iamalegambetti/aigen-foodreview) 데이터셋 위에서 텍스트/이미지 특성이 fake review 탐지에 기여하는지 검증
- **실험 B (Text + Reviewer Behavior)**: [YelpChi](https://github.com/YingtongDou/CARE-GNN) 데이터셋 위에서 reviewer-review-product 그래프 구조가 기여하는지 검증
- **실험 C (난이도 대조군)**: [Ott et al. Deceptive Opinion Spam Corpus](https://github.com/PauDK/Deceptive-Review-Detection) 위에서 같은 text 파이프라인을 돌려서, AiGen-FoodReview(LLM이 대필한 "쉬운" 가짜)와 사람이 직접 쓴 "어려운" 가짜 리뷰 사이의 난이도 차이를 정량적으로 비교
- **실험 D (종료 - 실현 불가 확인)**: "사람이 쓴 어려운 가짜 + 실제 이미지"가 둘 다 있는 공개 데이터셋이 없어서, [Hollenbeck et al. 가짜/진짜 라벨 데이터](https://github.com/bretthollenbeck/fake-reviews-data)와 [Amazon Reviews 2023 실제 리뷰 이미지](https://amazon-reviews-2023.github.io/)를 직접 조인해서 만들어보려 시도했으나, 실제 스키마를 확인해보니 Hollenbeck 데이터가 리뷰 단위가 아닌 "상품×주" 집계 패널 데이터라 조인에 필요한 ASIN 컬럼 자체가 없어 실현 불가능함을 확인했다. 자세한 내용(조사 과정, 근거)은 `reports/experiment_d_plan.md` 참고.

세 데이터셋을 합치면 쉬움(AiGen-FoodReview, LLM 생성) → 보통(YelpChi, Yelp 자체 필터 라벨) → 어려움(Ott, 사람이 작정하고 쓴 기만 리뷰)으로 이어지는 난이도 스펙트럼이 만들어진다. 자세한 내용은 `reports/experiment_c_findings.md` 참고.

## ⚠️ 중요: 실행 환경별 제약

이 코드는 클라우드 샌드박스(Claude Cowork)에서 최초 작성되었고, 그 환경은 `huggingface.co`, `zenodo.org`, `drive.google.com`, `download.pytorch.org` 등이 네트워크 차단되어 있다(허용된 도메인만 접속 가능한 proxy 환경). 반면 `github.com`과 `pypi.org`는 접속 가능하다. 이 제약 때문에:

- **사전학습 가중치(BERT/ResNet/CLIP)를 다운로드해야 하는 코드는 이 샌드박스에서 실행 불가능하다.** 로컬 컴퓨터, Colab, 학교 서버(GPU 있고 인터넷 열려있는 환경)처럼 HuggingFace Hub에 접속 가능한 환경에서 실행해야 한다.
- 대신 지금 당장 실행 가능한 **handcrafted feature 기반 baseline**(`src/baseline_textimage.py`의 `--mode lite`)과 **from-scratch로 학습하는 경량 GNN**(`src/behavior_gnn.py`, `src/behavior_gnn_fusion.py`)은 사전학습 가중치가 필요 없어서 샌드박스에서도 바로 돌아가고, 실제로 Day1~2에 이 리포 안에서 실행해서 결과를 얻었다.
- `--mode full`(BERT+ResNet+CLIP 사용)은 인터넷이 열려있는 환경에서 `pip install -r requirements-full.txt` 후 실행할 것. 아래 "학교 서버/Colab에서 실행하기" 참고.

이건 설계 결함이 아니라 실행 환경 제약이므로, 인터넷이 열려있는 환경에서 돌리면 `--mode full` 경로가 정상 작동한다.

## Self-MM / MISA-lite / CLIP consistency (MMIM 대체) 구현 현황

`src/models/fusion_modules.py`에 두 가지 fusion 메커니즘을 구현했고, 실험 A/B 양쪽에서 재사용한다.

- **Self-MM 아이디어**: 각 modality branch에 자체 auxiliary classification head를 달아서, fusion 이전에도 각 branch가 유의미한 정보를 담도록 학습을 유도 (`SelfMMAuxHead`)
- **MISA 아이디어(경량화)**: 모든 branch가 공유하는 linear layer(shared, modality-invariant)와 branch별 linear layer(specific)로 나눠서 fusion. 원 논문의 CMD loss 대신 orthogonality loss(내적 최소화)로 shared/specific을 분리 (`MISALiteFusion`)
- **MMIM 아이디어**: mutual information estimator를 직접 구현하는 대신, CLIP text-image cosine similarity를 명시적 feature로 사용 (`src/models/multimodal_model.py`의 `extract_features()`)
- **MAG**: 구현하지 않음. BERT 내부 hook이 필요해 리스크 대비 이득이 불확실 (`reports/architecture_rationale.md` 4절 참고)
- **MPLMM-lite**: 실험 B에 추가로 적용. YelpChi의 "신규 유저(그래프 정보 없음)"를 결측 modality 상황으로 보고, naive(0 채움) → static prompt(GNN 입력단, 개선 미미) → **conditional prompt(feat_repr 기반 노드별 생성, 뚜렷한 개선)** 순서로 3단계에 걸쳐 발전시킴. 왜 각 단계가 개선/미개선됐는지까지 구조적으로 설명 가능. 전체 정리는 `reports/experiment_b_missing_modality_v3_findings.md` 참고

실험 B(`src/behavior_gnn_fusion.py`)에서 이 메커니즘을 "text-feature branch vs graph branch"라는 두 pseudo-modality에 먼저 프로토타이핑했고, 실험 A(`src/baseline_textimage.py --mode full`)에서 진짜 text/image 두 modality에 동일한 메커니즘을 적용한다.

## 학교 서버 / Colab에서 실행하기 (--mode full)

이미지 원본과 사전학습 가중치가 필요한 부분은 인터넷이 열려있는 GPU 환경에서 실행한다.

```bash
git clone https://github.com/fasteel01/ADV_FakeReview.git
cd ADV_FakeReview
bash scripts/download_aigen.sh          # 텍스트+handcrafted feature CSV
bash scripts/download_yelpchi.sh        # YelpChi.mat
bash scripts/download_aigen_images.sh   # AiGen-FoodReview 원본 이미지 (Google Drive, gdown 사용)
pip install -r requirements-full.txt

python src/baseline_textimage.py --mode full
```

`--mode full`은 내부적으로 BERT/ResNet-50/CLIP으로 feature를 한 번 추출해서 `reports/features_cache/*.npz`에 캐싱하고(처음 한 번만 느림), 그 위에서 4가지 fusion 설정(concat only / +CLIP consistency / +Self-MM / +Self-MM+MISA-lite)을 비교한다. 결과는 `reports/experiment_a_full_results.csv`.

## 데이터 출처

| 데이터 | 원 출처 | 실제로 받아온 경로 | modality |
|---|---|---|---|
| AiGen-FoodReview | Gambetti & Han, ICWSM 2024 (Zenodo: https://zenodo.org/records/10511456) | GitHub 미러: https://github.com/iamalegambetti/aigen-foodreview (data/*.csv에 텍스트+라벨+24개 handcrafted text/image feature 포함, 원본 이미지 파일 자체는 Google Drive에 있어 이번엔 못 받음) | Text + Image(handcrafted feature) |
| YelpChi | Rayana & Akoglu, KDD 2015 / CARE-GNN(Dou et al., CIKM 2020) 전처리본 | GitHub: https://github.com/YingtongDou/CARE-GNN (data/YelpChi.zip → YelpChi.mat) | Text(handcrafted feature) + Reviewer/Product 관계 그래프(R-U-R, R-T-R, R-S-R) |

라이선스/재배포 조건이 있는 데이터라 원본 raw 파일은 git에 커밋하지 않았다(`data/raw/`는 `.gitignore` 대상). `data/sample/`에 스모크 테스트용 50행짜리 샘플만 포함. 전체 데이터는 `scripts/download_*.sh`로 받는다.

```bash
bash scripts/download_aigen.sh      # data/raw/*.csv
bash scripts/download_yelpchi.sh    # data/raw/yelpchi/YelpChi.mat
```

## 실행 순서

```bash
pip install -r requirements.txt

# EDA
python src/eda_aigen.py
python src/eda_yelpchi.py

# 실험 A: Text+Image baseline (lite = handcrafted feature, 샌드박스에서 즉시 실행 가능)
python src/baseline_textimage.py --mode lite

# 실험 B: Text+Behavior graph baseline
python src/behavior_gnn.py

# 실험 B 확장: Self-MM + MISA-lite ablation (샌드박스에서도 실행 가능, 사전학습 가중치 불필요)
python src/behavior_gnn_fusion.py

# 실험 C: 난이도 대조군 (Ott Deceptive Opinion Spam, 사람이 쓴 가짜 리뷰)
bash scripts/download_opspam.sh
python src/baseline_opspam.py

# 실험 D (종료됨 - 조인 불가로 확인, 조사 기록용으로만 남김. reports/experiment_d_plan.md 참고)
# bash scripts/download_amazon_hard.sh
# python src/build_amazon_hard_dataset.py --explore
```

결과와 해석은 `reports/day1_findings.md`, `reports/experiment_b_fusion_results.csv` 참고.

## 디렉토리 구조

```
ADV_FakeReview/
├── data/
│   ├── raw/            # 다운로드한 원본 데이터 (git 미포함)
│   └── sample/          # 스모크 테스트용 소량 샘플 (git 포함)
├── scripts/             # 데이터 다운로드 스크립트
├── src/
│   ├── eda_aigen.py
│   ├── eda_yelpchi.py
│   ├── baseline_textimage.py       # 실험 A (lite: 샌드박스용, full: BERT+ResNet+CLIP, 학교서버/Colab용)
│   ├── behavior_gnn.py             # 실험 B (Day1 baseline)
│   ├── behavior_gnn_fusion.py      # 실험 B 확장 (Self-MM + MISA-lite ablation)
│   ├── baseline_opspam.py          # 실험 C (난이도 대조군, Ott)
│   ├── build_amazon_hard_dataset.py # 실험 D (종료됨, 조사 기록용)
│   ├── behavior_gnn_fusion_sweep.py # 실험 B aux_weight/ortho_weight robustness check
│   ├── behavior_gnn_missing_modality.py # 실험 B + MPLMM v1 (분류기 직전 대체, 개선 없음)
│   ├── behavior_gnn_missing_modality_v2.py # v2 (GNN 입력단 static prompt, 미미한 개선)
│   ├── behavior_gnn_missing_modality_v3.py # v3 (feat_repr 기반 conditional prompt, 뚜렷한 개선)
│   └── models/
│       ├── simple_gnn.py           # PyG 없이 순수 PyTorch로 구현한 경량 GNN
│       ├── fusion_modules.py       # Self-MM / MISA-lite 공용 모듈
│       └── multimodal_model.py     # BERT+ResNet+CLIP feature 추출 + fusion 모델 (full 모드용)
└── reports/
    ├── architecture_rationale.md   # 전체 구조 설계 근거 (논문별 채택/제외 이유)
    ├── day1_findings.md            # Day1 실행 결과
    ├── experiment_c_findings.md    # 실험 C 결과 및 해석
    ├── experiment_d_plan.md        # 실험 D 조사 기록 (종료 - 실현 불가 확인)
    ├── experiment_b_hparam_sweep.md # 실험 B aux_weight/ortho_weight robustness check
    ├── experiment_b_missing_modality.md # v1 상세 결과 (분류기 직전 대체)
    ├── experiment_b_missing_modality_v3_findings.md # v1→v2→v3 종합 정리 (결측 modality 처리)
    └── 실험결과_쉽게정리.md          # 쉬운 말로 정리한 결과 요약
```
