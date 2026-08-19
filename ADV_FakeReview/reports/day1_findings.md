# Day 1 결과 (2026-08-18 실행)

## 실행 환경 노트

이 실행은 Claude Cowork 클라우드 샌드박스에서 진행됐다. 이 환경은 `huggingface.co`, `zenodo.org`, `drive.google.com`, `download.pytorch.org`가 네트워크 차단되어 있고 `github.com`, `pypi.org`만 접속 가능하다. 그래서:

- 데이터는 원 출처(Zenodo, 학교 서버) 대신 **GitHub에 미러링된 버전**을 사용했다 (`README.md`의 데이터 출처 표 참고). AiGen-FoodReview는 원 논문 저자의 GitHub 저장소(`iamalegambetti/aigen-foodreview`)에 CSV가 직접 커밋되어 있어서 문제없이 받았지만, **원본 이미지 파일 자체(Google Drive)는 이번엔 받지 못했다** — 대신 논문이 이미 계산해둔 24개 image handcrafted feature(색감/구도/aesthetic score)로 이미지 modality를 대체했다.
- BERT/ResNet/CLIP 같은 사전학습 가중치가 필요한 파이프라인(`--mode full`)은 이 환경에서 실행이 불가능해서, 이번 Day1은 **handcrafted feature 기반 baseline**(실험 A)과 **from-scratch로 학습하는 경량 GNN**(실험 B)만 실행했다. 이 둘은 사전학습 가중치가 필요 없어 이 샌드박스에서 완결적으로 돌아간다.
- `--mode full` 경로는 로컬 컴퓨터나 Colab에서 실행하면 된다(코드는 이미 스켈레톤으로 작성되어 있음, `src/baseline_textimage.py`의 `run_full()`).

## 실험 A: Text + Image (AiGen-FoodReview, n=20,144)

`python src/baseline_textimage.py --mode lite`

| setting | test acc | test f1 | test auc |
|---|---|---|---|
| text-only (TF-IDF) | 0.9968 | 0.9968 | 1.0000 |
| text_feat-only (readability/ppl) | 0.9499 | 0.9505 | 0.9890 |
| image-only (aesthetic/color feats) | 0.7829 | 0.7854 | 0.8619 |
| text + image | 0.9975 | 0.9976 | 0.9999 |
| text + text_feat + image (combined) | 0.9916 | 0.9917 | 0.9994 |

**해석**

- 이미지만으로도 (raw pixel 없이 24개 handcrafted feature만 갖고도) test AUC 0.86을 낼 수 있다. 즉 **이미지 modality 자체는 확실히 fake review와 상관관계가 있는 신호를 담고 있다** — CLIP consistency 같은 더 정교한 feature를 쓰면 이보다 더 오를 여지가 있다는 뜻.
- 다만 text-only가 이미 acc 0.997/auc 1.000으로 거의 포화 상태라, text+image 조합이 text-only 대비 눈에 띄게 개선되지는 않는다(ceiling effect). 이건 이 데이터셋의 fake 라벨이 "사람이 작정하고 쓴 기만적 리뷰"가 아니라 **"GPT-4가 대필한 리뷰"**라서, 어휘적으로 이미 사람 글과 뚜렷이 구분되기 때문이다(TF-IDF만으로 거의 다 잡힘). AiGen-FoodReview 발표에서는 이 점을 반드시 명시해야 한다 — "우리가 검증한 건 LLM-generated review 탐지이고, 전통적 의미의 고용된 가짜 리뷰 탐지와는 태스크 난이도가 다르다"는 caveat.
- 발표 스토리로 쓰기 좋은 포인트: "이미지 단독으로도 상당한 판별력이 있다(AUC 0.86) → 이미지가 fake review 탐지에 독립적인 신호를 제공한다는 근거는 이 실험으로 이미 확보했고, 전체 라벨이 포화된 이 데이터셋보다 더 어려운(사람이 쓴 기만적 리뷰) 세팅에서 CLIP consistency의 가치는 더 커질 것으로 기대한다."

## 실험 B: Text(feature) vs + Reviewer Behavior Graph (YelpChi, n=45,954)

`python src/behavior_gnn.py`

| setting | test acc | test f1 | test auc |
|---|---|---|---|
| feature-only (text-derived feature, 그래프 없음) | 0.8268 | 0.5518 | 0.8733 |
| + behavior graph (R-U-R/R-T-R/R-S-R, 경량 GNN) | **0.8934** | **0.6895** | **0.9416** |

**해석**

- Reviewer-Review-Product 관계 그래프를 추가하니 f1이 0.552 → 0.690 (+25% 상대 향상), AUC가 0.873 → 0.942로 뚜렷하게 개선됐다. **Reviewer Behavior modality가 실제로 강한 신호를 담고 있다는 게 정량적으로 확인됨.**
- EDA에서도 이 결과를 뒷받침하는 근거가 있다: R-U-R(같은 유저) 관계의 label homophily가 0.996으로 거의 1에 가깝다 — 즉 같은 유저가 쓴 리뷰끼리는 거의 항상 같은 라벨(정상 or fraud)이다. R-T-R/R-S-R(같은 상품 기준 관계)도 homophily 0.76~0.77로 무작위보다 훨씬 높다. 이게 바로 FraudSquad/REAL이 강조하는 "개별 리뷰만 보면 정상처럼 보이는 fraudster도 관계로 보면 패턴이 드러난다"는 주장의 실증 근거다.
- 이 실험은 원 텍스트가 아니라 텍스트에서 추출된 32-dim handcrafted feature를 썼다는 한계가 있다(원 raw 텍스트 재구성 불가 데이터). 로컬 환경에서 BERT embedding을 이 그래프에 얹으면(FraudSquad 방식) 추가 개선 여지가 있을 것으로 예상.

## 다음 단계 (Day 2 이후, `reports/architecture_rationale.md`의 9일 타임라인 참고)

1. 로컬/Colab 환경에서 `--mode full`(BERT+ResNet) 실행 → 실험 A를 raw text/이미지 기반으로 재현하고 handcrafted-feature 결과와 비교
2. 실험 A에 CLIP text-image consistency score(코사인 유사도) 추가 → ablation
3. 실험 B에 LM(BERT) embedding을 node feature로 교체해서 FraudSquad 원 설계에 더 가깝게 재현
4. 두 실험 결과를 발표 슬라이드의 "각 modality가 독립적으로 신호를 담고 있다"는 근거로 정리, 최종 3-modality 통합 구조는 비전/향후 연구로 프레이밍
