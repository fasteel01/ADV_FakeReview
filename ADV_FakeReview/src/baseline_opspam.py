"""
"어려운" 가짜 리뷰 벤치마크: Ott et al. Deceptive Opinion Spam Corpus

AiGen-FoodReview는 fake class가 GPT가 대필한 리뷰라 텍스트만으로 거의 100%
구분되는(너무 쉬운) 문제였다. 이 데이터셋은 반대로 **사람이 Amazon Mechanical
Turk에서 돈을 받고 작정하고 쓴 가짜 리뷰**(vs 실제 TripAdvisor 등 진짜 리뷰)라서,
전통적인 "고용된 가짜 리뷰" 탐지 문제에 훨씬 가깝고 실제로 훨씬 어렵다
(원 논문 기준 n-gram+SVM으로 accuracy ~90% 수준, LLM 생성 리뷰 탐지의 99%대와 대비됨).

이미지/리뷰어 행동 정보는 없어서(text-only) 3-modality 실험엔 못 쓰지만,
"우리 text 파이프라인이 훨씬 어려운 데이터에서는 얼마나 떨어지는가"를 보여주는
난이도 대조군으로 아주 좋다. AiGen-FoodReview / YelpChi와 함께 3개 데이터셋
난이도 스펙트럼(쉬움-보통-어려움)을 만든다.

같은 TF-IDF + LogisticRegression 파이프라인(baseline_textimage.py의 lite 모드와
동일한 방식)을 그대로 적용해서, "같은 방법이 데이터셋에 따라 얼마나 다르게
작동하는가"를 직접 비교할 수 있게 했다. 표본이 1600개로 작아서 단일 split 대신
5-fold cross-validation으로 평가한다.
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "opspam", "Chicago_Hotel_Reviews.csv")


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Ori_Review": "text", "Label": "label"})
    print("=== Ott Deceptive Opinion Spam ===")
    print(f"n={len(df)}, label 분포: {df['label'].value_counts().to_dict()} "
          f"(1=deceptive/사람이 쓴 가짜, 0=truthful/진짜)")
    print(f"rating 분포: {df['Rating'].value_counts().to_dict()} (1=negative, 5=positive review)")

    X_text = df["text"].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X_train = tfidf.fit_transform(X_text[train_idx])
        X_test = tfidf.transform(X_text[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred))
        aucs.append(roc_auc_score(y_test, proba))
        print(f"  fold {fold+1}: acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}  auc={aucs[-1]:.4f}")

    print("\n=== 5-fold 평균 (TF-IDF + LogisticRegression, text-only) ===")
    print(f"acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"f1  = {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print(f"auc = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")

    print("\n=== 참고: 데이터셋 난이도 비교 ===")
    print("AiGen-FoodReview (text-only, LLM 생성 가짜)  : acc≈0.997  auc≈1.000  (매우 쉬움)")
    print(f"Ott Deceptive Opinion Spam (사람이 쓴 가짜)   : acc≈{np.mean(accs):.3f}  auc≈{np.mean(aucs):.3f}  (훨씬 어려움)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_c_opspam_results.csv")
    pd.DataFrame({
        "fold": list(range(1, 6)) + ["mean", "std"],
        "acc": accs + [np.mean(accs), np.std(accs)],
        "f1": f1s + [np.mean(f1s), np.std(f1s)],
        "auc": aucs + [np.mean(aucs), np.std(aucs)],
    }).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
