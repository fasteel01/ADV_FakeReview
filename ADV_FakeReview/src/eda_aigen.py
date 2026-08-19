"""
AiGen-FoodReview EDA
- 클래스 분포, 텍스트 길이, handcrafted text/image feature 분포 요약
- 결과는 reports/figs/ 와 stdout으로 출력
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# 눈에 편안한 2-color 팔레트 (genuine / fake)
COLOR_GENUINE = "#4C72B0"
COLOR_FAKE = "#DD8452"


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    return pd.read_csv(path)


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    full = pd.concat([train, val, test], ignore_index=True)

    print("=== AiGen-FoodReview EDA ===")
    print(f"train={len(train)}, val={len(val)}, test={len(test)}, total={len(full)}")
    print("\n클래스 분포 (0=authentic, 1=machine-generated):")
    print(full["label"].value_counts())
    print("\n텍스트 길이(단어 수) 통계 (label별):")
    full["n_words"] = full["text"].str.split().apply(len)
    print(full.groupby("label")["n_words"].describe()[["mean", "std", "min", "max"]])

    text_feats = [
        "automated_readability_index", "difficult_words", "flesch_reading_ease",
        "gunning_fog", "words_per_sentence", "reading_time", "ppl",
    ]
    image_feats = [
        "bright", "cont", "warm", "colorf", "sd", "cd", "td",
        "diag_dom", "rot", "hpvb", "vpvb", "hcvb", "vcvb", "sat", "clar",
    ]
    print("\n텍스트 handcrafted feature 평균 (label별):")
    print(full.groupby("label")[text_feats].mean().T)
    print("\n이미지 handcrafted feature 평균 (label별):")
    print(full.groupby("label")[image_feats].mean().T)

    # 클래스 분포 plot
    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts = full["label"].value_counts().sort_index()
    ax.bar(["authentic (0)", "generated (1)"], counts.values,
           color=[COLOR_GENUINE, COLOR_FAKE])
    ax.set_title("AiGen-FoodReview class balance")
    ax.set_ylabel("count")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "aigen_class_balance.png"), dpi=150)
    print(f"\nSaved: {FIG_DIR}/aigen_class_balance.png")

    # 텍스트 perplexity 분포 (readability 논문에서 가장 차이가 크다고 보고한 변수)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, color, name in [(0, COLOR_GENUINE, "authentic"), (1, COLOR_FAKE, "generated")]:
        subset = full[full["label"] == label]["ppl"]
        ax.hist(subset, bins=40, alpha=0.6, label=name, color=color)
    ax.set_title("Perplexity (ppl) distribution by class")
    ax.set_xlabel("ppl")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "aigen_ppl_distribution.png"), dpi=150)
    print(f"Saved: {FIG_DIR}/aigen_ppl_distribution.png")


if __name__ == "__main__":
    main()
