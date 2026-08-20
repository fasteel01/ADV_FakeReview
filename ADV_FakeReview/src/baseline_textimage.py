"""
실험 A: Text + Image baseline (AiGen-FoodReview)

--mode lite (기본, 이 샌드박스에서 실행 가능):
    사전학습 가중치 없이 돌아가는 handcrafted-feature 기반 baseline.
    - text-only  : TF-IDF(review text)
    - image-only : 논문이 제공하는 24개 handcrafted image feature(색감/구도 등)
    - text-feat  : readability/perplexity 등 7개 텍스트 handcrafted feature
    - combined   : 위 셋을 모두 concat
    ablation으로 "이미지 정보가 실제로 fake 탐지에 기여하는가"를 검증한다.

--mode full (로컬/Colab, HuggingFace Hub 접속 가능한 환경에서만):
    BERT([CLS] 768d) + ResNet-50(2048d) concat -> classifier.
    (BERT+ResNet 논문의 baseline 재현. 이 샌드박스에선 huggingface.co가
    막혀 있어 실행 불가 - 함수만 정의해두고 로컬에서 실행할 것)
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TEXT_FEATS = [
    "automated_readability_index", "difficult_words", "flesch_reading_ease",
    "gunning_fog", "words_per_sentence", "reading_time", "ppl",
]
IMAGE_FEATS = [
    "bright", "cont", "warm", "colorf", "sd", "cd", "td",
    "diag_dom", "rot", "hpvb", "vpvb", "hcvb", "vcvb", "sat", "clar",
]


def load_split(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def eval_split(clf, X, y, name):
    proba = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    return {
        "split": name,
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
    }


def run_lite(ablation):
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    y_train, y_val, y_test = train["label"].values, val["label"].values, test["label"].values

    feature_blocks_train = []
    feature_blocks_val = []
    feature_blocks_test = []

    if "text" in ablation:
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
        feature_blocks_train.append(tfidf.fit_transform(train["text"]))
        feature_blocks_val.append(tfidf.transform(val["text"]))
        feature_blocks_test.append(tfidf.transform(test["text"]))

    if "text_feat" in ablation:
        med = train[TEXT_FEATS].median()
        scaler = StandardScaler()
        feature_blocks_train.append(scaler.fit_transform(train[TEXT_FEATS].fillna(med)))
        feature_blocks_val.append(scaler.transform(val[TEXT_FEATS].fillna(med)))
        feature_blocks_test.append(scaler.transform(test[TEXT_FEATS].fillna(med)))

    if "image" in ablation:
        med = train[IMAGE_FEATS].median()
        scaler = StandardScaler()
        feature_blocks_train.append(scaler.fit_transform(train[IMAGE_FEATS].fillna(med)))
        feature_blocks_val.append(scaler.transform(val[IMAGE_FEATS].fillna(med)))
        feature_blocks_test.append(scaler.transform(test[IMAGE_FEATS].fillna(med)))

    X_train = hstack(feature_blocks_train).tocsr() if len(feature_blocks_train) > 1 else feature_blocks_train[0]
    X_val = hstack(feature_blocks_val).tocsr() if len(feature_blocks_val) > 1 else feature_blocks_val[0]
    X_test = hstack(feature_blocks_test).tocsr() if len(feature_blocks_test) > 1 else feature_blocks_test[0]

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)

    return [eval_split(clf, X_val, y_val, "val"), eval_split(clf, X_test, y_test, "test")]


def run_full():
    """
    실제 사전학습 BERT + ResNet-50 + CLIP 기반 파이프라인.
    HuggingFace Hub / torchvision weight 다운로드가 필요해 클라우드 샌드박스에서는
    실행 불가 (huggingface.co, download.pytorch.org 차단됨) - 학교 서버/Colab에서 실행할 것.

    1) data/raw/images/{id}.jpg 가 없으면 먼저 scripts/download_aigen_images.sh 실행
    2) reports/features_cache/*.npz 가 없으면 BERT/ResNet/CLIP feature를 추출해서 캐싱
       (한 번만 하면 됨 - 이후 fusion 구조 비교는 캐싱된 feature로 몇 초만에 반복 가능)
    3) MultimodalFusionModel로 ablation: plain concat vs +Self-MM vs +Self-MM+MISA-lite,
       각각 CLIP consistency feature 유무까지 곱해서 비교
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    from models.multimodal_model import extract_features, MultimodalFusionModel

    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "features_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for split in ["train", "val", "test"]:
        cache_path = os.path.join(CACHE_DIR, f"{split}.npz")
        if not os.path.exists(cache_path):
            extract_features(os.path.join(DATA_DIR, f"{split}.csv"), IMAGE_DIR, cache_path, device=device)

    cached = {split: np.load(os.path.join(CACHE_DIR, f"{split}.npz")) for split in ["train", "val", "test"]}

    def to_tensors(d):
        return (
            torch.tensor(d["bert"], dtype=torch.float32),
            torch.tensor(d["resnet"], dtype=torch.float32),
            torch.tensor(d["clip_sim"], dtype=torch.float32),
            torch.tensor(d["label"], dtype=torch.int64),
        )

    train_t = to_tensors(cached["train"])
    val_t = to_tensors(cached["val"])
    test_t = to_tensors(cached["test"])

    def evaluate(model, split_t):
        bert_e, resnet_e, clip_sim, y = split_t
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(bert_e, resnet_e, clip_sim)
        proba = F.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(dim=1).numpy()
        y_np = y.numpy()
        return {
            "acc": accuracy_score(y_np, pred), "f1": f1_score(y_np, pred),
            "auc": roc_auc_score(y_np, proba),
        }

    def train_and_eval(use_aux, use_misa, use_clip_consistency, epochs=60, lr=1e-3,
                        aux_weight=0.3, ortho_weight=0.1):
        model = MultimodalFusionModel(use_aux=use_aux, use_misa=use_misa,
                                       use_clip_consistency=use_clip_consistency)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        bert_e, resnet_e, clip_sim, y = train_t

        best_val_f1, best_test = -1, None
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits, aux_logits, ortho_loss = model(bert_e, resnet_e, clip_sim)
            loss = F.cross_entropy(logits, y)
            if aux_logits is not None:
                for al in aux_logits:
                    loss = loss + aux_weight * F.cross_entropy(al, y)
            loss = loss + ortho_weight * ortho_loss
            loss.backward()
            optimizer.step()

            val_metrics = evaluate(model, val_t)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_test = evaluate(model, test_t)

        return best_test

    settings = [
        ("concat only (fusion 메커니즘 없음)", dict(use_aux=False, use_misa=False, use_clip_consistency=False)),
        ("+ CLIP consistency feature", dict(use_aux=False, use_misa=False, use_clip_consistency=True)),
        ("+ Self-MM", dict(use_aux=True, use_misa=False, use_clip_consistency=True)),
        ("+ Self-MM + MISA-lite (최종)", dict(use_aux=True, use_misa=True, use_clip_consistency=True)),
    ]

    print("\n=== 실험 A full: BERT+ResNet+CLIP fusion ablation ===\n")
    rows = []
    for name, kwargs in settings:
        test_metrics = train_and_eval(**kwargs)
        rows.append({"setting": name, **test_metrics})
        print(f"[{name}] test acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_a_full_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lite", "full"], default="lite")
    args = parser.parse_args()

    if args.mode == "full":
        run_full()
        return

    ablations = {
        "text-only (TF-IDF)": {"text"},
        "text_feat-only (readability/ppl)": {"text_feat"},
        "image-only (aesthetic/color feats)": {"image"},
        "text + image": {"text", "image"},
        "text + text_feat + image (combined)": {"text", "text_feat", "image"},
    }

    print("=== 실험 A: Text+Image baseline (lite / handcrafted feature) ===\n")
    rows = []
    for name, ablation in ablations.items():
        results = run_lite(ablation)
        for r in results:
            r["setting"] = name
            rows.append(r)
        test_r = [r for r in results if r["split"] == "test"][0]
        print(f"[{name}] test acc={test_r['acc']:.4f}  f1={test_r['f1']:.4f}  auc={test_r['auc']:.4f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "experiment_a_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
