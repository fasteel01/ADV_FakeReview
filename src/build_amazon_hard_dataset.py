"""
실험 D: 실제 사람이 쓴 가짜 리뷰 라벨 + 실제 사용자 사진을 조인해서 만드는
"진짜 어려운" 멀티모달 가짜 리뷰 데이터셋 (공개된 기성 데이터셋이 없어서 직접 구축).

배경: 실험 A(AiGen-FoodReview)는 fake가 GPT-4 대필이라 너무 쉽고, 실험 C(Ott)는
사람이 작정하고 쓴 어려운 가짜지만 이미지가 없다. "사람이 쓴 어려운 가짜 + 진짜
이미지"가 둘 다 있는 공개 데이터셋을 찾아봤지만 없어서(reports/experiment_d_plan.md
참고), 서로 다른 두 데이터셋을 asin 기준으로 조인해서 직접 만든다.

  1) Hollenbeck et al. fake-reviews-data (.dta, Dropbox)
     - Amazon 가짜 리뷰 언더그라운드 마켓 추적 + Amazon 자체 삭제 데이터 기반
       실제 가짜/진짜 라벨. 이미지 없음.
  2) McAuley Lab Amazon Reviews 2023 - Health_and_Personal_Care (jsonl.gz)
     - 리뷰마다 실제 사용자가 올린 사진(images 필드) 포함. 가짜 라벨 없음.

이 스크립트는 반드시 --explore 모드부터 실행해서 실제 컬럼명/조인 성공률을
먼저 확인해야 한다. 두 데이터셋 다 문서상 정확한 스키마를 확인 못 했기 때문에
(Hollenbeck 쪽은 특히 "fake 여부"가 단일 컬럼이 아니라 여러 플래그의 조합일
가능성이 높음 - README.rtf 참고), --explore 출력을 보고 나서 조인/라벨링 로직을
확정한다. 그 전까지 본 실행 모드는 일부러 NotImplementedError로 막아둔다.

실행 위치: 학교 서버/Colab처럼 인터넷이 열려있는 환경 (scripts/download_amazon_hard.sh
로 먼저 데이터를 받아둘 것. mcauleylab.ucsd.edu, dropbox.com은 Claude 샌드박스에서
막혀있어 이 스크립트 자체를 샌드박스에서 실행/검증할 수 없었다).
"""
import argparse
import gzip
import json
import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "amazon_hard")
DTA_PATH = os.path.join(BASE_DIR, "final-dataset-all.dta")
README_PATH = os.path.join(BASE_DIR, "README.rtf")
JSONL_PATH = os.path.join(BASE_DIR, "Health_and_Personal_Care.jsonl.gz")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "amazon_hard_processed")


def explore_dta():
    print("=" * 70)
    print(f"[1] Hollenbeck fake/real 라벨 데이터: {DTA_PATH}")
    print("=" * 70)
    if not os.path.exists(DTA_PATH):
        print(f"  !! 파일 없음. 먼저 bash scripts/download_amazon_hard.sh 실행 필요")
        return None

    df = pd.read_stata(DTA_PATH, convert_categoricals=False)
    print(f"shape: {df.shape}")
    print("\n컬럼 + dtype:")
    print(df.dtypes.to_string())
    print("\n상위 5행:")
    print(df.head(5).to_string())

    print("\n카디널리티 낮은(<=10 unique) 컬럼의 값 분포 (라벨/플래그 컬럼 후보):")
    for c in df.columns:
        try:
            nu = df[c].nunique(dropna=True)
        except TypeError:
            continue
        if nu <= 10:
            print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")

    if os.path.exists(README_PATH):
        print(f"\n(참고) README.rtf도 받아져 있음: {README_PATH} — 열어서 컬럼 설명 직접 확인 권장")

    return df


def explore_jsonl(n=200000):
    print("\n" + "=" * 70)
    print(f"[2] Amazon Reviews 2023 - Health_and_Personal_Care: {JSONL_PATH}")
    print("=" * 70)
    if not os.path.exists(JSONL_PATH):
        print(f"  !! 파일 없음. 먼저 bash scripts/download_amazon_hard.sh 실행 필요")
        return set()

    count = 0
    with_images = 0
    sample_printed = False
    asins = set()

    with gzip.open(JSONL_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if not sample_printed:
                print("샘플 레코드 키:", list(rec.keys()))
                print("샘플 레코드 (앞부분):", json.dumps(rec, ensure_ascii=False)[:800])
                sample_printed = True
            if rec.get("images"):
                with_images += 1
            a = rec.get("asin") or rec.get("parent_asin")
            if a:
                asins.add(a)
            count += 1
            if n and (i + 1) >= n:
                break

    print(f"\n스캔한 레코드 수: {count} (n={n if n else 'ALL'} 기준)")
    print(f"images 필드가 비어있지 않은 레코드: {with_images} ({100 * with_images / max(count, 1):.2f}%)")
    print(f"unique asin/parent_asin: {len(asins)}")
    return asins


def check_join(dta_df, jsonl_asins):
    print("\n" + "=" * 70)
    print("[3] 조인 가능성 체크")
    print("=" * 70)
    if dta_df is None or not jsonl_asins:
        print("  둘 중 하나가 없어서 조인 체크 스킵")
        return

    candidate_cols = [c for c in dta_df.columns if "asin" in c.lower() or "product" in c.lower()]
    print(f"dta 쪽 asin/product 후보 컬럼: {candidate_cols}")
    if not candidate_cols:
        print("  !! asin으로 보이는 컬럼을 못 찾음 - README.rtf 확인 후 실제 컬럼명 알려줄 것")
        return

    for c in candidate_cols:
        vals = set(dta_df[c].astype(str))
        overlap = vals & jsonl_asins
        print(f"  {c}: dta 쪽 unique {len(vals)}개 / Health_and_Personal_Care와 교집합 {len(overlap)}개")

    print(
        "\n위 출력(컬럼 목록 + dtype + value_counts + 조인 교집합 수)을 그대로 복사해서 전달하면, "
        "그걸 보고 정확한 조인 키 + 라벨링 규칙(예: fake_product & fake_reviewer 조합)을 확정해서 "
        "이 스크립트의 build() 함수를 채워줄게."
    )


def build():
    """실제 조인 + 라벨링 + 이미지 필터링 + 다운로드.

    --explore 결과로 정확한 컬럼명/라벨 규칙을 확인하기 전까지는 일부러 구현하지
    않음 (잘못된 컬럼명을 가정하고 구현하면 조용히 틀린 라벨을 만들 위험이 큼).
    """
    raise NotImplementedError(
        "먼저 `python src/build_amazon_hard_dataset.py --explore`를 실행해서 "
        "실제 컬럼명과 조인 성공률을 확인한 뒤, 그 출력을 공유해줘. "
        "그걸 보고 이 build() 함수(조인 키 + fake 라벨 규칙 + 이미지 다운로드)를 채워넣을게."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true", help="스키마/조인 가능성만 확인하고 종료")
    parser.add_argument(
        "--explore-n", type=int, default=200000,
        help="jsonl.gz는 수십만~수백만 줄이라 explore에서는 앞부분만 스캔(기본 20만). 0이면 전체 스캔(느림).",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.explore:
        dta_df = explore_dta()
        n = None if args.explore_n == 0 else args.explore_n
        jsonl_asins = explore_jsonl(n=n)
        check_join(dta_df, jsonl_asins)
        return

    build()


if __name__ == "__main__":
    main()
