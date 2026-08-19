#!/usr/bin/env bash
# 실험 D: "진짜" 어려운 멀티모달 가짜 리뷰 데이터셋 (실험용, 조인 성공 여부 미확정)
#
# 지금까지 없던 조합: 실제 사람이 쓴 가짜 리뷰 라벨(LLM 생성 아님) + 실제 사용자가
# 올린 리뷰 사진. 이 조합을 공개 데이터셋 하나로는 구할 수 없어서, 서로 다른 두
# 데이터셋을 조인해서 직접 만든다.
#
#   1) Hollenbeck et al. "fake-reviews-data": Amazon 가짜 리뷰 언더그라운드 마켓
#      추적 + Amazon 자체 삭제 데이터를 결합해 만든 (asin, reviewer, date 단위)
#      실제 가짜/진짜 라벨. 이미지는 없음.
#      https://github.com/bretthollenbeck/fake-reviews-data
#   2) McAuley Lab Amazon Reviews 2023: 리뷰마다 실제 사용자가 올린 사진(images
#      필드)이 포함된 원본 리뷰 텍스트. 가짜 여부 라벨은 없음.
#      https://amazon-reviews-2023.github.io/
#
# asin/user_id/date를 키로 두 데이터를 조인해서 "실제 가짜 라벨 + 실제 사진"이
# 남는 리뷰만 추출하는 게 목표. 정확한 컬럼명과 실제 조인 성공률은 다운로드 전엔
# 알 수 없으므로, 먼저 --explore로 스키마를 확인한 뒤 조인 로직을 확정한다.
#
# ⚠️ mcauleylab.ucsd.edu, dropbox.com은 Claude 샌드박스에서 막혀있어(403) 이 스크립트를
# 학교 서버/Colab처럼 인터넷이 열려있는 환경에서 실행해야 한다.
set -e
OUT_DIR="$(dirname "$0")/../data/raw/amazon_hard"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "[1/2] Hollenbeck et al. fake/real 라벨 데이터 (Dropbox, ~90MB) 다운로드..."
curl -L -o final-dataset-all.dta "https://www.dropbox.com/s/4ll1k7b5bt2em2s/final-dataset-all.dta?dl=1"
curl -L -o README.rtf "https://www.dropbox.com/s/fj2pi2svgocj6th/README.rtf?dl=1"

echo "[2/2] Amazon Reviews 2023 - Health_and_Personal_Care (McAuley Lab, ~494K reviews) 다운로드..."
# 이 카테고리를 고른 이유: (a) 33개 카테고리 중 상대적으로 작아서(다른 카테고리는
# 수천만 건) 학교 서버에서도 다루기 쉽고, (b) 건강기능식품/화장품류는 실제로
# 가짜 리뷰 문제가 잘 알려진 카테고리라 Hollenbeck 데이터의 가짜 라벨과 겹칠
# 가능성이 상대적으로 높다.
curl -L -o Health_and_Personal_Care.jsonl.gz \
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Health_and_Personal_Care.jsonl.gz"

echo ""
echo "Done. 다음 순서로 실행:"
echo "  1) python src/build_amazon_hard_dataset.py --explore   # 실제 컬럼/조인 가능성부터 확인"
echo "  2) (탐색 결과 보고 조인 로직 확정 후) python src/build_amazon_hard_dataset.py"
