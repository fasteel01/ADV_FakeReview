#!/usr/bin/env bash
# AiGen-FoodReview 텍스트+handcrafted feature CSV 다운로드 (GitHub 미러 사용)
# 원본 이미지 파일이 필요하면 README의 Google Drive 링크에서 별도로 받을 것.
set -e
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/iamalegambetti/aigen-foodreview.git "$TMP_DIR/aigen"
mkdir -p "$(dirname "$0")/../data/raw"
cp "$TMP_DIR/aigen/data/"*.csv "$(dirname "$0")/../data/raw/"
rm -rf "$TMP_DIR"
echo "Done. See data/raw/{train,val,test}.csv"
