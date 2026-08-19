#!/usr/bin/env bash
# YelpChi (Rayana & Akoglu 2015, CARE-GNN 전처리본) 다운로드
set -e
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/YingtongDou/CARE-GNN.git "$TMP_DIR/caregnn"
mkdir -p "$(dirname "$0")/../data/raw/yelpchi"
unzip -o "$TMP_DIR/caregnn/data/YelpChi.zip" -d "$(dirname "$0")/../data/raw/yelpchi"
rm -rf "$TMP_DIR"
echo "Done. See data/raw/yelpchi/YelpChi.mat"
