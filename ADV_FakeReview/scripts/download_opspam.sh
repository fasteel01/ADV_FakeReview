#!/usr/bin/env bash
# Ott et al. Deceptive Opinion Spam Corpus (ACL 2011 / NAACL 2013)
# 사람이 직접 쓴 가짜 리뷰(Amazon Mechanical Turk) vs 진짜 리뷰(TripAdvisor 등),
# 시카고 호텔 20곳, 총 1600개 (positive/negative x deceptive/truthful 각 400개).
# 원 출처(myleott.com)가 이 프로젝트의 네트워크 환경에 따라 막혀있을 수 있어
# GitHub에 커밋되어 있는 사본을 사용한다.
set -e
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/PauDK/Deceptive-Review-Detection.git "$TMP_DIR/opspam"
mkdir -p "$(dirname "$0")/../data/raw/opspam"
cp "$TMP_DIR/opspam/Chicago_Hotel_Review/Chicago_Hotel_Reviews.csv" "$(dirname "$0")/../data/raw/opspam/"
rm -rf "$TMP_DIR"
echo "Done. See data/raw/opspam/Chicago_Hotel_Reviews.csv"
