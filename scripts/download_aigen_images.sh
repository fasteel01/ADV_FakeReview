#!/usr/bin/env bash
# AiGen-FoodReview 원본 이미지 다운로드 (Google Drive)
# 클라우드 샌드박스에서는 drive.google.com이 막혀있어 여기서는 못 받는다.
# 학교 서버/Colab처럼 인터넷이 열려있는 환경에서 실행할 것.
set -e
pip install -q gdown

TARGET_DIR="$(dirname "$0")/../data/raw/images"
mkdir -p "$TARGET_DIR"

# https://drive.google.com/file/d/1FzBIklsUkNaBKdCWvjbeb3h4PH1zUI3Q/view (AiGen-FoodReview README 참고)
FILE_ID="1FzBIklsUkNaBKdCWvjbeb3h4PH1zUI3Q"
ZIP_PATH="/tmp/aigen_images.zip"

gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$ZIP_PATH"
unzip -oq "$ZIP_PATH" -d "$TARGET_DIR"
rm -f "$ZIP_PATH"

echo "Done. 이미지 개수: $(ls "$TARGET_DIR" | wc -l)"
echo "구조가 data/raw/images/{id}.jpg 형태가 아니면(중첩 폴더 등) 직접 확인해서 옮겨줄 것"
