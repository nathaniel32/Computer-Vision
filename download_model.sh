#!/bin/bash

# curl and unzip
apt-get update && apt-get install -y curl unzip

URL="https://storage.googleapis.com/drive-bulk-export-anonymous/20251112T011258.505Z/4133399871716478688/fd025c1c-5adb-4625-b744-156a86d7fd76/1/07f4fb01-c58d-4a97-a734-e4a6b3dd22ec"
ZIP_FILE="segmentation_model_tmp.zip"
DEST="."

[ ! -d "$DEST" ] && mkdir -p "$DEST"

curl -L -o "$ZIP_FILE" "$URL"
unzip -o "$ZIP_FILE" -d "$DEST"
rm "$ZIP_FILE"