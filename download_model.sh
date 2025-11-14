#!/bin/bash

# curl and unzip
apt-get update && apt-get install -y curl unzip

URL="https://github.com/nathaniel32/Computer-Vision/releases/download/v1/best_model.zip"
ZIP_FILE="segmentation_model_tmp.zip"
DEST="."

[ ! -d "$DEST" ] && mkdir -p "$DEST"

curl -L -o "$ZIP_FILE" "$URL"
unzip -o "$ZIP_FILE" -d "$DEST"
rm "$ZIP_FILE"