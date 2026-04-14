#!/bin/bash
# Download experiment results from xdlab23
# Usage: bash scripts/download-results.sh [EXP_NAME]
#   e.g., bash scripts/download-results.sh Qwen_Qwen2.5-VL-7B-Instruct
set -e

REMOTE_HOST="xdlab23_yang"
REMOTE_DIR="/data1/ybyang/vlla/output"
LOCAL_DIR="/Users/sum_young/code/projects/vlla/output"

EXP_NAME=${1:-""}

if [ -z "$EXP_NAME" ]; then
    echo "Downloading all results..."
    rsync -avz --progress -e "ssh -p 66" \
        "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
else
    echo "Downloading results for: $EXP_NAME"
    mkdir -p "$LOCAL_DIR/$EXP_NAME"
    rsync -avz --progress -e "ssh -p 66" \
        "$REMOTE_HOST:$REMOTE_DIR/$EXP_NAME/" "$LOCAL_DIR/$EXP_NAME/"
fi

echo "=== Download complete ==="
