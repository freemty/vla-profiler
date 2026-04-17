#!/bin/bash
# One-command remote experiment launch via SSH
# Usage: bash scripts/run_remote.sh <GPU_ID> <CONFIG_NAME>
#   e.g., bash scripts/run_remote.sh 0 qwen_vl_7b/profiling
#         bash scripts/run_remote.sh 1 qwen_vl_7b/attention
#         bash scripts/run_remote.sh 0 act/profiling
set -e

REMOTE_HOST="xdlab23_yang"
REMOTE_DIR="/data1/ybyang/vlla"

GPU_ID=${1:-0}
CONFIG_NAME=${2:?Usage: run_remote.sh <GPU_ID> <CONFIG_NAME>}

echo "=== Launching on xdlab23: GPU ${GPU_ID}, config ${CONFIG_NAME} ==="
ssh -p 66 "$REMOTE_HOST" "cd $REMOTE_DIR && bash scripts/launch_exp.sh $GPU_ID $CONFIG_NAME"
