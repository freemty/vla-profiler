#!/bin/bash
# Run experiment locally (for testing or if local GPU available)
# Usage: bash scripts/run_local.sh <GPU_ID> <CONFIG_NAME>
#   e.g., bash scripts/run_local.sh 0 qwen_vl_7b/profiling
set -eo pipefail

GPU_ID=${1:-0}
CONFIG_NAME=${2:?Usage: run_local.sh <GPU_ID> <CONFIG_NAME>}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[$(date)] Starting local experiment: ${CONFIG_NAME} on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.run_tasks \
    --config-path ../configs \
    --config-name "$CONFIG_NAME" \
    2>&1 | tee "logs/${CONFIG_NAME//\//_}_local.log"
echo "[$(date)] Experiment complete"
