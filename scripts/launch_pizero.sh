#!/bin/bash
# Launch Pi-Zero experiment on xdlab23 (uses pizero uv venv, NOT conda)
# Usage: bash scripts/launch_pizero.sh [GPU_ID] [CONFIG_NAME]
#   e.g., bash scripts/launch_pizero.sh 0 pizero/profiling
set -e

cd /data1/ybyang/vlla
source .venvs/pizero/bin/activate

export HF_HOME=/data1/ybyang/huggingface
export TRANSFORMERS_CACHE=/data1/ybyang/huggingface

GPU_ID=${1:-0}
CONFIG_NAME=${2:-pizero/profiling}

mkdir -p logs

echo "[$(date)] Starting Pi-Zero experiment: ${CONFIG_NAME} on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.run_tasks \
    --config-path ../configs \
    --config-name "$CONFIG_NAME" \
    2>&1 | tee "logs/${CONFIG_NAME//\//_}.log"
echo "[$(date)] Experiment complete"
