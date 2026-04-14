#!/bin/bash
# Launch vlla experiment on xdlab23
# Usage: bash scripts/launch_exp.sh [GPU_ID] [CONFIG_NAME]
#   e.g., bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling
set -e

cd /data1/ybyang/vlla
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit-probe

export HF_HOME=/data1/ybyang/huggingface
export TRANSFORMERS_CACHE=/data1/ybyang/huggingface

GPU_ID=${1:-0}
CONFIG_NAME=${2:-qwen_vl_7b/profiling}

mkdir -p logs

echo "[$(date)] Starting experiment: ${CONFIG_NAME} on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.run_tasks \
    --config-path ../configs \
    --config-name $CONFIG_NAME \
    device=cuda:0 \
    2>&1 | tee "logs/${CONFIG_NAME//\//_}.log"
echo "[$(date)] Experiment complete"
