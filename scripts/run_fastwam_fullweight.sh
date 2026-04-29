#!/usr/bin/env bash
# Fast-WAM full-weight latency rerun (requires Wan2.2-TI2V-5B downloaded)
# Usage: bash scripts/run_fastwam_fullweight.sh [gpu_id]
set -euo pipefail

GPU=${1:-3}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit-probe
export HF_HOME=/data1/ybyang/huggingface

cd /data1/ybyang/vlla

# Verify Wan2.2 base exists
WAN22=/data1/ybyang/huggingface/Wan-AI/Wan2.2-TI2V-5B
if [[ ! -f "$WAN22/diffusion_pytorch_model-00001-of-00003.safetensors" ]]; then
  echo "[ERROR] Wan2.2 base not complete — missing part 1. Abort."
  exit 1
fi

echo "[fastwam-fullweight] Running on GPU $GPU with real weights (5-step)..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/profile_fastwam.py \
  --mode full \
  --checkpoint /data1/ybyang/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats /data1/ybyang/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --num-inference-steps 5 \
  --warmup 15 \
  --iterations 20 \
  --gpu "$GPU" \
  --output exp/exp04c/fastwam_fullweight.json \
  2>&1 | tee exp/exp04c/fastwam_fullweight.log

echo "[fastwam-fullweight] Done."
