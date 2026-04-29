#!/bin/bash
# exp09a sweep: action-only + step sweep on xdlab23
set -euo pipefail

source /home/ybyang/miniconda3/etc/profile.d/conda.sh
conda activate vit-probe
cd /data1/ybyang/vlla

export HF_HOME=/data1/ybyang/huggingface
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_DISABLED=1

CKPT="/data1/ybyang/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8"
SCRIPT="scripts/exp09a_cosmos_policy_profiling.py"

echo "=== Action-only (no parallel gen) ==="
python $SCRIPT --gpu 0 --warmup 15 --iterations 20 --denoise-steps 5 \
    --no-parallel-gen --local-ckpt "$CKPT" \
    --output exp/exp09a/results_action_only.json

echo ""
echo "=== Step sweep (action-only) ==="
for N in 1 2 5 10 20; do
    echo "--- steps=$N ---"
    python $SCRIPT --gpu 0 --warmup 15 --iterations 10 --denoise-steps $N \
        --no-parallel-gen --local-ckpt "$CKPT" \
        --output exp/exp09a/results_steps_${N}.json
done

echo ""
echo "=== DONE ==="
