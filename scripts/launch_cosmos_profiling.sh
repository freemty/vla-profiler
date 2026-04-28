#!/bin/bash
# exp09a — Cosmos Policy profiling launcher for xdlab23
# Usage: bash scripts/launch_cosmos_profiling.sh [GPU_ID]
#
# Requires: cosmos-policy Docker built on xdlab23
# See exp/exp09a/README.md for full setup

set -euo pipefail

GPU=${1:-0}
VLLA_ROOT="/data1/ybyang/vlla"
IMAGE="cosmos-policy"

echo "============================================"
echo "exp09a — Cosmos Policy Direct-Mode Profiling"
echo "GPU: $GPU"
echo "============================================"

# Persistence mode (exp07a canonical)
nvidia-smi -pm 1 2>/dev/null || echo "[warn] nvidia-smi -pm 1 failed (run as root?)"

# Run inside Docker
docker run -u root \
    -e CUDA_VISIBLE_DEVICES="$GPU" \
    -e HF_HOME=/data1/ybyang/huggingface \
    -v /data1/ybyang:/data1/ybyang \
    -v "$HOME/.cache:/home/cosmos/.cache" \
    --gpus "device=$GPU" \
    --ipc=host \
    --rm \
    -w "$VLLA_ROOT" \
    "$IMAGE" \
    bash -c "
        echo '[docker] Running canonical 5-step profiling...'
        python scripts/exp09a_cosmos_policy_profiling.py \
            --gpu 0 --warmup 15 --iterations 20 --denoise-steps 5

        echo ''
        echo '[docker] Running step sweep (1,2,5,10,20)...'
        for N in 1 2 5 10 20; do
            echo \"--- steps=\$N ---\"
            python scripts/exp09a_cosmos_policy_profiling.py \
                --gpu 0 --warmup 15 --iterations 20 --denoise-steps \$N \
                --output exp/exp09a/results_steps_\${N}.json
        done
    "

echo ""
echo "============================================"
echo "exp09a COMPLETE at $(date)"
echo "Results in exp/exp09a/"
echo "============================================"
