#!/bin/bash
# exp08b — Full EPDA interference matrix launcher
# Usage: bash scripts/launch_exp08b.sh <gpu_id> [--pairs-only]
#
# Runs all 11 combos sequentially on one GPU.
# Estimated time: ~4-6 hours for all, ~3h for pairs only.

set -euo pipefail

GPU=${1:-0}
PAIRS_ONLY=${2:-""}
WARMUP=15
ITERATIONS=40
K=10
OUTDIR="exp/exp08b"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTDIR"

echo "============================================"
echo "exp08b — EPDA Interference Matrix"
echo "GPU: $GPU, warmup: $WARMUP, iter: $ITERATIONS"
echo "Output: $OUTDIR/"
echo "============================================"

SCRIPT="scripts/exp08b_interference_matrix.py"
COMMON="--gpu $GPU --warmup $WARMUP --iterations $ITERATIONS --k $K"

# Phase 1: All 6 pairs
PAIRS="EP ED EA PD PA DA"
for PAIR in $PAIRS; do
    echo ""
    echo ">>> [$PAIR] Starting at $(date '+%H:%M:%S')"
    python "$SCRIPT" --combo "$PAIR" $COMMON \
        --output "$OUTDIR/results_${PAIR}.json" \
        2>&1 | tee "$OUTDIR/log_${PAIR}.txt"
    echo ">>> [$PAIR] Done at $(date '+%H:%M:%S')"
done

if [ "$PAIRS_ONLY" = "--pairs-only" ]; then
    echo ""
    echo "=== Pairs complete. Skipping triples/quad (--pairs-only). ==="
    python "$SCRIPT" --combo PAIRS $COMMON --output-dir "$OUTDIR" 2>/dev/null || true
    exit 0
fi

# Phase 2: Triples + Quad
MULTIS="EPD EPA EDA PDA EPDA"
for MULTI in $MULTIS; do
    echo ""
    echo ">>> [$MULTI] Starting at $(date '+%H:%M:%S')"
    python "$SCRIPT" --combo "$MULTI" $COMMON \
        --output "$OUTDIR/results_${MULTI}.json" \
        2>&1 | tee "$OUTDIR/log_${MULTI}.txt"
    echo ">>> [$MULTI] Done at $(date '+%H:%M:%S')"
done

echo ""
echo "============================================"
echo "exp08b COMPLETE at $(date)"
echo "Results in $OUTDIR/"
echo "============================================"
