#!/bin/bash
# exp08b — Full EPDA interference matrix launcher
# Usage: bash scripts/launch_exp08b.sh <gpu_id> [MODE]
#   MODE: (empty)     — pairs (6) + triples/quad (5)     ~4-6h
#         --pairs-only — 6 pairs only                    ~3h
#         --multi-only — 4 triples + 1 quad only         ~3-4h  (M4 外推验证)
#
# Estimated time: ~4-6 hours for all, ~3h for pairs only, ~3-4h for multi only.

set -euo pipefail

GPU=${1:-0}
MODE=${2:-""}
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

PAIRS="EP ED EA PD PA DA"
MULTIS="EPD EPA EDA PDA EPDA"

# Phase 1: pairs (skip if --multi-only)
if [ "$MODE" != "--multi-only" ]; then
    for PAIR in $PAIRS; do
        echo ""
        echo ">>> [$PAIR] Starting at $(date '+%H:%M:%S')"
        python "$SCRIPT" --combo "$PAIR" $COMMON \
            --output "$OUTDIR/results_${PAIR}.json" \
            2>&1 | tee "$OUTDIR/log_${PAIR}.txt"
        echo ">>> [$PAIR] Done at $(date '+%H:%M:%S')"
    done
fi

if [ "$MODE" = "--pairs-only" ]; then
    echo ""
    echo "=== Pairs complete. Skipping triples/quad (--pairs-only). ==="
    exit 0
fi

# Phase 2: Triples + Quad
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
