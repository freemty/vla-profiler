#!/bin/bash
# exp08b — Full EPDA interference matrix launcher
# Usage: bash scripts/launch_exp08b.sh <gpu_id> [MODE]
#   MODE: (empty)     — pairs (6) + triples/quad (5)     ~4-6h
#         --pairs-only — 6 pairs only                    ~3h
#         --multi-only — 4 triples + 1 quad only         ~3-4h  (M4 外推验证)
#
# Enforces exp07a canonical conditions: persistence mode + warmup>=15.

set -euo pipefail

GPU=${1:-0}
MODE=${2:-""}
WARMUP=${WARMUP:-15}
ITERATIONS=${ITERATIONS:-40}
K=10
OUTDIR="exp/exp08b"

# --- Guard: warmup must be >= 15 (exp07a canonical) ---
if [[ "$WARMUP" -lt 15 ]]; then
  echo "ERROR: WARMUP must be >= 15 (exp07a canonical). Got $WARMUP." >&2
  exit 2
fi

# --- Persistence mode (mandatory — exp07a bimodal-pollution fix) ---
if ! nvidia-smi -pm 1 >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi -pm 1 failed. Persistence mode required for stable power state." >&2
  echo "       Without it, GPU clock ramps across warmup and contaminates measurements." >&2
  exit 3
fi

# Lock clocks if driver supports it (RTX 5880 Ada does).
nvidia-smi -i "$GPU" --lock-gpu-clocks=tdp,tdp >/dev/null 2>&1 || \
  echo "[warn] could not lock GPU clocks; continuing (persistence mode alone is usually enough)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTDIR"

echo "============================================"
echo "exp08b — EPDA Interference Matrix"
echo "GPU: $GPU, warmup: $WARMUP, iter: $ITERATIONS"
echo "Persistence mode: ON"
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
