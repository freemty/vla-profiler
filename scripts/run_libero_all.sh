#!/usr/bin/env bash
# Run LIBERO-4 eval for Fast-WAM / Pi-Zero / LingBot-VLA in parallel on GPUs 0-2.
# Usage: bash scripts/run_libero_all.sh [episodes_per_task]
#   Default: 2 (smoke test). Use 20 for full eval.
set -euo pipefail

EP=${1:-2}
echo "[run_libero_all] Episodes per task: $EP"

# GPU assignments
GPU_FASTWAM=0
GPU_PIZERO=1
GPU_LINGBOTVLA=2

# Shared setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit-probe
export MUJOCO_GL=egl
export HF_HOME=/data1/ybyang/huggingface

cd /data1/ybyang/vlla

########################################
# Fast-WAM (exp04e)
########################################
run_fastwam() {
  local FW=/data1/ybyang/FastWAM
  local CKPT="$FW/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  local OUT=exp/exp04e
  mkdir -p "$OUT"

  for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "[FastWAM] $suite (GPU $GPU_FASTWAM, $EP ep)"
    CUDA_VISIBLE_DEVICES=$GPU_FASTWAM python "$FW/experiments/libero/eval_libero_single.py" \
      --checkpoint_path "$CKPT" \
      --suite "$suite" \
      --num_episodes_per_task "$EP" \
      --num_inference_steps 5 \
      --save_dir "$OUT/${suite}" \
      2>&1 | tee "$OUT/${suite}.log" || echo "[FastWAM] $suite FAILED"
  done
  echo "[FastWAM] done"
}

########################################
# Pi-Zero (exp07c)
########################################
run_pizero() {
  local OPENPI=/data1/ybyang/openpi
  local CKPT=/data1/ybyang/huggingface/models--allenzren--open-pi-zero
  local OUT=exp/exp07c
  mkdir -p "$OUT"

  # Pi-Zero uses server-client arch
  for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "[Pi-Zero] $suite (GPU $GPU_PIZERO, $EP ep)"

    # Start server
    CUDA_VISIBLE_DEVICES=$GPU_PIZERO python "$OPENPI/scripts/serve_policy.py" \
      --env LIBERO policy:checkpoint \
      --policy.config pi0_libero --policy.dir "$CKPT" \
      > "$OUT/${suite}_server.log" 2>&1 &
    local SERVER_PID=$!
    sleep 45  # let server load model

    # Run client
    MUJOCO_GL=egl python "$OPENPI/examples/libero/main.py" \
      --args.task-suite-name "$suite" \
      --args.num-episodes "$EP" \
      --args.save-dir "$OUT/${suite}/" \
      2>&1 | tee "$OUT/${suite}_client.log" || echo "[Pi-Zero] $suite client FAILED"

    kill $SERVER_PID 2>/dev/null || true
    sleep 5
  done
  echo "[Pi-Zero] done"
}

########################################
# LingBot-VLA (exp03b)
########################################
run_lingbotvla() {
  local CKPT=/data1/ybyang/modelscope/Robbyant/lingbot-vla-4b
  local OUT=exp/exp03b
  mkdir -p "$OUT"

  # Check if lingbot-va has a shared eval harness
  local EVAL_SCRIPT=/data1/ybyang/lingbot-va/evaluation/libero/eval_libero.py
  if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "[LingBot-VLA] eval script not found at $EVAL_SCRIPT, checking alternatives..."
    EVAL_SCRIPT=$(find /data1/ybyang/lingbot-va -name "eval*libero*" -type f 2>/dev/null | head -1)
  fi

  if [[ -z "$EVAL_SCRIPT" || ! -f "$EVAL_SCRIPT" ]]; then
    echo "[LingBot-VLA] ERROR: no eval script found. Skipping."
    echo "NOT_RUN: eval script not found" > "$OUT/SKIP.md"
    return 1
  fi

  for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "[LingBot-VLA] $suite (GPU $GPU_LINGBOTVLA, $EP ep)"
    CUDA_VISIBLE_DEVICES=$GPU_LINGBOTVLA python "$EVAL_SCRIPT" \
      --checkpoint "$CKPT" \
      --suite "$suite" \
      --num_episodes "$EP" \
      --save_dir "$OUT/${suite}" \
      2>&1 | tee "$OUT/${suite}.log" || echo "[LingBot-VLA] $suite FAILED"
  done
  echo "[LingBot-VLA] done"
}

########################################
# Run all three in parallel
########################################
echo "[run_libero_all] Starting 3 evals in parallel..."
run_fastwam &
PID_FW=$!

run_pizero &
PID_PZ=$!

run_lingbotvla &
PID_LV=$!

echo "[run_libero_all] PIDs: FastWAM=$PID_FW Pi-Zero=$PID_PZ LingBot-VLA=$PID_LV"
wait $PID_FW $PID_PZ $PID_LV
echo "[run_libero_all] All evals complete."
