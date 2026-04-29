#!/usr/bin/env bash
# Run LIBERO-4 eval for Fast-WAM / LingBot-VA / Pi-Zero on separate GPUs.
# Each model uses its own conda env + eval interface.
# Usage: bash scripts/run_libero_all.sh [episodes_per_task]
#   Default: 2 (smoke test). Use 20 for full eval.
set -euo pipefail

EP=${1:-2}
SUITES="libero_spatial libero_object libero_goal libero_10"

echo "[run_libero_all] Episodes per task: $EP"
echo "[run_libero_all] Suites: $SUITES"

export MUJOCO_GL=egl
export HF_HOME=/data1/ybyang/huggingface

########################################
# Fast-WAM (exp04e) — fastwam conda env, Hydra interface
########################################
run_fastwam() {
  local GPU=0
  local FW=/data1/ybyang/FastWAM
  local OUT=/data1/ybyang/vlla/exp/exp04e
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate fastwam

  cd "$FW"
  for suite in $SUITES; do
    echo "[FastWAM] $suite (GPU $GPU, $EP ep)"
    CUDA_VISIBLE_DEVICES=$GPU python experiments/libero/eval_libero_single.py \
      --config-name sim_libero \
      ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
      EVALUATION.task_suite_name="$suite" \
      EVALUATION.num_trials="$EP" \
      EVALUATION.num_inference_steps=5 \
      EVALUATION.dataset_stats_path=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
      EVALUATION.output_dir="$OUT/${suite}" \
      2>&1 | tee "$OUT/${suite}.log" || echo "[FastWAM] $suite FAILED"
  done
  echo "[FastWAM] all suites done"
}

########################################
# LingBot-VA (exp04d) — vit-probe env, server-client interface
########################################
run_lingbotva() {
  local GPU=2
  local LVA=/data1/ybyang/lingbot-va
  local CKPT=/data1/ybyang/huggingface/robbyant/lingbot-va-posttrain-libero-long
  local OUT=/data1/ybyang/vlla/exp/exp04d
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe

  cd "$LVA"
  local PORT=29056

  # Start server
  echo "[LingBot-VA] Starting server on GPU $GPU, port $PORT..."
  CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port $PORT \
    --save_root "$OUT/vis" \
    > "$OUT/server.log" 2>&1 &
  local SERVER_PID=$!
  echo "[LingBot-VA] Server PID=$SERVER_PID, waiting 120s for model load..."
  sleep 120

  for suite in $SUITES; do
    local suite_short=${suite/libero_/}
    echo "[LingBot-VA] $suite (GPU $GPU, $EP ep)"
    python evaluation/libero/client.py \
      --libero-benchmark "$suite" \
      --port $PORT \
      --test-num "$EP" \
      --task-range 0 10 \
      --out-dir "$OUT/${suite}" \
      2>&1 | tee "$OUT/${suite}.log" || echo "[LingBot-VA] $suite FAILED"
  done

  kill $SERVER_PID 2>/dev/null || true
  echo "[LingBot-VA] all suites done"
}

########################################
# Pi-Zero (exp07c) — openpi env, server-client interface
########################################
run_pizero() {
  local GPU=1
  local OPENPI=/data1/ybyang/openpi
  local CKPT=/data1/ybyang/huggingface/models--allenzren--open-pi-zero
  local OUT=/data1/ybyang/vlla/exp/exp07c
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe

  cd "$OPENPI"
  export PYTHONPATH=${OPENPI}/third_party/libero:${PYTHONPATH:-}

  for suite in $SUITES; do
    echo "[Pi-Zero] $suite (GPU $GPU, $EP ep)"

    # Start server
    CUDA_VISIBLE_DEVICES=$GPU python scripts/serve_policy.py \
      --env LIBERO policy:checkpoint \
      --policy.config pi0_libero --policy.dir "$CKPT" \
      > "$OUT/${suite}_server.log" 2>&1 &
    local SERVER_PID=$!
    echo "[Pi-Zero] Server PID=$SERVER_PID, waiting 60s..."
    sleep 60

    # Run client
    python examples/libero/main.py \
      --args.task-suite-name "$suite" \
      --args.num-episodes "$EP" \
      --args.save-dir "$OUT/${suite}/" \
      2>&1 | tee "$OUT/${suite}_client.log" || echo "[Pi-Zero] $suite FAILED"

    kill $SERVER_PID 2>/dev/null || true
    sleep 5
  done
  echo "[Pi-Zero] all suites done"
}

########################################
# Run all three in parallel (each in subshell for separate conda env)
########################################
echo "[run_libero_all] Starting 3 evals in parallel on GPUs 0/1/2..."

(run_fastwam) &
PID_FW=$!

(run_pizero) &
PID_PZ=$!

(run_lingbotva) &
PID_LV=$!

echo "[run_libero_all] PIDs: FastWAM=$PID_FW Pi-Zero=$PID_PZ LingBot-VA=$PID_LV"
wait $PID_FW $PID_PZ $PID_LV
echo "[run_libero_all] All evals complete."
