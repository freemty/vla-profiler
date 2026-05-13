#!/usr/bin/env bash
# Run LIBERO-4 eval for all VLA/WAM models on separate GPUs.
# Models: Fast-WAM(GPU0) / Pi-Zero(GPU1) / LingBot-VA(GPU2) / LingBot-VLA(GPU3) / Cosmos(GPU4)
# Usage: bash scripts/run_libero_all.sh [episodes_per_task]
#   Default: 2 (smoke test). Use 20 for full eval.
set -euo pipefail

EP=${1:-2}
SUITES="libero_spatial libero_object libero_goal libero_10"

echo "[run_libero_all] Episodes per task: $EP"
echo "[run_libero_all] Suites: $SUITES"

export MUJOCO_GL=egl
export HF_HOME=/data1/ybyang/huggingface
# cuDNN 9.10 from pip overrides system cuDNN 9.1.1 (needed by torch 2.9 DiT ops)
export LD_LIBRARY_PATH=/home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

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
# Pi-Zero (exp07c) — direct load via PiZeroController (no openpi/flax/JAX)
########################################
run_pizero() {
  local GPU=1
  local CKPT=/data1/ybyang/huggingface/models--allenzren--open-pi-zero/snapshots/8518347d4ae0c6cfc69fbdda970b3f38c6ff76ca/bridge_beta_step19296_2024-12-26_22-30_42.pt
  local OUT=/data1/ybyang/vlla/exp/exp07c
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe

  cd /data1/ybyang/vlla
  echo "[Pi-Zero] Running direct eval (GPU $GPU, $EP ep)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_exp07c_libero.py \
    --ckpt "$CKPT" \
    --episodes "$EP" \
    --out "$OUT" \
    2>&1 | tee "$OUT/eval.log" || echo "[Pi-Zero] FAILED"
  echo "[Pi-Zero] all suites done"
}

########################################
# LingBot-VLA (exp03b) — vit-probe env, direct load (LossKwargs shim)
########################################
run_lingbotvla() {
  local GPU=3
  local OUT=/data1/ybyang/vlla/exp/exp03b
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe

  cd /data1/ybyang/vlla
  echo "[LingBot-VLA] Running direct eval (GPU $GPU, $EP ep)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_exp03b_libero.py \
    --episodes "$EP" \
    --out "$OUT" \
    2>&1 | tee "$OUT/eval.log" || echo "[LingBot-VLA] FAILED"
  echo "[LingBot-VLA] all suites done"
}

########################################
# Cosmos Policy — cosmos-policy venv, vendor draccus CLI
########################################
run_cosmos() {
  local GPU=4
  local OUT=/data1/ybyang/vlla/exp/cosmos_libero
  mkdir -p "$OUT"

  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe

  cd /data1/ybyang/vlla
  echo "[Cosmos] Running eval (GPU $GPU, $EP ep)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_cosmos_libero.py \
    --all \
    --episodes "$EP" \
    --out "$OUT" \
    --standalone \
    2>&1 | tee "$OUT/eval.log" || echo "[Cosmos] FAILED"
  echo "[Cosmos] all suites done"
}

########################################
# Run all five in parallel (each in subshell for separate conda env)
########################################
echo "[run_libero_all] Starting 5 evals in parallel on GPUs 0-4..."

(run_fastwam) &
PID_FW=$!

(run_pizero) &
PID_PZ=$!

(run_lingbotva) &
PID_LV=$!

(run_lingbotvla) &
PID_VLA=$!

(run_cosmos) &
PID_CS=$!

echo "[run_libero_all] PIDs: FastWAM=$PID_FW Pi-Zero=$PID_PZ LingBot-VA=$PID_LV LingBot-VLA=$PID_VLA Cosmos=$PID_CS"
wait $PID_FW $PID_PZ $PID_LV $PID_VLA $PID_CS
echo "[run_libero_all] All evals complete."
