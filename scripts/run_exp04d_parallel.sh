#!/usr/bin/env bash
# exp04d: LingBot-VA LIBERO-4 eval — parallel across 4 GPUs.
# Each GPU runs its own server + client for one suite.
#
# Usage:
#   bash scripts/run_exp04d_parallel.sh 20     # full eval
#   bash scripts/run_exp04d_parallel.sh 2      # smoke test
set -euo pipefail

EP=${1:-20}
LVA=/data1/ybyang/lingbot-va
CKPT=/data1/ybyang/huggingface/robbyant/lingbot-va-posttrain-libero-long
OUT=/data1/ybyang/vlla/exp/exp04d

export MUJOCO_GL=egl
export HF_HOME=/data1/ybyang/huggingface
export LD_LIBRARY_PATH=/home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$LVA:${PYTHONPATH:-}

eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
conda activate vit-probe

run_suite() {
  local GPU=$1
  local SUITE=$2
  local PORT=$3
  local MASTER_PORT=$4
  mkdir -p "$OUT"

  cd "$LVA"

  echo "[exp04d:$SUITE] Starting server on GPU $GPU, port $PORT..."
  CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.run \
    --nproc_per_node 1 --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port $PORT \
    --save_root "$OUT/vis_${SUITE}" \
    > "$OUT/${SUITE}_server.log" 2>&1 &
  local SERVER_PID=$!
  echo "[exp04d:$SUITE] Server PID=$SERVER_PID, waiting 150s..."
  sleep 150

  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[exp04d:$SUITE] ERROR: Server died. Check $OUT/${SUITE}_server.log"
    return 1
  fi

  echo "[exp04d:$SUITE] Running client ($EP ep/task)..."
  python evaluation/libero/client.py \
    --libero-benchmark "$SUITE" \
    --port $PORT \
    --test-num "$EP" \
    --task-range 0 10 \
    --out-dir "$OUT/${SUITE}" \
    2>&1 | tee "$OUT/${SUITE}.log" || echo "[exp04d:$SUITE] FAILED"

  kill $SERVER_PID 2>/dev/null || true
  echo "[exp04d:$SUITE] Done."
}

echo "[exp04d-parallel] Starting 4 suites on GPUs 3/5/6/7, $EP ep each..."

(run_suite 3 libero_spatial 29060 29070) &
PID_SP=$!

(run_suite 5 libero_object 29061 29071) &
PID_OB=$!

(run_suite 6 libero_goal 29062 29072) &
PID_GO=$!

(run_suite 7 libero_10 29063 29073) &
PID_10=$!

echo "[exp04d-parallel] PIDs: spatial=$PID_SP object=$PID_OB goal=$PID_GO 10=$PID_10"
wait $PID_SP $PID_OB $PID_GO $PID_10
echo "[exp04d-parallel] All suites complete."
