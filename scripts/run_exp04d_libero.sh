#!/usr/bin/env bash
# exp04d: LingBot-VA LIBERO-4 eval (server-client mode).
#
# LingBot-VA uses a server-client architecture (wan_va_server.py + client.py).
# This script starts the server, waits for warmup, runs all 4 suites, then kills server.
#
# Usage:
#   bash scripts/run_exp04d_libero.sh          # smoke test (2 ep)
#   bash scripts/run_exp04d_libero.sh 20       # full eval (20 ep)
#   bash scripts/run_exp04d_libero.sh 20 2     # full eval on GPU 2
set -euo pipefail

EP=${1:-2}
GPU=${2:-2}
PORT=29056
MASTER_PORT=29061

LVA=/data1/ybyang/lingbot-va
CKPT=/data1/ybyang/huggingface/robbyant/lingbot-va-posttrain-libero-long
OUT=/data1/ybyang/vlla/exp/exp04d
SUITES="libero_spatial libero_object libero_goal libero_10"

export MUJOCO_GL=egl
export HF_HOME=/data1/ybyang/huggingface
export LD_LIBRARY_PATH=/home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUT"

# Activate conda
eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
conda activate vit-probe

cd "$LVA"

echo "[exp04d] Starting LingBot-VA server on GPU $GPU, port $PORT..."
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.run \
  --nproc_per_node 1 --master_port $MASTER_PORT \
  wan_va/wan_va_server.py \
  --config-name libero \
  --port $PORT \
  --save_root "$OUT/vis" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "[exp04d] Server PID=$SERVER_PID, waiting 120s for model load..."
sleep 120

# Check server is alive
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "[exp04d] ERROR: Server died during warmup. Check $OUT/server.log"
  exit 1
fi

FAILED=0
for suite in $SUITES; do
  echo "[exp04d] $suite (GPU $GPU, $EP ep/task)"
  python evaluation/libero/client.py \
    --libero-benchmark "$suite" \
    --port $PORT \
    --test-num "$EP" \
    --task-range 0 10 \
    --out-dir "$OUT/${suite}" \
    2>&1 | tee "$OUT/${suite}.log" || { echo "[exp04d] $suite FAILED"; FAILED=1; }
done

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

if [ $FAILED -eq 0 ]; then
  echo "[exp04d] All suites complete."
else
  echo "[exp04d] Some suites failed. Check logs in $OUT/"
fi

# Consolidate results
cd /data1/ybyang/vlla
python -c "
import json, os, glob
out = '$OUT'
suites = '$SUITES'.split()
results = {}
for suite in suites:
    pattern = os.path.join(out, suite, '*.json')
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        results[suite] = data
with open(os.path.join(out, 'libero_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Results consolidated to', os.path.join(out, 'libero_results.json'))
" 2>/dev/null || echo "[exp04d] Result consolidation skipped (no results yet)"
