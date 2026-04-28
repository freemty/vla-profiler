#!/bin/bash
# Setup Cosmos Policy environment on xdlab23 (no Docker needed)
# Creates a dedicated venv at /data1/ybyang/vlla/.venvs/cosmos-policy/
#
# Usage: bash scripts/setup_cosmos_policy.sh
set -euo pipefail

VLLA_ROOT="/data1/ybyang/vlla"
COSMOS_ROOT="$VLLA_ROOT/vendor/cosmos-policy"
VENV_DIR="$VLLA_ROOT/.venvs/cosmos-policy"
HF_HOME="/data1/ybyang/huggingface"

export HF_HOME
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "============================================"
echo "Setting up Cosmos Policy environment"
echo "============================================"

# Step 1: Create venv with Python 3.10
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv..."
    python3.10 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Venv exists, reusing."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools 2>&1 | tail -1

# Step 2: Install cosmos-policy in editable mode with minimal deps
echo "[2/4] Installing cosmos-policy..."
cd "$COSMOS_ROOT"

# Install core deps first (avoid megatron-core which needs special handling)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

# Install cosmos-policy itself (editable)
pip install -e "." 2>&1 | tail -5

echo "[3/4] Verifying imports..."
python -c "
from cosmos_policy.experiments.robot.cosmos_utils import get_model, get_action
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
print('OK — cosmos_policy imports work')
"

# Step 4: Download checkpoint (if not cached)
echo "[4/4] Downloading Cosmos Policy LIBERO checkpoint..."
python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    'nvidia/Cosmos-Policy-LIBERO-Predict2-2B',
    cache_dir=os.environ.get('HF_HOME'),
    resume_download=True,
)
print(f'Checkpoint at: {path}')
"

echo ""
echo "============================================"
echo "Setup complete! Run profiling with:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $VLLA_ROOT/scripts/exp09a_cosmos_policy_profiling.py --gpu 0"
echo "============================================"
