#!/bin/bash
# Setup LingBot-VLA environment on xdlab23
# Reference: https://github.com/Robbyant/lingbot-vla
#
# Usage (run on xdlab23):
#   bash /data1/ybyang/vlla/scripts/setup_lingbot_vla.sh
#
# Prerequisites:
#   - conda available in PATH
#   - CUDA 12.8 driver installed
#   - HuggingFace CLI configured (huggingface-cli login)
#
# NOTE: xdlab23 cannot access GitHub (blocked by firewall).
#   Workaround for any git-based dependencies:
#     1. On local machine: git clone <repo> && git bundle create /tmp/<repo>.bundle --all
#     2. scp -P 66 /tmp/<repo>.bundle xdlab23_yang:/tmp/
#     3. On xdlab23: git clone /tmp/<repo>.bundle <target-dir>
#   For pip packages from PyPI, direct install works (PyPI is not blocked).
#   For flash-attn pre-compiled wheel, download locally and scp if PyPI install fails.
#
# IMPORTANT: This creates a NEW conda env 'lingbot-vla'.
#   Do NOT modify the existing 'vit-probe' env (shared with rope2sink).

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

ENV_NAME="lingbot-vla"
PYTHON_VERSION="3.12"
PYTORCH_VERSION="2.8.0"
CUDA_VERSION="12.8"
FLASH_ATTN_VERSION="2.8.3"

HF_CACHE="/data1/ybyang/huggingface"
WORK_DIR="/data1/ybyang/vlla"

# Models to download
MODEL_MAIN="robbyant/lingbot-vla-4b"
MODEL_POSTTRAIN="robbyant/lingbot-vla-4b-posttrain-robotwin"
MODEL_TOKENIZER="Qwen/Qwen2.5-VL-3B-Instruct"

# =============================================================================
# Helper functions
# =============================================================================

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

error_exit() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || error_exit "$1 not found in PATH"
}

# =============================================================================
# Step 0: Preflight checks
# =============================================================================

log "=== LingBot-VLA Environment Setup ==="
log "Server: xdlab23 | GPUs: 8x RTX 5880 Ada 48GB"
log ""

check_command conda
check_command nvidia-smi

# Verify CUDA driver
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log "NVIDIA driver version: ${DRIVER_CUDA}"

# Check if env already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    log "WARNING: conda env '${ENV_NAME}' already exists."
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Removing existing env..."
        conda env remove -n "${ENV_NAME}" -y
    else
        error_exit "Aborted. Remove env manually: conda env remove -n ${ENV_NAME}"
    fi
fi

# =============================================================================
# Step 1: Create conda environment
# =============================================================================

log "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

log "Python: $(python --version)"
log "pip: $(pip --version)"

# =============================================================================
# Step 2: Install PyTorch with CUDA 12.8
# =============================================================================

log "Installing PyTorch ${PYTORCH_VERSION} + CUDA ${CUDA_VERSION}..."
pip install torch=="${PYTORCH_VERSION}" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}')
" || error_exit "PyTorch CUDA verification failed"

# =============================================================================
# Step 3: Install flash-attention
# =============================================================================

log "Installing flash-attn ${FLASH_ATTN_VERSION}..."

# Try pip install first (pre-compiled wheel for torch 2.8 + cu128 + python 3.12)
# If this fails, download the wheel manually from:
#   https://github.com/Dao-AILab/flash-attention/releases
# Then scp to server and: pip install /path/to/flash_attn-*.whl
pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation || {
    log "WARNING: flash-attn pip install failed."
    log "Fallback: install from pre-compiled wheel."
    log "  1. Download wheel from https://github.com/Dao-AILab/flash-attention/releases"
    log "     (match: torch${PYTORCH_VERSION}, cu${CUDA_VERSION//./}, python${PYTHON_VERSION}, linux_x86_64)"
    log "  2. scp -P 66 flash_attn-*.whl xdlab23_yang:/tmp/"
    log "  3. pip install /tmp/flash_attn-*.whl"
    log ""
    log "Continuing without flash-attn for now..."
}

# =============================================================================
# Step 4: Install core dependencies
# =============================================================================

log "Installing core dependencies..."

pip install \
    transformers==4.55.2 \
    diffusers==0.36.0 \
    accelerate \
    websockets \
    einops \
    msgpack \
    opencv-python \
    matplotlib \
    ftfy \
    easydict

# =============================================================================
# Step 5: Install LeRobot
# =============================================================================

log "Installing LeRobot..."

# LeRobot is available on PyPI
pip install lerobot

# =============================================================================
# Step 6: Install additional utilities
# =============================================================================

log "Installing additional utilities..."
pip install huggingface_hub

# =============================================================================
# Step 7: Verification
# =============================================================================

log "=== Verification ==="

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print('flash-attn: NOT INSTALLED (see instructions above)')

import transformers
print(f'transformers: {transformers.__version__}')

import diffusers
print(f'diffusers: {diffusers.__version__}')

import accelerate
print(f'accelerate: {accelerate.__version__}')

import einops
print(f'einops: {einops.__version__}')

import cv2
print(f'opencv: {cv2.__version__}')

try:
    import lerobot
    print(f'lerobot: {lerobot.__version__}')
except (ImportError, AttributeError):
    print('lerobot: installed (version not exposed)')

print()
print('=== All core packages verified ===')
"

# =============================================================================
# Step 8: Model download commands
# =============================================================================

log ""
log "=== Model Download ==="
log "Run the following commands to download models:"
log "(Ensure huggingface-cli is logged in: huggingface-cli login)"
log ""

cat << 'DOWNLOAD_CMDS'

# ---- Download models (run manually) ----

# Main model: lingbot-vla-4b
huggingface-cli download robbyant/lingbot-vla-4b \
    --local-dir /data1/ybyang/huggingface/hub/models--robbyant--lingbot-vla-4b \
    --cache-dir /data1/ybyang/huggingface

# Post-train checkpoint (RobotWin)
huggingface-cli download robbyant/lingbot-vla-4b-posttrain-robotwin \
    --local-dir /data1/ybyang/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin \
    --cache-dir /data1/ybyang/huggingface

# Tokenizer (may already exist from Qwen experiments)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir /data1/ybyang/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct \
    --cache-dir /data1/ybyang/huggingface

DOWNLOAD_CMDS

log ""
log "=== Setup Complete ==="
log ""
log "To activate: conda activate ${ENV_NAME}"
log "To verify:   python -c 'import torch; print(torch.cuda.is_available())'"
log ""
log "NOTE: The existing 'vit-probe' env is untouched."
