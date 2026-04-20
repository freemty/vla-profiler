#!/bin/bash
# Setup LingBot-VLA environment on xdlab23 using uv
#
# Usage (run on xdlab23):
#   bash /data1/ybyang/vlla/scripts/setup_lingbot_vla.sh
#
# NOTE: xdlab23 cannot access GitHub (blocked by firewall).
#   For flash-attn wheel, download locally and scp if PyPI install fails.
#
# IMPORTANT: This creates a NEW venv via uv in .venvs/lingbot-vla/
#   Do NOT modify the existing 'vit-probe' conda env.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VENV_DIR="/data1/ybyang/vlla/.venvs/lingbot-vla"
PYTHON_VERSION="3.12"
HF_CACHE="/data1/ybyang/huggingface"

# =============================================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }
error_exit() { echo "[ERROR] $*" >&2; exit 1; }

# =============================================================================
# Step 0: Ensure uv is available
# =============================================================================

log "=== LingBot-VLA Environment Setup (uv) ==="

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv --version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true

# =============================================================================
# Step 1: Create venv
# =============================================================================

if [ -d "$VENV_DIR" ]; then
    log "Removing existing venv at $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi

log "Creating venv with Python ${PYTHON_VERSION}..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

# Activate
source "$VENV_DIR/bin/activate"
log "Python: $(python --version)"

# =============================================================================
# Step 2: Install PyTorch + CUDA 12.8
# =============================================================================

log "Installing PyTorch 2.8.0 + CUDA 12.8..."
uv pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# Step 3: flash-attn
# =============================================================================

log "Installing flash-attn..."
uv pip install flash-attn --no-build-isolation || {
    log "WARNING: flash-attn install failed. Will try sdpa fallback."
    log "  Manual fix: download wheel from https://github.com/Dao-AILab/flash-attention/releases"
    log "  Then: uv pip install /path/to/flash_attn-*.whl"
}

# =============================================================================
# Step 4: Core dependencies
# =============================================================================

log "Installing core dependencies..."
uv pip install \
    transformers==4.55.2 \
    diffusers==0.36.0 \
    accelerate \
    websockets \
    einops \
    msgpack \
    opencv-python \
    matplotlib \
    ftfy \
    easydict \
    qwen-vl-utils \
    huggingface_hub \
    safetensors \
    omegaconf \
    hydra-core \
    ipdb

# =============================================================================
# Step 4b: lingbotvla + lerobot (from source)
# =============================================================================

log "Installing lingbotvla (from local clone)..."
LINGBOT_REPO="/data1/ybyang/lingbot-vla"
if [ -d "$LINGBOT_REPO" ]; then
    uv pip install -e "$LINGBOT_REPO"
else
    log "WARNING: lingbotvla repo not found at $LINGBOT_REPO"
    log "  Clone it first: git clone git@github.com:Robbyant/lingbot-vla.git $LINGBOT_REPO"
fi

log "Installing lerobot..."
uv pip install lerobot || {
    log "WARNING: lerobot install failed. Install manually."
}

# =============================================================================
# Step 5: Verify
# =============================================================================

log "=== Verification ==="
python -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
try:
    import flash_attn; print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print('flash-attn: NOT INSTALLED')
import transformers; print(f'transformers: {transformers.__version__}')
print('=== OK ===')
"

# =============================================================================
# Step 6: Model download commands
# =============================================================================

log ""
log "=== Model Download (run manually) ==="
cat << 'EOF'

# Activate first:
source /data1/ybyang/vlla/.venvs/lingbot-vla/bin/activate

# Main model (downloads to ModelScope path matching profiling.yaml config)
# Config expects: /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b
pip install modelscope
modelscope download --model Robbyant/lingbot-vla-4b \
    --local_dir /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b

# Alternative: HuggingFace (need to update config model_name if using this path)
# huggingface-cli download robbyant/lingbot-vla-4b \
#     --local-dir /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b

EOF

log "=== Setup Complete ==="
log "Activate: source $VENV_DIR/bin/activate"
