#!/bin/bash
# Setup Pi-Zero (allenzren/open-pi-zero) environment on xdlab23 using uv
#
# Usage (run on xdlab23):
#   bash /data1/ybyang/vlla/scripts/setup_pizero.sh
#
# Strategy: Upstream uses `src/` as package root with internal imports
#   `from src.model...`. We rename to `pizero_src/` AND sed-rewrite all
#   internal imports, so our own project's `src/` namespace is preserved.
#
# Runtime: pizero_controller adds vendor/open_pi_zero/ to sys.path and
#   imports `from pizero_src.model.vla.pizero import PiZero`.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VENV_DIR="/data1/ybyang/vlla/.venvs/pizero"
PYTHON_VERSION="3.10"
HF_CACHE="/data1/ybyang/huggingface"

VENDOR_DIR="/data1/ybyang/vlla/vendor/open_pi_zero"
REPO_URL="https://github.com/allenzren/open-pi-zero"

# Pinned to upstream pyproject.toml
TORCH_VERSION="2.5.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# =============================================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }
error_exit() { echo "[ERROR] $*" >&2; exit 1; }

# =============================================================================
# Step 0: Ensure uv is available
# =============================================================================

log "=== Pi-Zero Environment Setup (uv) ==="

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv --version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true

# =============================================================================
# Step 1: Create venv (Python 3.10 — matches upstream pyproject.toml pin)
# =============================================================================

if [ -d "$VENV_DIR" ]; then
    log "Removing existing venv at $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi

log "Creating venv with Python ${PYTHON_VERSION}..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

source "$VENV_DIR/bin/activate"
log "Python: $(python --version)"

# =============================================================================
# Step 2: Install PyTorch 2.5.0 + CUDA 12.1 (upstream pin)
# =============================================================================

log "Installing PyTorch ${TORCH_VERSION} (cu121)..."
uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"

# =============================================================================
# Step 3: Core dependencies (upstream pyproject.toml pins + controller needs)
# =============================================================================

log "Installing core dependencies (upstream-pinned)..."
uv pip install \
    "transformers==4.47.1" \
    "bitsandbytes==0.45.0" \
    "numpy==1.26.4" \
    "tensorflow==2.15.0" \
    "tensorflow_datasets==4.9.2" \
    "protobuf==3.20.3" \
    hydra-core \
    omegaconf \
    einops \
    safetensors \
    wandb \
    easydict \
    huggingface_hub \
    accelerate \
    sentencepiece \
    pillow

# =============================================================================
# Step 4: Clone open-pi-zero repo into vendor/
# =============================================================================

log "Preparing $VENDOR_DIR..."
mkdir -p "$(dirname "$VENDOR_DIR")"

if [ -d "$VENDOR_DIR/.git" ]; then
    log "Vendor repo exists — pulling latest..."
    (cd "$VENDOR_DIR" && git pull --ff-only) || {
        log "WARNING: git pull failed (firewall?). Using existing checkout."
    }
elif [ -d "$VENDOR_DIR" ] && [ "$(ls -A "$VENDOR_DIR" 2>/dev/null)" ]; then
    log "Vendor dir exists (non-git). Leaving as-is."
else
    git clone "$REPO_URL" "$VENDOR_DIR" || error_exit \
        "git clone failed. xdlab23 may be firewalled from GitHub. \
Clone locally and scp -r to $VENDOR_DIR, then rerun from Step 5."
fi

# =============================================================================
# Step 5: Rename src/ -> pizero_src/ and rewrite internal imports
# =============================================================================
#
# Upstream uses `src/` as package root with internal imports like:
#   from src.model.vla.pizero import ...
# Our own project already has `src/`. Renaming upstream's package to
# `pizero_src/` avoids namespace collision.
#
# This step is idempotent: if already renamed, skip.

if [ -d "$VENDOR_DIR/src" ] && [ ! -d "$VENDOR_DIR/pizero_src" ]; then
    log "Renaming $VENDOR_DIR/src -> $VENDOR_DIR/pizero_src..."
    mv "$VENDOR_DIR/src" "$VENDOR_DIR/pizero_src"

    log "Rewriting 'from src.' / 'import src.' -> 'pizero_src.' in .py files..."
    # macOS and Linux sed differ on -i flag; use portable in-place edit
    find "$VENDOR_DIR/pizero_src" "$VENDOR_DIR/scripts" "$VENDOR_DIR/slurm" \
        -name "*.py" -type f 2>/dev/null | while read -r f; do
        # Create temp file, sed, replace (portable)
        python - <<PYEOF "$f"
import sys, re
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    content = fh.read()
new = re.sub(r"(^|\n)(\s*)(from|import)\s+src(\.|\s)",
             r"\1\2\3 pizero_src\4", content)
if new != content:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"  rewrote {path}")
PYEOF
    done

    # Ensure __init__.py at package root
    [ -f "$VENDOR_DIR/pizero_src/__init__.py" ] || touch "$VENDOR_DIR/pizero_src/__init__.py"

elif [ -d "$VENDOR_DIR/pizero_src" ]; then
    log "pizero_src/ already present — skipping rename."
else
    error_exit "Neither src/ nor pizero_src/ found in $VENDOR_DIR. Check clone."
fi

# =============================================================================
# Step 6: Env vars (HF cache)
# =============================================================================
#
# Controller injects vendor path at runtime via sys.path.insert().
# HF_HOME is exported at invocation time (not persisted in venv).

log "HF cache: $HF_CACHE (export HF_HOME=\"$HF_CACHE\" when running)"

# =============================================================================
# Step 7: Verify
# =============================================================================

log "=== Verification ==="
python - <<PYEOF
import sys
sys.path.insert(0, "$VENDOR_DIR")

import torch
print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

import transformers; print(f"transformers: {transformers.__version__}")
import easydict; print("easydict: OK")
import omegaconf; print(f"omegaconf: {omegaconf.__version__}")
import safetensors; print("safetensors: OK")
import sentencepiece; print("sentencepiece: OK")

# Deep import — the one pizero_controller actually uses
try:
    from pizero_src.model.vla.pizero import PiZero  # noqa: F401
    print("pizero_src.model.vla.pizero.PiZero: OK")
except Exception as e:
    print(f"pizero_src import FAILED: {e}")
    print("  -> Check $VENDOR_DIR layout and Step 5 rename/rewrite.")
    sys.exit(1)

print("=== OK ===")
PYEOF

# =============================================================================
# Step 8: Final info
# =============================================================================

log ""
log "=== Setup Complete ==="
log "Vendor: $VENDOR_DIR  (controller adds to sys.path at runtime)"
log "Activate:"
log "  source $VENV_DIR/bin/activate"
log ""
log "Run profiling (random weights — no checkpoint needed):"
log "  export HF_HOME=$HF_CACHE"
log "  cd /data1/ybyang/vlla"
log "  python src/run_tasks.py +experiment=pizero/profiling"
log ""
log "For real checkpoints (demo/inference):"
log "  bash scripts/download_pizero_ckpt.sh [bridge_beta|fractal_beta|...]"
log "  # Also needs PaliGemma base (gated): --with-paligemma flag"
