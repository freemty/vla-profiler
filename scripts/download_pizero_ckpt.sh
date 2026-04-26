#!/bin/bash
# Download Pi-Zero (allenzren/open-pi-zero) pretrained checkpoints
#
# NOTE: Profiling uses random weights (no checkpoint needed). This script is
#       for demo / real-inference runs only.
#
# Usage (run on xdlab23):
#   source /data1/ybyang/vlla/.venvs/pizero/bin/activate
#   bash /data1/ybyang/vlla/scripts/download_pizero_ckpt.sh [CKPT] [--with-paligemma]
#
#   CKPT: one of {bridge_uniform, bridge_beta, fractal_uniform, fractal_beta}
#         (default: fractal_beta — recommended)
#
#   --with-paligemma: also download the PaliGemma-3B base weights
#         (requires HUGGINGFACE_HUB_TOKEN with accepted gated license)

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[WARN] $*" >&2; }
error_exit() { echo "[ERROR] $*" >&2; exit 1; }

# =============================================================================
# Configuration
# =============================================================================

CKPT_ROOT="/data1/ybyang/huggingface/pizero-ckpt"
HF_CACHE="/data1/ybyang/huggingface"
HF_REPO_ID="allenzren/open-pi-zero"
PALIGEMMA_REPO_ID="google/paligemma-3b-pt-224"

# Map short key -> canonical filename
declare -A CKPT_FILES=(
    [bridge_uniform]="bridge_uniform_step19296_2024-12-26_22-31_42.pt"
    [bridge_beta]="bridge_beta_step19296_2024-12-26_22-30_42.pt"
    [fractal_uniform]="fractal_uniform_step29576_2024-12-31_22-26_42.pt"
    [fractal_beta]="fractal_beta_step29576_2024-12-29_13-10_42.pt"
)

# =============================================================================
# Parse arguments
# =============================================================================

CKPT_KEY="fractal_beta"
WITH_PALIGEMMA=0

for arg in "$@"; do
    case "$arg" in
        --with-paligemma)
            WITH_PALIGEMMA=1
            ;;
        bridge_uniform|bridge_beta|fractal_uniform|fractal_beta)
            CKPT_KEY="$arg"
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            error_exit "Unknown argument: $arg (expected one of: ${!CKPT_FILES[*]} or --with-paligemma)"
            ;;
    esac
done

CKPT_FILENAME="${CKPT_FILES[$CKPT_KEY]}"

# =============================================================================
# Environment checks
# =============================================================================

log "=== Pi-Zero Checkpoint Download ==="
log "Checkpoint key : $CKPT_KEY"
log "Filename       : $CKPT_FILENAME"
log "Target dir     : $CKPT_ROOT"
log "HF cache       : $HF_CACHE"
log "With PaliGemma : $WITH_PALIGEMMA"

mkdir -p "$CKPT_ROOT"
export HF_HOME="$HF_CACHE"

# Tool availability
HAVE_HF_CLI=0
if command -v huggingface-cli &>/dev/null; then
    HAVE_HF_CLI=1
fi
HAVE_CURL=0
if command -v curl &>/dev/null; then
    HAVE_CURL=1
fi

if [ "$HAVE_HF_CLI" -eq 0 ] && [ "$HAVE_CURL" -eq 0 ]; then
    error_exit "Neither huggingface-cli nor curl is available. Install one of them first: 'uv pip install huggingface_hub' or 'apt install curl'."
fi

# Disk space check (warn if < 15GB free on target FS)
TARGET_FS_PARENT="$(dirname "$CKPT_ROOT")"
if command -v df &>/dev/null; then
    FREE_KB=$(df -Pk "$TARGET_FS_PARENT" | awk 'NR==2 {print $4}')
    FREE_GB=$((FREE_KB / 1024 / 1024))
    log "Free space on $TARGET_FS_PARENT : ${FREE_GB} GB"
    if [ "$FREE_GB" -lt 15 ]; then
        warn "Less than 15 GB free on $TARGET_FS_PARENT — checkpoint is ~11 GB."
    fi
fi

# =============================================================================
# Download Pi-Zero checkpoint
# =============================================================================

DEST_FILE="$CKPT_ROOT/$CKPT_FILENAME"

if [ -f "$DEST_FILE" ]; then
    log "Checkpoint already exists: $DEST_FILE — skipping re-download."
else
    if [ "$HAVE_HF_CLI" -eq 1 ]; then
        log "Downloading via huggingface-cli: $HF_REPO_ID :: $CKPT_FILENAME"
        huggingface-cli download "$HF_REPO_ID" "$CKPT_FILENAME" \
            --local-dir "$CKPT_ROOT" \
            || error_exit "HF download failed. If repo is gated: run 'huggingface-cli login' and accept the license."
    else
        URL="https://huggingface.co/${HF_REPO_ID}/resolve/main/${CKPT_FILENAME}"
        log "Downloading via curl: $URL"
        curl -L --fail -o "$DEST_FILE" "$URL" \
            || error_exit "curl download failed: $URL"
    fi
fi

log "Pi-Zero checkpoint ready at: $DEST_FILE"
if command -v du &>/dev/null; then
    du -h "$DEST_FILE" | awk '{printf "[size] %s  %s\n", $1, $2}'
fi

# =============================================================================
# Optional: PaliGemma-3B base weights
# =============================================================================

if [ "$WITH_PALIGEMMA" -eq 1 ]; then
    log ""
    log "=== PaliGemma-3B base weights ==="

    if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
        warn "HUGGINGFACE_HUB_TOKEN is not set."
        warn "  google/paligemma-3b-pt-224 is a GATED repo."
        warn "  Steps:"
        warn "    1. Visit https://huggingface.co/google/paligemma-3b-pt-224 and accept the license."
        warn "    2. Run 'huggingface-cli login' (or export HUGGINGFACE_HUB_TOKEN=hf_xxx)."
        warn "    3. Re-run this script with --with-paligemma."
        error_exit "Missing HUGGINGFACE_HUB_TOKEN."
    fi

    if [ "$HAVE_HF_CLI" -eq 0 ]; then
        error_exit "huggingface-cli required for PaliGemma download. Install: uv pip install huggingface_hub"
    fi

    PALIGEMMA_DIR="$HF_CACHE/hub/models--google--paligemma-3b-pt-224"
    log "Downloading $PALIGEMMA_REPO_ID to $PALIGEMMA_DIR"
    huggingface-cli download "$PALIGEMMA_REPO_ID" \
        --local-dir "$PALIGEMMA_DIR" \
        || error_exit "PaliGemma download failed. Verify license accepted on HF Hub."

    log "PaliGemma base ready at: $PALIGEMMA_DIR"
    if command -v du &>/dev/null; then
        du -sh "$PALIGEMMA_DIR" | awk '{printf "[size] %s  %s\n", $1, $2}'
    fi
fi

# =============================================================================

log ""
log "=== Download Complete ==="
log "Checkpoint file : $DEST_FILE"
log "Set config      : model_name=\"$DEST_FILE\" in configs/pizero/*.yaml"
log ""
log "Reminder: for PROFILING only, random weights suffice — leave model_name"
log "  empty in the config and skip this script."
