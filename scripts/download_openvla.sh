#!/bin/bash
# Download OpenVLA-7B from HuggingFace locally, then rsync to xdlab23
# (ModelScope returns 404 for openvla/openvla-7b)
#
# Usage: bash scripts/download_openvla.sh [--local-only] [--remote-only LOCAL_CACHE_DIR]
#   --local-only          Only download to local cache, skip rsync to server
#   --remote-only DIR     Skip local download, rsync from existing DIR to server
#
# Requirements (local machine):
#   pip install huggingface_hub   (provides huggingface-cli)
#   ssh alias xdlab23_yang configured in ~/.ssh/config
set -eo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID="openvla/openvla-7b"
MODEL_EST_SIZE="~16 GB"

# Local: huggingface-cli writes to HF_HOME/hub/models--<org>--<model>/snapshots/...
# We resolve the final snapshot dir after download and pass that path to rsync.
LOCAL_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOCAL_SNAPSHOT_PARENT="${LOCAL_HF_HOME}/hub/models--openvla--openvla-7b"

REMOTE_HOST="xdlab23_yang"
REMOTE_TARGET="/data1/ybyang/huggingface/openvla/openvla-7b"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
LOCAL_ONLY=0
REMOTE_ONLY=0
REMOTE_ONLY_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-only)
            LOCAL_ONLY=1
            shift
            ;;
        --remote-only)
            REMOTE_ONLY=1
            REMOTE_ONLY_DIR="${2:-}"
            if [[ -z "$REMOTE_ONLY_DIR" ]]; then
                echo "ERROR: --remote-only requires a directory path argument" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash $0 [--local-only] [--remote-only LOCAL_CACHE_DIR]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "───────────────────────────────────────────────────────"; }

# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------
hr
echo "=== OpenVLA-7B Download & Upload ==="
hr
echo "  Model   : ${MODEL_ID}"
echo "  Est size: ${MODEL_EST_SIZE}"
echo "  HF cache: ${LOCAL_HF_HOME}"
echo "  Remote  : ${REMOTE_HOST}:${REMOTE_TARGET}"
hr

if [[ "$REMOTE_ONLY" -eq 0 ]]; then
    if ! command -v huggingface-cli &>/dev/null; then
        echo "ERROR: huggingface-cli not found." >&2
        echo "  Install: pip install huggingface_hub" >&2
        exit 1
    fi
fi

if [[ "$LOCAL_ONLY" -eq 0 ]]; then
    if ! ssh -p 66 -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" true 2>/dev/null; then
        echo "ERROR: Cannot reach ${REMOTE_HOST} (port 66). Check SSH config." >&2
        exit 1
    fi
    log "SSH connectivity to ${REMOTE_HOST}: OK"
fi

# ---------------------------------------------------------------------------
# Confirm before proceeding
# ---------------------------------------------------------------------------
if [[ "${CI:-}" != "true" ]]; then
    echo ""
    echo "This will download ${MODEL_EST_SIZE} locally, then rsync to the server."
    echo "Continue? [y/N] "
    read -r CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Download locally via huggingface-cli
# ---------------------------------------------------------------------------
if [[ "$REMOTE_ONLY" -eq 0 ]]; then
    hr
    log "Step 1/2 — Downloading ${MODEL_ID} via huggingface-cli ..."
    log "  HF_HOME=${LOCAL_HF_HOME}"
    log "  (Resume-safe: re-running this command will skip already-downloaded files)"
    hr

    HF_HOME="${LOCAL_HF_HOME}" huggingface-cli download \
        "${MODEL_ID}" \
        --repo-type model \
        --local-dir-use-symlinks False

    log "Download complete."
fi

# ---------------------------------------------------------------------------
# Resolve the local snapshot directory
# ---------------------------------------------------------------------------
if [[ "$REMOTE_ONLY" -eq 1 ]]; then
    # User supplied an explicit path
    LOCAL_MODEL_DIR="${REMOTE_ONLY_DIR}"
    if [[ ! -d "$LOCAL_MODEL_DIR" ]]; then
        echo "ERROR: Supplied directory does not exist: ${LOCAL_MODEL_DIR}" >&2
        exit 1
    fi
else
    # huggingface-cli download without --local-dir stores files under:
    #   $HF_HOME/hub/models--<org>--<model>/snapshots/<hash>/
    # Find the latest snapshot.
    SNAPSHOT_DIR="${LOCAL_SNAPSHOT_PARENT}/snapshots"
    if [[ ! -d "$SNAPSHOT_DIR" ]]; then
        echo "ERROR: Expected snapshot dir not found: ${SNAPSHOT_DIR}" >&2
        echo "  huggingface-cli download may have failed." >&2
        exit 1
    fi

    # Pick the most-recently-modified snapshot hash directory
    LOCAL_MODEL_DIR="$(ls -td "${SNAPSHOT_DIR}"/*/  2>/dev/null | head -1)"
    LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR%/}"   # strip trailing slash

    if [[ -z "$LOCAL_MODEL_DIR" || ! -d "$LOCAL_MODEL_DIR" ]]; then
        echo "ERROR: No snapshot directory found under ${SNAPSHOT_DIR}" >&2
        exit 1
    fi
    log "Resolved local model dir: ${LOCAL_MODEL_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 2: rsync to server
# ---------------------------------------------------------------------------
if [[ "$LOCAL_ONLY" -eq 0 ]]; then
    hr
    log "Step 2/2 — rsyncing to ${REMOTE_HOST}:${REMOTE_TARGET} ..."
    log "  Source : ${LOCAL_MODEL_DIR}/"
    log "  Dest   : ${REMOTE_HOST}:${REMOTE_TARGET}"
    log "  (Resume-safe: interrupted transfers can be re-run; rsync skips unchanged files)"
    hr

    # Create remote target directory first
    ssh -p 66 "$REMOTE_HOST" "mkdir -p '${REMOTE_TARGET}'"

    rsync -avz \
        --progress \
        --partial \
        --checksum \
        -e "ssh -p 66" \
        "${LOCAL_MODEL_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_TARGET}/"

    log "rsync complete."

    # ---------------------------------------------------------------------------
    # Verification: file count & rough size on remote
    # ---------------------------------------------------------------------------
    hr
    log "Verifying remote ..."
    ssh -p 66 "$REMOTE_HOST" "
        echo 'Remote path  : ${REMOTE_TARGET}'
        echo 'File count   :' \$(find '${REMOTE_TARGET}' -type f | wc -l)
        echo 'Disk usage   :' \$(du -sh '${REMOTE_TARGET}' | cut -f1)
    "
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
hr
echo "=== Done ==="
if [[ "$LOCAL_ONLY" -eq 0 ]]; then
    echo ""
    echo "Next steps on xdlab23:"
    echo "  ssh xdlab23_yang"
    echo "  ls ${REMOTE_TARGET}"
    echo ""
    echo "In Hydra config (configs/openvla/...):"
    echo '  model_name: "${oc.env:HF_HOME,/data1/ybyang/huggingface}/openvla/openvla-7b"'
fi
hr
