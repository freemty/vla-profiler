#!/bin/bash
# Sync vlla code to xdlab23 via git bundle (GitHub blocked by firewall)
# Usage: bash scripts/sync_to_remote.sh
set -e

LOCAL_DIR="/Users/sum_young/code/projects/vlla"
REMOTE_HOST="xdlab23_yang"
REMOTE_DIR="/data1/ybyang/vlla"
BUNDLE_PATH="/tmp/vlla.bundle"

echo "=== Syncing vlla to xdlab23 ==="

# Step 1: Create git bundle
cd "$LOCAL_DIR"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Creating bundle from branch: ${BRANCH}"
git bundle create "$BUNDLE_PATH" "$BRANCH"

# Step 2: Upload bundle
echo "Uploading bundle..."
scp -P 66 "$BUNDLE_PATH" "$REMOTE_HOST:/tmp/"

# Step 3: Also sync submodule (model-probe-core)
echo "Syncing model-probe-core submodule..."
cd "$LOCAL_DIR/src/core"
git bundle create /tmp/model-probe-core.bundle main
scp -P 66 /tmp/model-probe-core.bundle "$REMOTE_HOST:/tmp/"

# Step 4: Apply on remote
echo "Applying on remote..."
ssh -p 66 "$REMOTE_HOST" "
    # Initialize vlla repo if first time
    if [ ! -d '$REMOTE_DIR/.git' ]; then
        git clone /tmp/vlla.bundle '$REMOTE_DIR'
        cd '$REMOTE_DIR'
        # Initialize submodule from bundle
        mkdir -p src/core
        cd src/core
        git clone /tmp/model-probe-core.bundle .
        cd '$REMOTE_DIR'
    else
        cd '$REMOTE_DIR'
        git stash 2>/dev/null || true
        git fetch /tmp/vlla.bundle '$BRANCH':refs/remotes/local/'$BRANCH'
        git merge local/'$BRANCH'
        git stash pop 2>/dev/null || true
        # Update submodule
        cd src/core
        git fetch /tmp/model-probe-core.bundle main:refs/remotes/local/main
        git merge local/main
    fi
"

echo "=== Sync complete ==="
