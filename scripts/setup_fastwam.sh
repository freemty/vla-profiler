#!/bin/bash
# Setup Fast-WAM for profiling on xdlab23
# Usage: bash scripts/setup_fastwam.sh
set -eo pipefail

FASTWAM_DIR="/data1/ybyang/FastWAM"
CONDA_ENV="fastwam"

echo "=== Step 1: Clone Fast-WAM repo ==="
if [ -d "$FASTWAM_DIR" ]; then
    echo "Fast-WAM already cloned at $FASTWAM_DIR, pulling latest..."
    cd "$FASTWAM_DIR" && git pull
else
    git clone https://github.com/yuantianyuan01/FastWAM.git "$FASTWAM_DIR"
fi

echo "=== Step 2: Create conda env ==="
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Conda env '$CONDA_ENV' already exists, skipping creation."
else
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

echo "=== Step 3: Install dependencies ==="
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
cd "$FASTWAM_DIR"
pip install -e .

echo "=== Step 4: Download checkpoint ==="
pip install -U huggingface_hub
mkdir -p checkpoints/fastwam_release

huggingface-cli download yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --local-dir ./checkpoints/fastwam_release

echo "=== Step 5: Prepare ActionDiT backbone ==="
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    --device cuda \
    --dtype bfloat16

echo "=== Setup complete ==="
echo "Checkpoint: $FASTWAM_DIR/checkpoints/fastwam_release/"
echo "Next: run profiling with 'conda activate fastwam && python profile_fastwam.py'"
