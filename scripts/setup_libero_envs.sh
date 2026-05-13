#!/usr/bin/env bash
# Setup isolated venvs for LIBERO evaluation on xdlab23.
#
# Each model may need a different transformers version:
#   - lingbot-vla-eval: transformers 4.51.3 (lingbotvla requirement)
#   - vit-probe (conda): transformers 4.57.1 (Pi-Zero, Cosmos, Fast-WAM)
#
# Usage: bash scripts/setup_libero_envs.sh
set -euo pipefail

UV=/home/ybyang/.local/bin/uv
VENVS=/data1/ybyang/vlla/.venvs

########################################
# lingbot-vla-eval: transformers 4.51.3
########################################
setup_lingbot_vla_eval() {
  local VENV=$VENVS/lingbot-vla-eval
  echo "[setup] Creating lingbot-vla-eval venv..."

  $UV venv "$VENV" --python 3.12
  $UV pip install --python "$VENV/bin/python" \
    torch torchvision \
    transformers==4.51.3 \
    numpy==1.26.4 \
    datasets==3.6.0 \
    safetensors \
    easydict \
    Pillow \
    mujoco==3.8.0

  # Install lingbotvla (editable, pulls lerobot matching transformers 4.51)
  $UV pip install --python "$VENV/bin/python" \
    -e /data1/ybyang/lingbot-vla

  # Install LIBERO
  if [ -d /data1/ybyang/LIBERO ]; then
    $UV pip install --python "$VENV/bin/python" \
      -e /data1/ybyang/LIBERO
  fi

  echo "[setup] lingbot-vla-eval ready at $VENV"
  $VENV/bin/python -c "
import transformers; print(f'  transformers: {transformers.__version__}')
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy; print('  LingbotVlaPolicy: OK')
"
}

########################################
# Verify vit-probe (conda) has what Pi-Zero/Cosmos need
########################################
verify_vit_probe() {
  echo "[setup] Verifying vit-probe conda env..."
  eval "$(/home/ybyang/miniconda3/bin/conda shell.bash hook)"
  conda activate vit-probe
  python -c "
import transformers; print(f'  transformers: {transformers.__version__}')
import mujoco; print(f'  mujoco: {mujoco.__version__}')
import libero; print('  libero: OK')
"
}

echo "=== LIBERO Eval Environment Setup ==="
setup_lingbot_vla_eval
verify_vit_probe
echo "=== Setup complete ==="
