#!/usr/bin/env bash
# scripts/install_libero.sh — Install LIBERO into vit-probe conda env on xdlab23
#
# Usage (on xdlab23, inside vit-probe env):
#   conda activate vit-probe
#   bash scripts/install_libero.sh
#
# Or from local machine:
#   ssh -p 66 xdlab23_yang "bash -c '
#     eval \"\$(/home/ybyang/miniconda3/bin/conda shell.bash hook)\"
#     && conda activate vit-probe
#     && cd /data1/ybyang/vlla
#     && bash scripts/install_libero.sh
#   '"
#
# Idempotent: re-running skips install if LIBERO is already importable.
set -euo pipefail

CONDA_ENV_NAME="vit-probe"
DATASET_PATH="/data1/ybyang/libero_datasets"

# ── Guard: must be in the right conda env ──────────────────────────
CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ "$CURRENT_ENV" != "$CONDA_ENV_NAME" ]]; then
    echo "[install-libero] ERROR: expected conda env '$CONDA_ENV_NAME', got '$CURRENT_ENV'"
    echo "[install-libero] Run: conda activate $CONDA_ENV_NAME"
    exit 1
fi

# ── Step 1: check if already installed ─────────────────────────────
echo "[install-libero] checking if LIBERO is already importable..."
if python -c "from libero.libero import benchmark" 2>/dev/null; then
    echo "[install-libero] LIBERO already installed, skipping to smoke test."
else
    echo "[install-libero] LIBERO not found, installing..."

    # ── Step 1a: pre-create ~/.libero/config.yaml ──────────────────
    # LIBERO's __init__.py prompts interactively on first import if no config
    # exists. We create it beforehand to avoid the prompt.
    echo "[install-libero] pre-creating LIBERO config (~/.libero/config.yaml)..."
    python -c "
import os, yaml, importlib.util

# Find where LIBERO will be installed (site-packages)
# We use the spec to find the path without triggering the import prompt
spec = importlib.util.find_spec('libero.libero') if importlib.util.find_spec('libero') else None

# If libero is not installed yet, we'll create the config after install
if spec is None:
    print('[install-libero] libero not yet installed, config will be created after install')
else:
    benchmark_root = os.path.dirname(spec.origin)
    config_dir = os.path.expanduser('~/.libero')
    os.makedirs(config_dir, exist_ok=True)
    config = {
        'benchmark_root': benchmark_root,
        'bddl_files': os.path.join(benchmark_root, 'bddl_files'),
        'init_states': os.path.join(benchmark_root, 'init_files'),
        'datasets': '$DATASET_PATH',
        'assets': os.path.join(benchmark_root, 'assets'),
    }
    config_file = os.path.join(config_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    print(f'[install-libero] config written to {config_file}')
" 2>/dev/null || true

    # ── Step 1b: install deps with egl_probe workaround ────────────
    # egl_probe (dep of robosuite) fails to build because its CMakeLists.txt
    # requires cmake < 3.5 compat which cmake 4.x dropped.
    # Workaround: install robosuite --no-deps, then install its deps manually,
    # skipping egl_probe (hf_egl_probe provides equivalent functionality).
    echo "[install-libero] installing hf_egl_probe (EGL rendering probe)..."
    pip install hf_egl_probe

    echo "[install-libero] installing robosuite (--no-deps to skip broken egl_probe)..."
    pip install --no-deps robosuite==1.4.0

    echo "[install-libero] installing robosuite deps (minus egl_probe)..."
    pip install "mujoco>=3.0" numba "PyOpenGL>=3.1" glfw

    echo "[install-libero] installing bddl + robomimic (--no-deps)..."
    pip install --no-deps bddl==1.0.1 robomimic==0.2.0

    echo "[install-libero] installing robomimic/bddl deps..."
    pip install future thop h5py tensorboard tensorboardX

    echo "[install-libero] installing LIBERO (--no-deps, deps already satisfied)..."
    pip install --no-deps libero

    echo "[install-libero] installing extras (video recording)..."
    pip install imageio imageio-ffmpeg

    echo "[install-libero] pip install complete."

    # ── Step 1c: create config now that libero is installed ────────
    echo "[install-libero] creating LIBERO config..."
    mkdir -p "$DATASET_PATH"
    python -c "
import os, yaml, importlib.util

spec = importlib.util.find_spec('libero.libero')
benchmark_root = os.path.dirname(spec.origin)
config_dir = os.path.expanduser('~/.libero')
os.makedirs(config_dir, exist_ok=True)
config = {
    'benchmark_root': benchmark_root,
    'bddl_files': os.path.join(benchmark_root, 'bddl_files'),
    'init_states': os.path.join(benchmark_root, 'init_files'),
    'datasets': '$DATASET_PATH',
    'assets': os.path.join(benchmark_root, 'assets'),
}
config_file = os.path.join(config_dir, 'config.yaml')
with open(config_file, 'w') as f:
    yaml.dump(config, f)
print(f'[install-libero] config written to {config_file}')
"
fi

# ── Step 2: smoke test — import benchmark registry ─────────────────
echo "[install-libero] running smoke test (MUJOCO_GL=egl)..."
MUJOCO_GL=egl python -c "
from libero.libero import benchmark
b = benchmark.get_benchmark_dict()
print('LIBERO suites:', list(b.keys()))
print('[install-libero] smoke test PASSED')
"

echo "[install-libero] done."
