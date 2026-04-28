# Install LIBERO on xdlab23

## Problem

LIBERO (robot manipulation benchmark, MuJoCo-based) is needed for VLA eval
reproducibility (exp09). The xdlab23 server has no GitHub access (firewall),
but PyPI is reachable via the Tsinghua mirror configured in pip.conf.

## Solution

Install LIBERO from PyPI into the existing `vit-probe` conda env (Python 3.12).

```bash
# From local machine (one-liner):
ssh -p 66 xdlab23_yang "bash -c '
  eval \"\$(/home/ybyang/miniconda3/bin/conda shell.bash hook)\" \
  && conda activate vit-probe \
  && cd /data1/ybyang/vlla \
  && bash scripts/install_libero.sh
'"

# Or SSH in and run manually:
ssh xdlab23_yang            # port 66 aliased in ~/.ssh/config
conda activate vit-probe
cd /data1/ybyang/vlla
bash scripts/install_libero.sh
```

The script is idempotent — re-running it skips the pip install if LIBERO is
already importable.

## What gets installed

| Package | Version | Purpose |
|---------|---------|---------|
| libero | 0.1.1 | Benchmark suite (130 tasks, 6 suites) |
| robosuite | 1.4.0 | MuJoCo sim environment |
| mujoco | 3.8.0 | Physics engine (replaces old mujoco-py) |
| robomimic | 0.2.0 | Offline RL baselines |
| bddl | 1.0.1 | Behavior description language |
| hf_egl_probe | 1.0.2 | EGL rendering probe (headless GPU) |

## Gotchas

### egl_probe cmake failure (SOLVED)

`robosuite==1.4.0` depends on `egl_probe`, whose `CMakeLists.txt` requires
cmake < 3.5 compat — incompatible with cmake 4.x on the server. The install
script works around this by:

1. Installing `robosuite` with `--no-deps`
2. Installing `hf_egl_probe` instead (same functionality, builds fine)
3. Installing remaining deps individually

The pip warning `robomimic 0.2.0 requires egl_probe>=1.0.1, which is not
installed` is harmless — `hf_egl_probe` covers the same functionality.

### LIBERO interactive prompt (SOLVED)

On first import, LIBERO asks interactively for a dataset path if
`~/.libero/config.yaml` doesn't exist. The install script pre-creates this
config using `importlib.util.find_spec` to locate the install path without
triggering the import.

Dataset path: `/data1/ybyang/libero_datasets`

### openpi third_party/libero is empty

The openpi repo has LIBERO as a git submodule pointing to GitHub, but the
submodule was never initialized (GitHub is blocked). The PyPI install (v0.1.1)
is functionally equivalent and simpler to maintain.

### Rendering backend

The server is headless. Use `MUJOCO_GL=egl` (GPU-accelerated) or
`MUJOCO_GL=osmesa` (CPU software rendering, slower).

```bash
# EGL (preferred, uses GPU):
MUJOCO_GL=egl python my_eval.py

# osmesa (fallback if EGL fails):
MUJOCO_GL=osmesa python my_eval.py

# glx requires a display — does NOT work headless.
```

If EGL fails with "EGL device not found", check:
1. `nvidia-smi` works (driver loaded)
2. `libegl1-mesa-dev` or `libegl-dev` is installed (may need sudo)
3. Try `PYOPENGL_PLATFORM=egl` alongside `MUJOCO_GL=egl`

### TMPDIR for long evals

MuJoCo writes temp files. On long eval runs `/tmp` can fill up. Set TMPDIR:

```bash
export TMPDIR=/data1/ybyang/tmp
mkdir -p "$TMPDIR"
```

### HuggingFace downloads

LIBERO task datasets are downloaded from HuggingFace.
If HF downloads fail, set the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Verification

```bash
conda activate vit-probe
MUJOCO_GL=egl python -c "
from libero.libero import benchmark
b = benchmark.get_benchmark_dict()
print('LIBERO suites:', list(b.keys()))
# Expected: ['libero_spatial', 'libero_object', 'libero_goal',
#            'libero_90', 'libero_10', 'libero_100']
"
```

## Installed 2026-04-28

Verified working on xdlab23 vit-probe env (Python 3.12, cmake 4.1.3).
