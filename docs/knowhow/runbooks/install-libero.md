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

### LIBERO assets (scene XMLs + 3D models) NOT in PyPI (CRITICAL, 2026-04-29)

PyPI `libero==0.1.1` wheel 不包含 `assets/` 目录（mujoco scene XML, .obj, .mtl, textures）。实际跑 LIBERO eval 时会报 `FileNotFoundError: libero_tabletop_base_style.xml`。

**LIBERO v0.1.1 内置了 HF 自动下载** (`download_assets_from_huggingface()` → `jadechoghari/libero-assets`)，但 xdlab23 上 hf-mirror 会触发 429 rate limit（586 个小文件）。

**解法 — 本地 clone + scp**:
```bash
# 1. 本地 macOS (能访问 GitHub):
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/libero-git
cd /tmp/libero-git && git sparse-checkout set libero/libero/assets && git checkout
tar czf /tmp/libero-assets-full.tar.gz -C /tmp/libero-git/libero/libero assets

# 2. scp 到服务器:
scp /tmp/libero-assets-full.tar.gz xdlab23_yang:/tmp/

# 3. 服务器上部署到每个 conda env:
ssh xdlab23_yang 'cd /tmp && tar xzf libero-assets-full.tar.gz
  cp -r assets /home/ybyang/miniconda3/envs/fastwam/lib/python3.10/site-packages/libero/libero/
  cp -r assets /home/ybyang/miniconda3/envs/vit-probe/lib/python3.12/site-packages/libero/libero/'
```

**验证**: `ls /path/to/site-packages/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml`

**注意**: HF repo `jadechoghari/libero-assets` 是第三方上传，**缺 `stable_scanned_objects/` 和 `textures/`**。必须从 GitHub 原始 repo 获取完整 assets。

**另一个 hf-mirror 绕过 429 的方法**: `HF_HUB_DOWNLOAD_CONCURRENCY=1` 降低并发。

### 多 conda env 的 LIBERO 安装 (2026-04-29)

每个 conda env 都需要独立安装 LIBERO + assets:
- `vit-probe` (Python 3.12): Pi-Zero profiling 用
- `fastwam` (Python 3.10): Fast-WAM eval 用

`pip install libero` 装包，但 assets 需要手动部署到每个 env 的 `site-packages/libero/libero/assets/`。

## Installed 2026-04-28, updated 2026-04-29

- 2026-04-28: PyPI install + smoke test (benchmark dict) OK
- 2026-04-29: Assets 缺失发现 + 解决 (GitHub clone → scp → 1370 files 部署). Fast-WAM LIBERO eval 跑通 (spatial task0 95% success)
