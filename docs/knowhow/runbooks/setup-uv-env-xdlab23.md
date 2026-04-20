# Setup uv-based venv on xdlab23

> 用 uv 替代 conda 创建独立 Python 环境 (更快、无 conda init 依赖)。

## Problem
xdlab23 的 conda 在非交互 SSH 下不可用 (`conda not found in PATH`)，因为 `.bashrc` 的 conda init 块在非登录 shell 不加载。

## Cause
conda 依赖 `.bashrc` 中的 `conda init bash` 块来配置 PATH。SSH 非交互模式 (`ssh host "cmd"`) 不 source `.bashrc`，导致 `conda` 不在 PATH 中。即使用 `bash -l` 登录 shell 也因 nvm 的 `.npmrc` 冲突有问题。

## Solution

使用 uv (Rust-based Python manager) 替代 conda:
- 不依赖 shell init
- 安装在 `~/.local/bin/` (已在 PATH)
- 创建标准 venv (无需 `conda activate` 等 shell hook)

### 安装 uv (一次性)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 安装到 ~/.local/bin/uv
```

### 创建 venv
```bash
uv venv /data1/ybyang/vlla/.venvs/<env-name> --python 3.12
source /data1/ybyang/vlla/.venvs/<env-name>/bin/activate
```

### 安装 PyTorch + CUDA
```bash
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### SSH 非交互调用
```bash
ssh xdlab23_yang "bash -l -c 'export PATH=\$HOME/.local/bin:\$PATH && source /data1/ybyang/vlla/.venvs/<env-name>/bin/activate && python script.py'"
```

## Commands
```bash
# 完整流程 (from local)
ssh xdlab23_yang "bash -l -c 'export PATH=\$HOME/.local/bin:\$PATH && cd /data1/ybyang/vlla && bash scripts/setup_lingbot_vla.sh'"
```

## Notes
- Date: 2026-04-20
- Environment: xdlab23, uv 0.8.23
- venv 目录约定: `/data1/ybyang/vlla/.venvs/<name>/`
- 现有 vit-probe conda env 保持不动 (rope2sink 共享)
- flash-attn 可能需要预编译 wheel (pip install --no-build-isolation 有时失败)
