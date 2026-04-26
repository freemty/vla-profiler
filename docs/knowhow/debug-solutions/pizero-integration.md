# Pi-Zero (allenzren/open-pi-zero) Integration

> Pi-Zero dual-stream flow VLA 集成过程中遇到的 5 个陷阱及解决方案

## Problem

集成 allenzren/open-pi-zero (Pi-Zero dual-stream flow VLA, PaliGemma + 300M Gemma Action Expert) 作为 profiling controller 时，遇到一系列特殊问题，包括：
- Vendor 包和项目自身的 `src/` 命名空间冲突
- Base class profiling hooks 与 manual phase timing 冲突
- 自包含 `infer_action()` 导致 hook-based profiling 只能捕获 Encode phase
- xdlab23 非交互 SSH 环境下 conda 不可用
- Profiling task 启动时意外 import `cv2` / `matplotlib` 失败

## Trap 1: Vendor `src/` Namespace Collision

### Cause
allenzren/open-pi-zero 源码以 `src/` 为包根，内部 import 全部写作 `from src.model.vla.pizero import PiZero`。将其作为 `vendor/open_pi_zero/` 引入后，`vendor/open_pi_zero/src/` 与项目自身 `src/` 在 Python 路径上构成两个不同的 `src` 包，后加入者会遮蔽前者，导致 `from src.controllers...` 或 `from src.model...` 行为不可预测。

### Solution
一次性 setup-time rename + sed rewrite：
1. `vendor/open_pi_zero/src/` → `vendor/open_pi_zero/pizero_src/`
2. 对 `pizero_src/**/*.py` 里所有 `from src.` → `from pizero_src.`
3. 将 `vendor/open_pi_zero/` 加入 `sys.path`，使 `from pizero_src.model.vla.pizero import PiZero` 可直接工作
4. Controller 中所有 OmegaConf `_target_` 路径也更新为 `pizero_src.*`

关键：rename 操作必须用绝对路径，不能在项目根目录运行 `mv src pizero_src`（会误改项目自己的 `src/`）。

## Trap 2: Manual Phase Timing vs Base Class Hooks

### Cause
`BaseVLAController.register_profiling_hooks()` 默认在 `vision_encoder.forward` 上注册 forward-hook 调用 `timer.mark_start("encode")`。子类重写 `model_inference` 做手工 E/C/A 分解（显式 `timer.mark_start/end`）时，hook 会先 call `mark_start("encode")`，紧接着 `model_inference` 的手工 `mark_start` 被忽略（已经 started），但 `mark_end` 却会抛错：`KeyError: "Cannot end phase 'encode': mark_start was not called"`（或反之 double start）。

### Solution
在使用 manual E/C/A timing 的 controller 中，override `register_profiling_hooks` 为 no-op：
```python
def register_profiling_hooks(self) -> None:
    self.logger.info(
        "PiZero: skipping hook-based profiling — using manual E/C/A "
        "timing in model_inference (dual-stream architecture requires "
        "explicit phase decomposition)"
    )
```
同模式已在 NitroGenController 使用。适用于所有有自包含 `infer_xxx()` 调用且 base-class hook path 不覆盖全部阶段的模型。

## Trap 3: `model.infer_action()` is Opaque to Hooks

### Cause
`PiZero.infer_action()` 是一个完整封装：SigLIP + Gemma prefill + N 步 flow denoise 全在函数内部。Hook-on-module 模式只能拿到 vision_encoder 的时间，Context 和 Action phase 无法切分。

### Solution
不调用 `model.infer_action()`，改为在 controller 的 `model_inference` 中手工复现其内部流程，并在每个 phase 边界插入 `timer.mark_start/end`：
- Phase E: `model._forward_siglip_and_text_embedding()` + `model.proprio_encoder()`
- Phase C: `model.joint_model(... return_caches=True)` — 构建 KV cache
- Phase A: loop `model.num_inference_steps` 次 `model.joint_model(... cache_mode="append_non_active")` + `model.action_decoder()`

每次 `mark_start/end` 前后加 `torch.cuda.synchronize()` 保证测量准确。

## Trap 4: xdlab23 Non-Interactive SSH — conda Unavailable

### Cause
xdlab23 默认 shell 是 sh (非 bash)，conda init 写在 `.bashrc` 里，非交互 SSH session 不 source 它 — `conda activate` 不可用。此前 LingBot-VLA 已踩过此坑。

### Solution
Pi-Zero 使用独立的 uv venv：`.venvs/pizero/`，Python 3.10 + torch 2.5.0+cu121。`scripts/setup_pizero.sh` 自动化全过程：uv venv create → 克隆 allenzren repo → 执行 Trap 1 的 rename/sed → 安装 upstream pinned deps (transformers==4.47.1 等)。

uv 装在 `~/.local/bin/`，不依赖 shell init；`uv pip install` 通过 `VIRTUAL_ENV=.venvs/pizero` 环境变量定位 venv，无需 `source activate`。

## Trap 5: Profiling Task Imports Pull In cv2 / matplotlib

### Cause
`src/run_tasks.py` 在启动时 import 所有已注册 task：`import src.tasks.attention_overlay_task` — 该 task 在模块顶层 `import cv2` 和间接引用 `matplotlib`。Pi-Zero uv venv 按 upstream 依赖安装，不含这两个包，导致 `ModuleNotFoundError: No module named 'cv2'` 在真正跑 profiling 之前就挂了。

### Solution
在 Pi-Zero venv 中补装：
```bash
VIRTUAL_ENV=/data1/ybyang/vlla/.venvs/pizero uv pip install opencv-python-headless matplotlib scipy
```
注意 `opencv-python-headless` 会把 numpy 从 1.26.4 升到 2.2.6（与 upstream pin 不一致），但 Pi-Zero profiling 路径不触发 numpy 不兼容代码，可忽略警告。

长期方案：改 `run_tasks.py` 为 lazy import (每个 task 首次被 dispatch 时才 import)，避免不相关 task 的依赖污染其他 env。

## Commands

```bash
# Setup (one-time on xdlab23)
bash scripts/setup_pizero.sh

#补装 profiling 依赖
VIRTUAL_ENV=/data1/ybyang/vlla/.venvs/pizero uv pip install \
  opencv-python-headless matplotlib scipy

# 运行 profiling
bash scripts/launch_pizero.sh 0 pizero/profiling
```

## Notes
- Date: 2026-04-26
- Environment: xdlab23 (RTX 5880 Ada 48GB), uv venv `.venvs/pizero/`
- Related: `nitrogen-controller-deployment.md` (same manual-timing pattern), `lingbotvla-integration.md` (same uv-over-conda rationale)
- Backend: allenzren/open-pi-zero (NOT Physical-Intelligence/openpi — that path was abandoned)
