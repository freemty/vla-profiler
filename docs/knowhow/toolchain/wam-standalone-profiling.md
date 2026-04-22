# WAM Standalone Profiling Patterns

> WAM (World Action Model) profiling 使用 standalone script 而非 Hydra controller — 适用于有自包含推理 pipeline 的外部模型。

## Problem

Fast-WAM 和 LingBot-VA 都有完整的推理 pipeline (model init → encode → denoise loop → output)，强行集成进 Hydra controller 体系增加了不必要的复杂度，且这些模型的 Python 包不在 vlla 的 import path 中。

## Cause

WAM 模型通常是独立 repo，有自己的 model class / scheduler / config 体系。与 VLM/VLA controller 不同，WAM 推理流程涉及多阶段 denoise loop (video + action)，无法简单映射到 E/C/A 三阶段。

## Solution

### 1. Standalone Script 模式

为每个 WAM 写独立 profiling 脚本 (`scripts/profile_*.py`)，在目标 repo 的 conda env 中运行：

```python
# scripts/profile_fastwam.py — 在 fastwam conda env 中运行
sys.path.insert(0, str(Path.cwd()))  # 在 FastWAM repo 根目录运行
from fastwam.models.wan22.fastwam import FastWAM
```

### 2. Random-Init 策略

Timing 取决于 compute graph (operators, shapes, dtypes)，不取决于 weight values。Random-init 节省 12GB+ checkpoint 下载：

```python
def build_model_random(device, dtype):
    """Build with random weights — same architecture, valid for timing."""
    video_expert = WanVideoDiT(hidden_dim=3072, num_layers=30, ...)
    action_expert = ActionDiT(hidden_dim=1024, num_layers=30, ...)
    # No checkpoint loading needed
```

### 3. sys.path Injection

WAM repo 的 Python 包不在 vlla 的 venv 中。使用 `repo_path` 参数注入：

```python
repo_path = getattr(cfg, "repo_path", "/data1/ybyang/lingbot-va")
sys.path.insert(0, repo_path)
from wan_va.modules.model import WanTransformer3DModel
```

**注意:** 不要用 `model_name` (权重路径) 替代 `repo_path` (代码路径)。

### 4. Text Encoder 省略

Text encoding 是 one-time cost (对话开始时预计算)，不在 control loop hot path 中。Profiling 使用 dummy embeddings：

```python
dummy_context = torch.randn(1, 20, 4096, device=device, dtype=dtype)
```

Full mode 也不需要加载 text encoder (~4.7B, ~9GB VRAM)。

### 5. Phase Timing 模式

WAM 无法用 module hooks 区分 video/action 阶段（同一 transformer 被两个 loop 调用）。使用手动 timer marks：

```python
self.timer.mark_start("video_denoise")
for step in video_timesteps:
    transformer(input_dict, action_mode=False)
self.timer.mark_end("video_denoise")
```

## Commands

```bash
# Fast-WAM (E/C/A profiling)
conda activate fastwam
cd /data1/ybyang/FastWAM
python /data1/ybyang/vlla/scripts/profile_fastwam.py --mode random --gpu 0

# LingBot-VA (E/V/A profiling)
conda activate vit-probe
cd /data1/ybyang/lingbot-va
python /data1/ybyang/vlla/scripts/profile_lingbot_va.py --mode random --gpu 0
```

## Notes
- Date: 2026-04-21
- Environment: xdlab23 (RTX 5880 Ada), fastwam conda env / vit-probe conda env
- Related: `scripts/profile_fastwam.py`, `scripts/profile_lingbot_va.py`
- Per-step cost baseline: Fast-WAM ActionDiT ~32ms/step, LingBot-VA WanDiT ~29ms/step
