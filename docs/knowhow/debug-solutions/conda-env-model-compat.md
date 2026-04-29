# Conda Env × Model 兼容性矩阵

> xdlab23 上每个 VLA 模型需要特定 conda env，不能混用。

## Problem

尝试在单个 conda env (vit-probe) 里跑所有 7 个模型的 LIBERO eval，遭遇多种依赖冲突。

## Cause

各模型依赖不同 PyTorch/CUDA 版本、JAX vs PyTorch、flash-attn 版本等。

## 兼容性矩阵 (2026-04-29 实测)

| 模型 | 推荐 env | 可选 env | 不兼容 env | Blocker |
|------|----------|---------|-----------|---------|
| NitroGen | vit-probe | fastwam | — | — |
| Pi-Zero (profiling) | vit-probe | — | fastwam | 我们的 controller 用 PyTorch reimpl |
| Pi-Zero (LIBERO eval) | openpi uv | — | vit-probe, fastwam | `openpi.policies` 需要 `flax` (JAX); `uv sync` 需要 GitHub (被墙) |
| Fast-WAM (profiling) | vit-probe | fastwam | — | profile_fastwam.py 有 random-weight mode |
| Fast-WAM (LIBERO eval) | **fastwam** | — | vit-probe | `from_wan22_pretrained` 需要 fastwam-specific torch+cuda; Hydra config |
| LingBot-VLA | vit-probe | — | fastwam | — |
| LingBot-VA | **需要新 env** | — | vit-probe (cuDNN crash), fastwam (flash-attn build fail) | `flash_attn` + `diffusers` + cuDNN 三方兼容 |
| ACT | vit-probe | — | — | — |
| Qwen-VL | vit-probe | — | — | — |

## Key Lessons

1. **fastwam env 只跑 FastWAM** — 它的 torch==2.7.1+cu128 和 numpy==1.26.4 与其他模型冲突
2. **vit-probe env 是默认** — 但 cuDNN 版本和 LingBot-VA 不兼容 (`cudnnGetLibConfig` undefined symbol)
3. **openpi 需要 uv + GitHub** — 在防火墙后面跑不起来，除非预建 .venv 然后 scp 过去
4. **flash-attn 编译** — 需要匹配的 CUDA toolkit + PyTorch 版本，fastwam env 编译失败

## Commands

```bash
# 查看 env 的 torch + cuda 版本
conda activate fastwam && python -c "import torch; print(torch.__version__, torch.version.cuda)"
# fastwam: 2.7.1+cu128, CUDA 12.8

conda activate vit-probe && python -c "import torch; print(torch.__version__, torch.version.cuda)"
# vit-probe: 2.9.1+cu126, CUDA 12.6
```

## Notes
- Date: 2026-04-29
- Environment: xdlab23, 4 conda envs (vit-probe, fastwam, traindit, base)
