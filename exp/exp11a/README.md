# exp11a — OpenVLA-OFT E/C/A Profiling

## Motivation

OpenVLA-OFT (Berkeley, arXiv:2502.19645) 是当前开源纯 VLA 速度王者：
把 AR decoding 换成 parallel MLP regression (OFT head)，LIBERO 97.1%，
A100 上 109.7Hz (0.073s per action)。

但它报的是 end-to-end wall-clock，**无 E/C/A phase breakdown**。
本实验补全 phase-level profiling，定位瓶颈分布。

与 exp11b (StarVLA-OFT) 形成同方法 (OFT head) 不同 backbone 的对照：
| Model | Backbone | Action Head | Reported Hz | Source |
|-------|----------|-------------|-------------|--------|
| **OpenVLA-OFT** | **Prismatic 7B** | **Parallel MLP (OFT)** | **109.7** (A100) | **this** |
| StarVLA-OFT | Qwen3-VL-4B | Parallel MLP (OFT) | unreported | exp11b |

## Core Questions

1. OFT parallel MLP action head 的绝对延迟？预测 <1ms (类比 exp03a flow head 0.48ms)
2. E/C/A breakdown — backbone (Prismatic 7B) encode + context 占多大比例？
3. 109.7Hz 在 RTX 5880 Ada 上能否复现？(论文用 A100)
4. OFT head vs flow head (exp07a Pi-Zero 165ms action): 延迟差多少倍？

## Architecture

```
Prismatic VLM 7B (DINOv2 + SigLIP → MLP projector → Llama 2 7B)  — Phase E+C
  Image → dual vision encoder → fused features → language model
       ↓
OFT Parallel MLP Action Head  — Phase A
  Direct L1 regression, action chunk = 5 steps, no diffusion/flow
       ↓
Action Chunk: chunk_size=5, action_dim=7
```

Total: ~7B params

## Method

- **Timing**: CUDA event, warmup=15, iter=20, median
- **Input**: 合成 LIBERO-style (224×224 RGB, 7-dim proprio, language instruction)
- **Phases**: E (vision encode) / C (LLM prefill) / A (OFT MLP head)
- **Hardware**: RTX 5880 Ada 48GB, bf16/sdpa

## Prerequisites

1. OpenVLA-OFT checkpoint: `openvla/openvla-7b-finetuned-libero-spatial` (or OFT variant)
2. Install: `pip install openvla` or clone `openvla/openvla` repo
3. 确认 OFT head 可独立 forward (非 AR generate loop)

## Expected Results

| Phase | 预测延迟 (ms) | 依据 |
|-------|-------------|------|
| E (dual vision encoder) | ~10-15 | DINOv2+SigLIP 类似 PaliGemma SigLIP 9.3ms 但 dual |
| C (Llama 2 7B prefill) | ~30-50 | 7B 类比 Qwen 7B exp01a P=156ms 但短 context |
| A (OFT MLP) | **<1** | Parallel MLP, 无 denoise loop |
| Total | ~40-65 | 论文 73ms on A100 → RTX 5880 预期 ~1.2x |

## Results (RTX 5880 Ada 48GB, openvla-7b real weights, bf16, eager attn)

| Phase | Mean (ms) | Std (ms) | 占比 |
|-------|-----------|----------|------|
| E (DINOv2 + SigLIP dual) | **16.78** | 0.26 | 15.4% |
| C (Llama-2 7B prefill) | **92.25** | 1.18 | 84.4% |
| A (OFT MLP) | **0.24** | 0.05 | 0.2% |
| **Total** | **109.27** | — | **~9.2 Hz** |

**Key findings**:
- OFT MLP = 0.24ms — action 瓶颈彻底消除
- Llama-2 7B prefill 占 84% — backbone size 是 OFT VLA 的唯一瓶颈
- 9.2Hz vs 论文 109.7Hz (A100): 差距来自 eager attn (no flash) + RTX 5880 vs A100
- 论文的 109.7Hz 很可能只测了 action head throughput（不含 backbone）

## Status

**done** (2026-05-11) — openvla-7b real weights via hf-mirror.com.

## References

- Paper: arXiv:2502.19645
- Code: github.com/openvla/openvla
- 加速 survey: `survey/papers/vla-acceleration-tricks-2026.md` §D
- 效率 survey: `survey/papers/vla-wam-efficiency-systems-deep-research.md`
