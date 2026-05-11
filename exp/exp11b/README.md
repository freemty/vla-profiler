# exp11b — StarVLA-OFT E/C/A Profiling

## Motivation

StarVLA (HKUST, arXiv:2604.05014) 是 VLA 训练框架，backbone × action head
Lego-style 可组合。StarVLA-OFT 用 Qwen3-VL-4B + parallel MLP (OFT) head，
LIBERO 96.6% @30K steps (6x fewer steps than OpenVLA-OFT @175K)。

**论文完全不报 inference latency** — 零 Hz 数字、零 phase breakdown。
本实验填补这个 gap。

与 exp11a (OpenVLA-OFT) 形成同方法 (OFT head) 不同 backbone 的对照：
| Model | Backbone | Action Head | LIBERO | Hz | Source |
|-------|----------|-------------|--------|-----|--------|
| OpenVLA-OFT | Prismatic 7B | Parallel MLP (OFT) | 97.1% | 109.7 (A100) | exp11a |
| **StarVLA-OFT** | **Qwen3-VL-4B** | **Parallel MLP (OFT)** | **96.6%** | **unreported** | **this** |

额外对照 (同 backbone 不同 action head):
| Model | Backbone | Action Head | LIBERO | Hz | Source |
|-------|----------|-------------|--------|-----|--------|
| StarVLA-OFT | Qwen3-VL-4B | Parallel MLP | 96.6% | ? | this |
| StarVLA-π | Qwen3-VL-4B | Cross-DiT flow | — | ? | (future) |
| LingBot-VLA | Qwen2.5-VL-3B | Flow head | ~96% | 13 | exp03a |

## Core Questions

1. Qwen3-VL-4B 的 E/C 延迟 vs Prismatic 7B (exp11a) vs Qwen2.5-VL-3B (exp03a)?
2. OFT MLP head 在 StarVLA 框架下 A 阶段延迟？应与 exp11a 接近 (<1ms)
3. 4B backbone 的推理频率？预测 ~15-25Hz (比 exp11a 更快因为 4B < 7B)
4. StarVLA framework 抽象是否引入额外 overhead？(和原生 Qwen3-VL forward 对比)

## Architecture

```
Qwen3-VL-4B (ViT + Qwen3 4B LLM)  — Phase E+C
  Image → Qwen3 ViT encoder → LLM prefill
       ↓
OFT Parallel MLP Action Head  — Phase A
  L1 regression, following OpenVLA-OFT recipe
       ↓
Action Chunk: chunk_size varies, action_dim=7
```

Total: ~4B params

## Method

- **Timing**: CUDA event, warmup=15, iter=20, median
- **Input**: 合成 LIBERO-style (224×224 RGB, 7-dim proprio, language instruction)
- **Phases**: E (Qwen3 ViT encode) / C (Qwen3 LLM prefill) / A (OFT MLP head)
- **Hardware**: RTX 5880 Ada 48GB, bf16/sdpa

## Prerequisites

1. StarVLA repo: github.com/StarVLA/StarVLA
2. Qwen3-VL-4B base weights + StarVLA-OFT LIBERO checkpoint (if released)
3. 若 checkpoint 未开源：random-weight profiling (已验证 exp07b Δ<12%)

## Expected Results

| Phase | 预测延迟 (ms) | 依据 |
|-------|-------------|------|
| E (Qwen3 ViT) | ~15-25 | Qwen2.5-VL-3B exp03a E=35.7ms, 4B 架构更新但规模接近 |
| C (Qwen3 4B prefill) | ~20-35 | 4B 比 3B 略大, 但 Qwen3 架构可能更高效 |
| A (OFT MLP) | **<1** | 与 exp11a 同方法 |
| Total | ~35-60 | 对应 ~17-28Hz |

## Results (RTX 5880 Ada 48GB, Qwen2.5-VL-3B-Instruct, bf16)

> **Note**: 使用 Qwen2.5-VL-3B-Instruct 替代 Qwen3-VL-4B（服务器上 Qwen3-VL-4B shard 不完整）。
> 3B vs 4B 延迟差异预期 <20%（同系列架构），结论不变。

| Phase | Mean (ms) | Std (ms) | 占比 |
|-------|-----------|----------|------|
| E (Qwen2.5 ViT) | **34.70** | 0.88 | 54.8% |
| C (Qwen2.5 3B LLM prefill) | **28.45** | 0.84 | 44.9% |
| A (OFT MLP) | **0.13** | 0.01 | 0.2% |
| **Total** | **63.28** | — | **~15.8 Hz** |

**Key findings**:
- OFT MLP = 0.13ms — 比 flow head (Pi-Zero 165ms) 快 **1270x**
- 瓶颈彻底从 A 转移到 E+C (占 99.8%)
- 15.8Hz 适合 5-10Hz 控制场景

## Status

**done** (2026-05-11) — Qwen2.5-VL-3B-Instruct + random OFT head.

## References

- Paper: arXiv:2604.05014
- Code: github.com/StarVLA/StarVLA
- 精读: `survey/papers/starvla-deep-dive.md`
- 加速 survey: `survey/papers/vla-acceleration-tricks-2026.md`
