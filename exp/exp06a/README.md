# exp06a — NitroGen 500M DiT Profiling

## Motivation

填补 VA 模型 per-step latency vs DiT size scaling curve 的关键空白。

现有数据点:
| Model | DiT params | Per-step cost | Source |
|-------|-----------|---------------|--------|
| LingBot-VLA flow head | ~轻量 (嵌入式) | 0.048ms | exp03a |
| **NitroGen DiT** | **~100M** | **待测 (估 5-10ms)** | **this** |
| Fast-WAM ActionDiT | ~350M | 32ms | exp04a |
| LingBot-VA DiT | ~5B | 28.5ms | exp04b |

## Core Questions

1. NitroGen 的 E/C/A breakdown (SigLIP / VL-SA / DiT)?
2. Per-step DiT cost at ~100M params?
3. k sweep (1/2/4/8/16) 的 latency-quality trade-off?
4. ~100M DiT 是 compute-bound 还是 memory-bandwidth-bound?
5. VL self-attention context phase 开销多大?

## Architecture

```
SigLIP 2 ViT (~400M)         — Phase E
  256x256 → 256 image tokens
       ↓
VL Self-Attention (4L, ~30M)  — Phase C
  vision token mixing
       ↓
DiT Action Head (~100M)       — Phase A
  12L, cross-attn to 256 tokens
  k=16 flow matching steps (Euler)
       ↓
Action Chunk: 16 steps × 20-dim
```

Total: ~500M params

## Baseline Predictions

| Phase | Prediction | Rationale |
|-------|-----------|-----------|
| Encode (SigLIP) | 10-25ms | 256x256 input, no dynamic tiling |
| Context (VL-SA) | 3-8ms | 4-layer self-attn on 256 tokens |
| Action (DiT×16) | 80-160ms | 100M DiT ≈ 5-10ms/step × 16 |
| **Total** | **93-193ms** | **5-10 Hz** |

## Method

NitroGenController with manual timer marks for E/C/A phases + per-step CUDA events.

Config: `configs/nitrogen/profiling.yaml`
- Random weights mode (valid for timing)
- 5 input variants: k=1,2,4,8,16

## Hardware

- GPU: RTX 5880 Ada 48GB (xdlab23)
- Model: NitroGen 500M (random weights, timing-only)

## Prerequisites

```bash
# On xdlab23:
cd /data1/ybyang
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen && pip install -e .
```

## Commands

```bash
bash scripts/sync_to_remote.sh
bash scripts/launch_exp.sh 0 nitrogen/profiling
bash scripts/download-results.sh
```

## Status

**pending** — controller written, needs deploy + run
