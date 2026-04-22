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

## Results

Model: 407M total (Vision=199M, VL-SA=28M, DiT=**174M**)

| k (steps) | Encode (ms) | Context (ms) | Action (ms) | Total (ms) | Per-step (ms) | Hz |
|-----------|-------------|-------------|-------------|------------|---------------|------|
| **16** | 8.70 | 1.95 | **115.39** | **126.04** | **7.21** | 7.9 |
| **8** | 8.98 | 1.95 | **57.40** | **68.33** | **7.18** | 14.6 |
| **4** | 8.67 | 1.93 | **28.80** | **39.40** | **7.20** | 25.4 |
| **2** | 9.06 | 1.94 | **14.45** | **25.45** | **7.22** | 39.2 |
| **1** | 8.71 | 1.93 | **7.30** | **17.94** | **7.30** | 55.9 |

### Key Findings

1. **Per-step DiT cost = 7.2ms** — perfectly linear scaling across all k values
2. **Action dominates 91.6%** at k=16 (consistent with Fast-WAM/LingBot-VA pattern)
3. **SigLIP encode = 8.7ms** — stable, low variance (256x256 fixed input, 256 tokens)
4. **VL self-attention = 1.9ms** — negligible overhead (4-layer, 28M params)
5. **k=1 achieves 55.9Hz** — real-time capable with step distillation

### DiT Per-Step Scaling Curve

| Model | DiT Params | Per-step Cost | Regime |
|-------|-----------|---------------|--------|
| LingBot-VLA flow head | ~轻量 (embedded) | 0.048ms | Compute-bound |
| **NitroGen DiT** | **174M** | **7.2ms** | **Compute-bound** |
| LingBot-VA DiT | ~5B (shared) | 28.5ms | Memory-BW-bound |
| Fast-WAM ActionDiT | ~350M | 32ms | Memory-BW-bound |

174M→350M: 2x params, 4.4x latency → super-linear → **transition from compute-bound to memory-BW-bound between 174M-350M**.

## Status

**done** — 2026-04-22
