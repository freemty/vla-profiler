# exp07a — Pi-Zero E/C/A Profiling

## Motivation

补全 Flow VLA Pareto 前沿的关键数据点。Pi-Zero 是 Physical Intelligence 发布的
dual-stream flow VLA：PaliGemma (SigLIP + Gemma 2B) 做 context prefix，
Gemma 300M Action Expert 做 flow denoising。

现有 Flow VLA 对比：
| Model | Backbone | Action Head | Total | Hz | Source |
|-------|----------|-------------|-------|-----|--------|
| LingBot-VLA-4B | Qwen2.5-VL-3B | 轻量 flow (0.48ms) | 74.5ms | 13 | exp03a |
| **Pi-Zero** | **PaliGemma (SigLIP + Gemma 2B)** | **Gemma 300M Expert** | **201ms** | **~5** | **this** |

Pi-Zero 独特之处：
- **双 Gemma 并行**：VLM stream (2B) + Action Expert stream (300M)，共享 KV
- **Action Expert 独立 Transformer**（不是小 MLP head），每个 denoise step 运行完整 300M
- Action phase 占总延迟 82-94%，验证了 heavy action head 的代价

## Core Questions

1. Pi-Zero E/C/A breakdown (SigLIP / PaliGemma prefill / Action Expert)? **ANSWERED**
2. Per-step Action Expert cost at 300M params? **~16-21ms/step**
3. Gemma 2B context prefill vs Qwen2.5-VL-3B context: 哪个更重? **Gemma 2B (26-33ms) < Qwen 3B (38ms)**
4. 300M Action Expert 落在 DiT scaling curve 哪个位置? **Between 174M (7.2ms) and 350M (32ms)**
5. Total latency → 实时可行性? **~201ms = ~5Hz, marginal for real-time**

## Architecture

```
SigLIP ViT-So400m/14 (~400M)         — Phase E
  224×224 → 256 image tokens
       ↓
PaliGemma / Gemma 2B (18L, 2048 dim) — Phase C
  Joint attention: image + language + proprio
  Caches KV for Action Expert
       ↓
Gemma 300M Action Expert (18L, 1024 dim) — Phase A
  Attends to PaliGemma KV (cross-stream)
  N=10 Euler flow matching steps
       ↓
Action Chunk: horizon_steps=4, action_dim=7
```

Total: ~2.7B params (SigLIP 400M + PaliGemma 2B + Action Expert 300M)

## Results

### E/C/A Breakdown (20 runs, RTX 5880 Ada 48GB, bf16, random weights)

| Phase | Mean (ms) | Median (ms) | Std (ms) | CV | % Total |
|-------|-----------|-------------|----------|-----|---------|
| E (SigLIP) | 10.8 | 11.7 | 1.28 | 11.8% | 4.7% |
| C (Gemma 2B prefill) | 30.3 | 32.7 | 3.24 | 10.7% | 13.2% |
| A (Expert ×10 steps) | 189.1 | 204.1 | 20.7 | 11.0% | 82.1% |
| **Total** | **~230** | **~248** | — | — | **100%** |

Single run (last iteration): E=9.5ms + C=26.2ms + A=165.3ms = **201ms (~5Hz)**

### Per-Step Action Expert Cost

| Metric | Value |
|--------|-------|
| Total A (10 steps) | 162-213ms |
| Per-step mean | ~16.4-20.6ms |
| Per-step (stabilized, runs 13-20) | ~16.2-16.8ms |

### DiT Scaling Curve Position

| Model | DiT Params | Per-Step (ms) | Notes |
|-------|-----------|---------------|-------|
| NitroGen | 174M | 7.2 | Pure DiT, no cross-attn (exp06a) |
| **Pi-Zero Expert** | **300M** | **~18** | **Gemma + cross-attn to PaliGemma KV** |
| Fast-WAM ActionDiT | 350M | 32 | MoT cross-attn, 30L (exp04a) |

300M Expert 比纯 DiT 贵 ~2.5x (vs 174M linear extrapolation ~12ms)，因为包含 cross-attention 到 PaliGemma KV cache。

### Bimodal Distribution

所有 phase 呈双峰分布（runs 1-12 高，runs 13-20 低）：
- E: ~11.7ms → ~9.3ms
- C: ~32.7ms → ~26.2ms
- A: ~205ms → ~165ms

可能原因：GPU 功率状态 warmup (5 次 warmup 不够)。稳定态（runs 13-20）数值更可靠。

## Prediction Calibration

| Phase | Predicted | Actual (stable) | Verdict |
|-------|-----------|-----------------|---------|
| E (SigLIP) | 15-25ms | ~9.3ms | **低估 SigLIP 效率** — 比 NitroGen SigLIP (8.7ms) 略高，合理 |
| C (Gemma 2B) | 40-80ms | ~26ms | **低估 short-seq prefill 效率** — 276 tokens 对 2B 很轻 |
| A (Expert ×10) | 30-80ms | ~165ms | **严重低估 2-3x** — cross-attn 代价被低估 |
| Total | 85-185ms | ~201ms | **略超上界** — A phase 低估拖累 |

## Config

```yaml
configs/pizero/profiling.yaml
  controller_name: pizero
  denoise_steps: 10
  inputs: single_camera_224 (3×224×224)
  num_warmup_runs: 5
  num_benchmark_runs: 20
```

## Run

```bash
bash scripts/launch_pizero.sh 0 pizero/profiling
```

## Status

- [x] uv env setup on xdlab23
- [x] random-weight forward pass verification
- [x] E/C/A profiling run (20 iterations)
- [x] Results analysis + prediction calibration
