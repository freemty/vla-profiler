# exp07a — Pi-Zero E/C/A Profiling

## Motivation

补全 Flow VLA Pareto 前沿的关键数据点。Pi-Zero 是 Physical Intelligence 发布的
dual-stream flow VLA：PaliGemma (SigLIP + Gemma 2B) 做 context prefix，
Gemma 300M Action Expert 做 flow denoising。

现有 Flow VLA 对比：
| Model | Backbone | Action Head | Total | Hz | Source |
|-------|----------|-------------|-------|-----|--------|
| LingBot-VLA-4B | Qwen2.5-VL-3B | 轻量 flow (0.48ms) | 74.5ms | 13 | exp03a |
| **Pi-Zero** | **PaliGemma (SigLIP + Gemma 2B)** | **Gemma 300M Expert** | **~200ms** (stable) | **~5** | **this** |

Pi-Zero 独特之处：
- **双 Gemma 并行**：VLM stream (2B) + Action Expert stream (300M)，共享 KV
- **Action Expert 独立 Transformer**（不是小 MLP head），每个 denoise step 运行完整 300M
- Action phase 占总延迟 82-94%，验证了 heavy action head 的代价

## Core Questions

1. Pi-Zero E/C/A breakdown (SigLIP / PaliGemma prefill / Action Expert)? **ANSWERED**
2. Per-step Action Expert cost at 300M params? **~16-21ms/step**
3. Gemma 2B context prefill vs Qwen2.5-VL-3B context: 哪个更重? **Gemma 2B (26-33ms) < Qwen 3B (38ms)**
4. 300M Action Expert 落在 DiT scaling curve 哪个位置? **Between 174M (7.2ms) and 350M (32ms)**
5. Total latency → 实时可行性? **~200ms (stable) = ~5Hz, marginal for real-time**

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

### E/C/A Breakdown (RTX 5880 Ada 48GB, bf16, random weights)

> **Canonical baseline = stable-window (runs 13-20)**. 原始 20-run mean 被 GPU 功率爬坡污染（前 12 次慢 ~1.25x），aggregated mean/median 均不可直接使用，详见 [Bimodal Distribution](#bimodal-distribution) 一节。

| Phase | Stable mean (ms) | Stable median (ms) | Stable std (ms) | Stable CV | % Total |
|-------|------------------|--------------------|-----------------|-----------|---------|
| E (SigLIP) | **9.32** | 9.28 | ~0.09 | <1% | 4.6% |
| C (Gemma 2B prefill) | **26.40** | 26.23 | ~0.34 | 1.3% | 13.2% |
| A (Expert ×10 steps) | **164.76** | 164.75 | ~1.98 | 1.2% | 82.2% |
| **Total** | **~200.5** | ~200.3 | — | — | **100%** |

**Polluted aggregated mean (runs 1-20, 仅参考)**: E=10.83 / C=30.26 / A=189.07 / Total=230.2 ms — 不作为 canonical。

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

所有 phase 呈双峰分布（runs 1-12 高，runs 13-20 低），ratio 约 1.25-1.27x：

| Phase | Unstable (runs 1-12) | Stable (runs 13-20) | Ratio |
|-------|---------------------|---------------------|-------|
| E | 11.84 ms | 9.32 ms | 1.27x |
| C | 32.83 ms | 26.40 ms | 1.24x |
| A | 205.28 ms | 164.76 ms | 1.25x |

**归因**：GPU 功率状态 warmup (5 次 warmup 不够)。跳变发生在 run 12→13，之后稳定态 CV < 2%。这是真实物理现象（SM clock 从节能态切换到高性能态），不是 timing code bug。

**对后续实验的影响**：所有 profiling 实验应改用 **warmup=15 + `nvidia-smi -pm 1`** 锁定 persistence mode。现有 exp01-06 数据在 CV <5% 的场景下仍可信，但 exp04b (CV 21%) 和 exp07a (bimodal) 应以稳定态或重跑数据为准。

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
