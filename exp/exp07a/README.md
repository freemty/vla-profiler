# exp07a — Pi-Zero E/C/A Profiling

## Motivation

补全 Flow VLA Pareto 前沿的关键数据点。Pi-Zero 是 Physical Intelligence 发布的
dual-stream flow VLA：PaliGemma (SigLIP + Gemma 2B) 做 context prefix，
Gemma 300M Action Expert 做 flow denoising。

现有 Flow VLA 对比：
| Model | Backbone | Action Head | Total | Hz | Source |
|-------|----------|-------------|-------|-----|--------|
| LingBot-VLA-4B | Qwen2.5-VL-3B | 轻量 flow (0.48ms) | 74.5ms | 13 | exp03a |
| **Pi-Zero** | **PaliGemma (SigLIP + Gemma 2B)** | **Gemma 300M Expert** | **?** | **?** | **this** |

Pi-Zero 独特之处：
- **双 Gemma 并行**：VLM stream (2B) + Action Expert stream (300M)，共享 KV
- **Action Expert 独立 Transformer**（不是小 MLP head），每个 denoise step 运行完整 300M
- 预测：Action phase 应显著高于 LingBot-VLA (0.48ms)，可能 30-80ms (10 steps × 300M)

## Core Questions

1. Pi-Zero E/C/A breakdown (SigLIP / PaliGemma prefill / Action Expert)?
2. Per-step Action Expert cost at 300M params?
3. Gemma 2B context prefill vs Qwen2.5-VL-3B context: 哪个更重?
4. 300M Action Expert 落在 DiT scaling curve 哪个位置? (ref: 174M=7.2ms, 350M=32ms)
5. Total latency → 实时可行性?

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

## Predictions

| Phase | Prediction | Reasoning |
|-------|-----------|-----------|
| E (SigLIP) | ~15-25ms | SigLIP So400m/14 类似 NitroGen SigLIP (8.7ms) 但可能更大 |
| C (PaliGemma prefill) | ~40-80ms | 2B Gemma, 276 tokens prefill. Ref: 3B Qwen = 38ms (exp03a) |
| A (Action Expert ×10) | ~30-80ms | 300M × 10 steps. Ref: 174M=7.2ms/step, 350M=32ms/step. 300M 可能 15-25ms/step |
| Total | ~85-185ms | 5-12 Hz — 比 LingBot-VLA 慢但可能接近实时 |

## Config

```yaml
configs/pizero/profiling.yaml
  controller_name: pizero
  denoise_steps: 10
  inputs: single_camera_224 (3×224×224)
  num_warmup_runs: 3
  num_benchmark_runs: 10
```

## Run

```bash
source /data1/ybyang/vlla/.venvs/pizero/bin/activate
export HF_HOME=/data1/ybyang/huggingface
cd /data1/ybyang/vlla
python src/run_tasks.py +experiment=pizero/profiling device=cuda:0
```

## Status

- [ ] uv env setup on xdlab23
- [ ] random-weight forward pass verification
- [ ] E/C/A profiling run
- [ ] Results analysis
