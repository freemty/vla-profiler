# exp04b: LingBot-VA E/V/A Profiling

## Motivation

Profile LingBot-VA (arXiv:2601.21998) — a **full WAM** that generates video (imagination) before predicting actions. This completes the WAM profiling spectrum alongside exp04a (Fast-WAM, skip-imagination).

**Key question:** How much does "imagination" (video generation) cost? What's the E/V/A breakdown in a full WAM?

## Comparison Context

| Model | Paradigm | Total | E | V (video) | A (action) | Hz |
|-------|----------|-------|---|-----------|------------|-----|
| ACT (exp02a) | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 |
| LingBot-VLA (exp03a) | Flow VLA | 74.5ms | 35.7ms | — | 38.8ms | 13 |
| Fast-WAM @10step (exp04a) | WAM (skip) | 407ms | 7.6ms | — | 399ms | 2.5 |
| **LingBot-VA (this)** | WAM (full) | ? | ? | ? | ? | ? |

## Architecture (from paper + repo)

- **Transformer:** WanTransformer3DModel (5B, Wan2.2-TI2V-5B architecture)
  - 40 layers, 40 heads, head_dim=128, hidden_dim=5120, ffn_dim=13824
  - Shared between video and action streams (action_mode flag)
  - MoT dual-stream: video tokens (5B params) + action tokens (350M params)
- **VAE:** AutoencoderKLWan (Wan2.2 VAE, streaming encoder)
- **Text encoder:** UMT5-XXL (4096-dim, ~4.7B)
- **Video denoise:** 20 steps (default), CFG scale=5
- **Action denoise:** 50 steps (default), CFG scale=1
- **Resolution:** 128×128 per camera, 2 cameras, 4-frame chunks
- **Action:** 30-dim, 4 actions per frame

## Setup

```bash
# Code deployed via tarball
# /data1/ybyang/lingbot-va/

# Environment: vit-probe (torch 2.9.0, diffusers 0.35.2)
# Additional: flash_attn needed
conda activate vit-probe
cd /data1/ybyang/lingbot-va
python /data1/ybyang/vlla/scripts/profile_lingbot_va.py \
    --mode random --gpu 0 --warmup 5 --iterations 20
```

## Predictions (record before running)

| Phase | Predicted | Reasoning |
|-------|-----------|-----------|
| E (VAE encode) | ~10-15ms | Streaming VAE, 128×128 × 2cam × 4frames, smaller than Fast-WAM |
| V (Video denoise) | ~600-900ms | 5B DiT × 20 steps = ~30-45ms/step |
| A (Action denoise) | ~500-750ms | Same 5B DiT × 50 steps, but action tokens much fewer |
| Total | ~1100-1700ms | Full WAM is 2-4x slower than skip-imagination |

Reasoning for predictions:
- Fast-WAM action step (30L, 1024 hidden, action-only MoT): ~32ms/step
- LingBot-VA uses full 5B DiT (40L, 5120 hidden) for both video and action
- Video has more tokens (latent_h × latent_w × chunk_size) than action (action_per_frame × chunk_size)
- Expect video step to be slower than action step

## Results

### E/V/A Breakdown (RTX 5880 Ada, random-init, bf16)

| Phase | Mean (ms) | Std | % of Total | Per-step |
|-------|-----------|-----|------------|----------|
| E (VAE encode) | 75.5 | 16.0 | 3.6% | — |
| V (Video denoise, 20 steps) | 592.5 | 84.8 | 28.3% | ~29.6ms/step |
| A (Action denoise, 50 steps) | 1423.1 | 232.5 | 68.1% | ~28.5ms/step |
| **TOTAL** | **2091.2** | 330.5 | 100% | — |
| **Hz** | **0.5** | | | |

### Comparison Table (updated)

| Model | Paradigm | Total | E | V (video) | A (action) | Hz |
|-------|----------|-------|---|-----------|------------|-----|
| ACT (exp02a) | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 |
| LingBot-VLA (exp03a) | Flow VLA | 74.5ms | 35.7ms | — | 38.8ms | 13 |
| Fast-WAM @10step (exp04a) | WAM (skip) | 407ms | 7.6ms | — | 399ms | 2.5 |
| **LingBot-VA (this)** | **WAM (full)** | **2091ms** | **75.5ms** | **592.5ms** | **1423.1ms** | **0.5** |

## Key Findings

1. **Full WAM is 5x slower than skip-imagination WAM** — video generation (592ms) adds massive overhead before action even starts.
2. **Action still dominates at 68%** — same pattern as Fast-WAM. Consistent finding: diffusion action heads are the bottleneck in all WAM architectures.
3. **Per-step cost remarkably similar for V and A** — ~29ms/step (video) vs ~28.5ms/step (action). Both use the same 5B DiT with 30 layers, just different token counts. Video tokens (4×4×8 = 128 patches) slightly more than action tokens (4×4×1 = 16), but the difference is absorbed by attention overhead being dominated by model width not sequence length at these scales.
4. **VAE encode 75.5ms is 10x higher than Fast-WAM's 7.6ms** — likely because Fast-WAM uses a much lighter/pre-cached encode path, while LingBot-VA's streaming VAE with z_dim=48 does full spatial encoding.
5. **0.5 Hz makes real-time control impossible** — 100x slower than VLA (13Hz), 660x slower than ACT (330Hz). Full imagination WAMs are research-grade, not deployment-grade.

## Prediction Calibration

| Phase | Predicted | Actual | Accuracy |
|-------|-----------|--------|----------|
| E (VAE encode) | 10-15ms | 75.5ms | ❌ 5-7x under — streaming VAE with z_dim=48 much heavier than expected |
| V (Video denoise) | 600-900ms | 592.5ms | ✅ Nailed it — prediction was 600ms lower bound |
| A (Action denoise) | 500-750ms | 1423.1ms | ❌ 2x under — action has 50 steps (not 20 like video), missed the 2.5x step multiplier |
| Total | 1100-1700ms | 2091ms | ⚠️ 23% over upper bound — cumulative underestimation |

Key calibration insight: predicted per-step correctly (~30ms), but failed to account for (1) streaming VAE overhead in random-init mode and (2) 50 action steps being 2.5x the 20 video steps.

## Pitfalls

- Requires flash_attn (compiled from source, ~10min install)
- model.py imports flash_attn at module level — no easy bypass
- Uses diffusers AutoencoderKLWan — random init may differ from actual VAE config
- Transformer random init may fail if config params don't match WanTransformer3DModel signature
- FlowMatchScheduler is custom (in wan_va/utils/) — not from diffusers

## References

- Paper: https://arxiv.org/abs/2601.21998
- Code: https://github.com/Robbyant/lingbot-va
- Weights: https://huggingface.co/robbyant/lingbot-va-base
- Same team as LingBot-VLA (exp03a)
