# exp04a: Fast-WAM E/A Profiling

## Motivation

Profile Fast-WAM (arXiv:2603.16666) — a WAM that **skips test-time video imagination** — to understand the latency breakdown of the "skip-imagination" paradigm. This fills the WAM/VA gap in our profiling spectrum.

**Key question:** Fast-WAM claims 190ms total, 4x faster than full WAMs. Where does the time go? Is it E-dominated (like LingBot-VLA) or A-dominated (like AR VLA)?

## Comparison Context

| Model | Paradigm | Total | E | C/V | A | Hz |
|-------|----------|-------|---|-----|---|-----|
| ACT (exp02a) | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 |
| LingBot-VLA (exp03a) | Flow VLA | 74.5ms | 35.7ms | 38.3ms | 0.48ms | 13 |
| **Fast-WAM (this)** | WAM (no imagination) | 190ms | ? | ? | ? | 5.3 |
| Qwen2.5-VL (exp01a) | VLM (AR) | 427ms | 253ms | 156ms | 18.6ms/tok | 2.3 |

## Architecture (from paper + repo)

- **Video Expert:** WanVideoDiT (Wan2.2-TI2V-5B), hidden_dim=3072, 30 layers, 24 heads
- **Action Expert:** ActionDiT, hidden_dim=1024, 30 layers, 24 heads
- **MoT (Mixture of Transformers):** Routes tokens between video/action experts with cross-attention
- **VAE:** WanVideoVAE38 (z_dim=48) — encodes single frame to latent
- **Text conditioning:** T5/UMT5-XXL (4096-dim), precomputed at inference
- **Proprio encoder:** Linear(7→4096) for proprioception
- **Action generation:** Flow matching (Euler solver), 10 steps default
- **No video denoising at test time** — key difference from full WAM
- **Total params: 6725M (6.7B)** — much larger than expected due to full video expert

## Setup

### Prerequisites

```bash
# Clone repo
git clone https://github.com/yuantianyuan01/FastWAM.git /data1/ybyang/FastWAM
cd /data1/ybyang/FastWAM

# Create env
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Prepare backbone
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# Download checkpoint
pip install -U huggingface_hub
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

### Profiling Approach

Unlike exp01-03 where we wrote custom controllers, Fast-WAM's inference pipeline is self-contained. Strategy:

1. **Instrument the inference loop** — add PhaseTimer around:
   - Image encoding (SigLIP forward)
   - T5 text embedding lookup
   - ActionDiT flow matching (per-step timing)
   - Total per-chunk time
2. **Measure with warmup** — 5 warmup, 20 measured iterations
3. **Vary conditions:**
   - Single cam vs 2-cam (224px)
   - Different action chunk lengths (if configurable)
   - Compare Euler steps (default vs reduced)

## Actual Phases (E/C/A mapping)

```
Fast-WAM inference pipeline:
  E: VAE encode (single-frame image → 48-dim latent)
  C: Video expert prefill + MoT KV cache (one-time per observation)
  A: Flow matching action denoising (ActionDiT through MoT, N Euler steps)
      Each step: ActionDiT pre_dit → MoT forward_action_with_video_cache (30 layers cross-attn)
```

## Predictions vs Actuals

| Phase | Predicted | Actual (20-step) | Actual (10-step) | Notes |
|-------|-----------|-----------------|-----------------|-------|
| E (VAE) | ~15-25ms | **7.0ms** | **7.6ms** | Much faster: VAE not SigLIP |
| C (Video prefill + KV cache) | ~100-130ms | **32.4ms** | **36.7ms** | Single-frame only, not full video |
| A (flow matching) | ~30-50ms | **637.9ms** | **362.4ms** | Way higher: 30-layer MoT cross-attn × N steps |
| Total | ~150-200ms | **677ms** | **407ms** | 3.5x over paper's 190ms |

## Key Findings

### Primary Results (RTX 5880 Ada, bf16, random init, 20 iterations)

| Config | Encode | Context | Action | Total | Hz |
|--------|--------|---------|--------|-------|-----|
| 20 steps | 7.0ms | 32.4ms | 637.9ms | 677ms | 1.5 |
| 10 steps (paper default) | 7.6ms | 36.7ms | 362.4ms | 407ms | 2.5 |
| 5 steps | 7.1ms | 33.8ms | 164.0ms | 205ms | 4.9 |

### Key Insights

1. **Action phase dominates (89-94%)** — completely A-dominated, opposite to LingBot-VLA (which is C≈E≈50%)
2. **Per-step cost: ~32ms/step** — each Euler step runs full 30-layer ActionDiT through MoT with cross-attention to video KV cache
3. **Paper's 190ms likely measured on A100/H100** — not RTX 5880 Ada. Or with 5 steps.
4. **Encode is trivial (7ms)** — single-frame VAE encode, much lighter than ViT-based encoders
5. **Context is cheap (32ms)** — one-time video expert prefill, amortized over all denoise steps
6. **Architecture surprise: 6.7B params** — video expert (5B Wan2.2 DiT) + action expert (350M ActionDiT). "Skip-imagination" = reuse video expert's learned representations without running video denoising, but still loads full video DiT weights.
7. **Comparison to LingBot-VLA (74.5ms):** Fast-WAM is 5-9x slower. The 30-layer cross-attention per denoise step is expensive. LingBot-VLA's flow matching (0.48ms, 10 steps) achieves similar effect with a much smaller action head.
8. **Prediction calibration:** Massively underestimated Action (predicted 30-50ms, got 362-638ms). Root cause: didn't realize action denoise uses full MoT (30 layers × 24 heads × cross-attn to video cache) per step. Encode/Context were overestimated.

### Comparison Table (updated)

| Model | Paradigm | Total | E | C | A | Hz | Params |
|-------|----------|-------|---|---|---|-----|--------|
| ACT (exp02a) | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 | ~50M |
| LingBot-VLA (exp03a) | Flow VLA | 74.5ms | 35.7ms | 38.3ms | 0.48ms | 13 | 4B |
| **Fast-WAM @5step** | WAM (skip-imag) | **205ms** | 7.1ms | 33.8ms | 164ms | **4.9** | 6.7B |
| **Fast-WAM @10step** | WAM (skip-imag) | **407ms** | 7.6ms | 36.7ms | 362ms | **2.5** | 6.7B |
| Qwen2.5-VL (exp01a) | VLM (AR) | 427ms | 253ms | 156ms | 18.6ms/tok | 2.3 | 7B |

## Pitfalls

- Fast-WAM evaluation defaults to 8 GPUs — we need single-GPU profiling mode
- Wan2.2 DiT checkpoint download is ~12GB (non-Diffusers format needed)
- HuggingFace download requires `HF_ENDPOINT=https://hf-mirror.com` on xdlab23
- LIBERO simulation environment not needed for pure latency profiling (dummy inputs valid)
- **Random init gives same timing as real weights** (timing depends on compute graph, not weight values)
- MoT constructor takes `mixtures={"video": ..., "action": ...}` dict, not keyword args
- VAE encode: use `model._encode_input_image_latents_tensor()`, not `vae.encode()` directly
- `configs/train.yaml` sets `eval_num_inference_steps: 10` (default, not the 20 in our initial test)
- High variance on RTX 5880 Ada with 10 steps (power management?); 20 steps shows stable std=2ms

## References

- Paper: https://arxiv.org/abs/2603.16666
- Code: https://github.com/yuantianyuan01/FastWAM
- Checkpoints: https://huggingface.co/yuanty/fastwam
- Project page: https://yuantianyuan01.github.io/FastWAM/
- Comparison: Fast-WAM vs LingBot-VA (same benchmark, different WAM approach)
