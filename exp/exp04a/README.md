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

- **Backbone:** ActionDiT (interpolated from Wan2.2 DiT), hidden_dim=1024
- **Encoder:** SigLIP vision encoder (image → latent tokens)
- **Text conditioning:** T5 embeddings (precomputed)
- **Action generation:** Flow matching (Euler solver), multiple steps
- **No video denoising at test time** — key difference from full WAM
- **Total params:** estimated ~1-2B (ActionDiT 1024-dim, 24-30 layers)

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

## Expected Phases (E/C/A mapping)

```
Fast-WAM inference pipeline:
  E: SigLIP encode (per-camera image → visual tokens)
  C: ActionDiT context build (visual tokens + T5 text + action history → KV cache)
  A: Flow matching action generation (Euler solver, N steps)
```

## Predictions (record before running)

| Phase | Predicted | Reasoning |
|-------|-----------|-----------|
| E (SigLIP) | ~15-25ms | SigLIP-384 is ~400M params, smaller than Qwen2.5-VL ViT |
| C (ActionDiT context) | ~100-130ms | 1024-dim DiT, ~24 layers, single forward pass |
| A (flow matching) | ~30-50ms | Multiple Euler steps but action-only (no video) |
| Total | ~150-200ms | Should match paper's 190ms claim |

## Key Findings (fill after experiment)

TBD

## Pitfalls

- Fast-WAM evaluation defaults to 8 GPUs — we need single-GPU profiling mode
- Wan2.2 DiT checkpoint download may be large (~10GB+)
- LIBERO simulation environment not needed for pure latency profiling (can use dummy inputs)
- mujoco 3.3.2 version lock

## References

- Paper: https://arxiv.org/abs/2603.16666
- Code: https://github.com/yuantianyuan01/FastWAM
- Checkpoints: https://huggingface.co/yuanty/fastwam
- Project page: https://yuantianyuan01.github.io/FastWAM/
- Comparison: Fast-WAM vs LingBot-VA (same benchmark, different WAM approach)
