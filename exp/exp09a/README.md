# exp09a — Cosmos Policy Direct-Mode Latency Profiling

## Goal

Measure the inference latency of Cosmos Policy (LIBERO config, Cosmos-Predict2-2B)
in **direct mode** (no planning). This fills the key gap in arXiv:2601.16163 which
reports zero latency numbers for direct-mode inference.

## Phase Decomposition

| Phase | What | Expected |
|-------|------|----------|
| E (Encode) | Image preprocessing + VAE tokenization | ~10-50ms (estimate from LingBot-VA VAE) |
| D (Denoise) | EDM solver × 5 steps through 2B DiT | Main cost. Per-step estimate ~30-100ms |
| X (Extract) | Decode action latent → action chunk (16×14) | <1ms (linear projection) |

## Model Details

- **Backbone**: Cosmos-Predict2-2B (DiT, flow matching / EDM)
- **VAE**: Wan2.1 tokenizer (temporal compression 4x, spatial 8x)
- **Action encoding**: "Latent frame injection" — action chunk normalized to [-1,1], replicated to fill H'×W'×C' latent volume
- **Denoise steps**: 5 (paper default for LIBERO)
- **Chunk size**: 16 timesteps @ 25Hz = 0.64s
- **VRAM**: ~6.8 GB (paper claim)

## Comparison Targets

| Model | Total (ms) | Hz | Action % | Source |
|-------|-----------|-----|----------|--------|
| Pi-Zero (exp07a) | 200.5 | 5.0 | 82% | vlla canonical |
| Fast-WAM (exp04a) | 407 | 2.5 | 89% | vlla canonical |
| LingBot-VA (exp04b) | 2518 | 0.4 | 69% | vlla canonical |
| **Cosmos Policy** | **???** | **???** | **???** | **this experiment** |

## Run

```bash
# Inside cosmos-policy Docker on xdlab23:
python /data1/ybyang/vlla/scripts/exp09a_cosmos_policy_profiling.py \
    --gpu 0 --warmup 15 --iterations 20 --denoise-steps 5

# Step sweep (1,2,5,10,20 steps):
for N in 1 2 5 10 20; do
    python /data1/ybyang/vlla/scripts/exp09a_cosmos_policy_profiling.py \
        --gpu 0 --warmup 15 --iterations 20 --denoise-steps $N \
        --output exp/exp09a/results_steps_${N}.json
done
```

## Setup on xdlab23

```bash
# 1. Clone cosmos-policy (if not synced via bundle)
cd /data1/ybyang/vlla/vendor
git clone https://github.com/NVlabs/cosmos-policy.git

# 2. Build Docker
cd cosmos-policy
docker build -t cosmos-policy docker

# 3. Launch container
docker run -u root \
    -v /data1/ybyang:/data1/ybyang \
    -v $HOME/.cache:/home/cosmos/.cache \
    --gpus all --ipc=host -it --rm \
    -w /data1/ybyang/vlla \
    cosmos-policy bash

# 4. Inside container: run profiling
nvidia-smi -pm 1
python scripts/exp09a_cosmos_policy_profiling.py --gpu 0
```

## Output

- `results_profiling.json` — canonical 5-step profiling
- `results_steps_{N}.json` — step sweep (1,2,5,10,20)
