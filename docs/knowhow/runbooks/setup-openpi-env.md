# Setup OpenPI Environment for Pi-Zero Profiling

> Pi-Zero 需要独立 conda env，因为依赖与主环境冲突 (torch 2.7 vs 2.9, transformers 4.53 vs 5.0)。

## Prerequisites

- xdlab23 server access
- CUDA 12.x
- gcloud CLI (for model weight download)

## Setup Steps

### 1. Create isolated conda env

```bash
ssh xdlab23_yang
conda create -n openpi python=3.11 -y
conda activate openpi
```

### 2. Clone OpenPI repo

```bash
cd /data1/ybyang
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
```

Note: If GitHub blocked by firewall, use git bundle from local machine.

### 3. Install dependencies

```bash
# Install uv if not available
pip install uv

# Install OpenPI with PyTorch backend
uv sync
uv pip install -e .
```

### 4. Patch transformers (CRITICAL)

OpenPI modifies transformers internals (AdaRMSNorm, KV cache, precision).

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    $(python -c "import transformers; print(transformers.__path__[0])")/
```

### 5. Download model weights

```bash
# Option A: From GCS (requires gcloud auth)
gcloud storage cp gs://openpi-assets/checkpoints/pi0_base/params .

# Option B: Convert from JAX checkpoint
python scripts/convert_jax_to_pytorch.py --checkpoint_path /path/to/jax/params
```

### 6. Verify

```bash
python -c "
from openpi.policies.pi0 import Pi0Policy
from openpi.training.config import Pi0Config
config = Pi0Config()
policy = Pi0Policy(config)
print('Pi-Zero loaded:', type(policy.model).__name__)
"
```

## Running vlla with Pi-Zero

```bash
# Must use openpi conda env
conda activate openpi

# Then run from vlla directory
cd /data1/ybyang/vlla
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name pizero/profiling
```

## Key Architecture Notes

```
Pi0 Model:
  paligemma_with_expert
    ├── paligemma
    │   ├── model.vision_tower    ← E: SigLIP ViT-So400m/14
    │   └── language_model        ← C: Gemma 2B (prefill, KV cached)
    └── gemma_expert              ← A: Gemma 300M (10 denoise steps)
  action_in_proj, action_out_proj ← action dim ↔ expert width
  time_mlp_*                      ← timestep embedding
```

- C phase runs ONCE → caches KV
- A phase runs 10x → only 300M Expert per step
- Action Expert attends to PaliGemma KV (read-only)

## Dependencies (pinned)

| Package | Version |
|---------|---------|
| torch | 2.7.1 |
| transformers | 4.53.2 |
| numpy | <2.0.0 |
| jax[cuda12] | 0.5.3 (optional, for JAX backend) |

## Notes
- Date: 2026-04-16
- Do NOT install openpi in the vit-probe env — dependency conflicts will break Qwen2.5-VL
