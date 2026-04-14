# Deploy vlla to xdlab23

## First-time setup

```bash
# 1. Sync code to remote
bash scripts/sync_to_remote.sh

# 2. SSH to server
ssh xdlab23_yang

# 3. Verify environment (reuse rope2sink's vit-probe conda env)
conda activate vit-probe
cd /data1/ybyang/vlla
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"

# 4. Install additional dependencies if needed
pip install qwen-vl-utils easydict
```

## Run profiling experiment

```bash
# From local machine — single command
ssh xdlab23_yang "cd /data1/ybyang/vlla && bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling"

# Or SSH in and run interactively
ssh xdlab23_yang
cd /data1/ybyang/vlla
conda activate vit-probe
CUDA_VISIBLE_DEVICES=0 python -m src.run_tasks \
    --config-path ../configs --config-name qwen_vl_7b/profiling
```

## Run attention analysis

```bash
ssh xdlab23_yang "cd /data1/ybyang/vlla && bash scripts/launch_exp.sh 1 qwen_vl_7b/attention"
```

## Download results

```bash
bash scripts/download-results.sh
```

## Update code

```bash
bash scripts/sync_to_remote.sh
```

## Quick reference

| What | Command |
|------|---------|
| SSH | `ssh xdlab23_yang` |
| Server path | `/data1/ybyang/vlla` |
| Conda env | `vit-probe` (shared with rope2sink) |
| HF cache | `/data1/ybyang/huggingface` |
| GPUs | 8x RTX 5880 Ada 48GB |
