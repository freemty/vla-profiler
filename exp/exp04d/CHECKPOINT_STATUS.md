# LingBot-VA Checkpoint Status (as of 2026-04-28)

**Status:** FOUND

## Available Checkpoints (HuggingFace / hf-mirror)

| Model ID | Description | Size | Released |
|----------|-------------|------|----------|
| `robbyant/lingbot-va-posttrain-libero-long` | **LIBERO posttrain (target)** | ~22.7 GB | 2026-04-24 |
| `robbyant/lingbot-va-base` | Pretrained base (shared backbone) | ~22.7 GB | 2026-01-29 |
| `robbyant/lingbot-va-posttrain-robotwin` | RoboTwin posttrain | ~22.7 GB | 2026-01-29 |

## Download Command

```bash
# On xdlab23 (use hf-mirror since github/HF blocked by firewall)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  robbyant/lingbot-va-posttrain-libero-long \
  --local-dir /data1/ybyang/huggingface/robbyant/lingbot-va-posttrain-libero-long
```

## Config Setup

After download, update the config to point to the checkpoint:

```python
# /data1/ybyang/lingbot-va/wan_va/configs/va_libero_cfg.py
va_libero_cfg.wan22_pretrained_model_name_or_path = "/data1/ybyang/huggingface/robbyant/lingbot-va-posttrain-libero-long"
```

Also set `attn_mode` for inference (per README):
```
# Edit <model-path>/transformer/config.json
# Set "attn_mode": "torch"  (for inference)
# NOT "sdpa" (which is for training)
```

## Model Structure

```
lingbot-va-posttrain-libero-long/
  text_encoder/     # 3x safetensors shards
  tokenizer/        # SentencePiece tokenizer
  transformer/      # 3x safetensors shards (MoT backbone)
  vae/              # VAE decoder
```

## Sources Checked

- **lingbot-va/README.md**: Lists `lingbot-va-base` and `lingbot-va-posttrain-robotwin`. LIBERO posttrain not listed yet (released 4 days after latest README update)
- **HuggingFace (hf-mirror.com)**: `robbyant/lingbot-va-posttrain-libero-long` found, public, Apache-2.0, 22.7 GB
- **ModelScope (modelscope.cn)**: No LIBERO posttrain variant found (only base + robotwin)
- **lingbot-va repo file search**: No weight files in repo (as expected, weights hosted externally)
- **eval scripts**: `va_libero_cfg.py` has placeholder path `/path/to/pretrained/model`

## LIBERO Eval Setup

Per README, LIBERO eval uses server-client architecture:
```bash
# 1. Download LIBERO LONG dataset (Google Drive link in README)
bash script/_download_assets.sh

# 2. Launch inference server
bash evaluation/libero/launch_server.sh

# 3. Run evaluation client
bash evaluation/libero/launch_client.sh
```

Also need: LIBERO LONG training dataset for posttrain (if re-training):
- Google Drive: https://drive.google.com/file/d/1QGNkvsb1hlRmRkKCgFlyWitv17sRuagS/view?usp=sharing

## Next Action

1. Download `robbyant/lingbot-va-posttrain-libero-long` via hf-mirror
2. Update `va_libero_cfg.py` with correct path
3. Set `attn_mode` to `"torch"` in `transformer/config.json`
4. Run LIBERO eval to get real (non-random-weight) numbers
