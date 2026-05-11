# OpenVLA / StarVLA OFT Controller Integration

> Prismatic VLM + Qwen2.5-VL OFT profiling 集成的 5 个陷阱和解法。

## Problem

exp11a (OpenVLA-OFT) 和 exp11b (StarVLA-OFT) 在 xdlab23 上首次部署时连续命中 5 个错误，每个需要一轮 fix-deploy-test。

## Cause

1. OpenVLA 使用自定义 model class (非标准 HuggingFace)
2. Prismatic VLM 的 dual vision encoder 有特殊输入格式
3. Qwen2.5-VL 的 ViT 需要 `grid_thw` 参数
4. 模型结构路径与文档不一致

## Solution

### Issue 1: AutoModelForCausalLM → AutoModelForVision2Seq

OpenVLA 的 `config.json` 有 `"auto_map": {"AutoModelForVision2Seq": "..."}` — 必须用 `AutoModelForVision2Seq`，不是 `AutoModelForCausalLM`。

### Issue 2: `_supports_sdpa` AttributeError

Prismatic 的 custom model class 没实现 `_supports_sdpa` 属性。HF transformers 新版默认尝试 SDPA dispatch 检查会崩。

Fix: 手动设 `config._attn_implementation = "eager"`。

### Issue 3: Prismatic 需要 6-channel pixel_values

DINOv2 (3ch) + SigLIP (3ch) 拼接 → `pixel_values.shape = (B, 6, H, W)`，不是标准 3 通道。

代码: `torch.split(pixel_values, [3, 3], dim=1)` — 如果输入只有 3ch 会崩。

### Issue 4: Qwen2.5-VL ViT 需要 grid_thw + pre-patchified input

`model.visual.forward()` 签名是 `(pixel_values, grid_thw)` 不是 `(pixel_values)`。

输入格式:
- `pixel_values`: `(num_patches, patch_dim)` 其中 `patch_dim = C * temporal * P²`
- `grid_thw`: `(batch, 3)` tensor `[[temporal, grid_h, grid_w]]`

对 224x224 输入: `num_patches = 256, patch_dim = 3 * 2 * 14 * 14 = 1176`

### Issue 5: embed_tokens 路径

- OpenVLA (Prismatic): `model.language_model.model.embed_tokens`
- Qwen2.5-VL: `model.model.language_model.embed_tokens` (不是 `model.model.embed_tokens`)

诊断命令:
```python
for name, mod in model.named_modules():
    if "embed_tokens" in name:
        print(f"Found: {name}")
        break
```

## Commands

```bash
# 检查 OpenVLA config 的 auto_map
python -c "import json; print(json.dumps(json.load(open('config.json')).get('auto_map', {}), indent=2))"

# 检查 timm 版本 (Prismatic 需要 >=0.9.10,<1.0.0)
python -c "import timm; print(timm.__version__)"
pip install "timm>=0.9.10,<1.0.0"

# 下载 OpenVLA via hf-mirror (防火墙环境)
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b', cache_dir='/data1/ybyang/huggingface', endpoint='https://hf-mirror.com')
"
```

## Notes
- Date: 2026-05-11
- Environment: xdlab23, vit-probe conda env, RTX 5880 Ada 48GB
- timm 1.0.27 → downgraded to 0.9.16 for Prismatic compat
- OpenVLA 期望 transformers==4.40.1 但我们用 4.57.1 — 需要 eager attn workaround
