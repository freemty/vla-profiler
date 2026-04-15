# xdlab23 Model Weights Management

> HuggingFace/ModelScope 权重路径管理、下载、symlink 策略。

## Problem
xdlab23 firewall 屏蔽 HuggingFace 直连，模型需要通过 ModelScope 下载并 symlink。

## Cause
校园网 (ZJU) firewall 限制 HTTPS 到 HuggingFace CDN。ModelScope 国内源无此限制。

## Solution

### 下载新模型 (via ModelScope)
```bash
ssh xdlab23_yang
conda activate vit-probe
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/data1/ybyang/modelscope')
print(f'Downloaded to: {model_dir}')
"
```

### 创建 HuggingFace symlink
```bash
mkdir -p /data1/ybyang/huggingface/Qwen
ln -sf /data1/ybyang/modelscope/Qwen/Qwen2.5-VL-7B-Instruct \
       /data1/ybyang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct
```

### 已有模型清单 (2026-04-15)

| 模型 | HF 路径 | 实际位置 | 大小 |
|------|---------|---------|------|
| Qwen2.5-VL-7B-Instruct | `/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct` | symlink → modelscope | ~16GB (5 shards) |
| Qwen3-VL-4B-Instruct | `/data1/ybyang/huggingface/Qwen/Qwen3-VL-4B-Instruct` | direct | incomplete (missing shard 1) |

### 在 Hydra config 中引用
```yaml
model_name: "${oc.env:HF_HOME,/data1/ybyang/huggingface}/Qwen/Qwen2.5-VL-7B-Instruct"
```
运行时设置 `export HF_HOME=/data1/ybyang/huggingface`。

## Notes
- Date: 2026-04-15
- Environment: xdlab23, modelscope 1.35.3
- ModelScope 下载速度: ~10MB/s (校园网)
- 总下载时间 (Qwen2.5-VL-7B): ~11 分钟
