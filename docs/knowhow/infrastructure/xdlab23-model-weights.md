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

### ModelScope 没有的模型 — 需要其他方式

部分模型在 ModelScope 上返回 404（如 `openvla/openvla-7b`）。备选方案：
1. 通过代理从 HuggingFace 下载
2. 在有网络的机器上下载后 scp 到 server
3. 用 `huggingface-cli download` + proxy

### 已有模型清单 (2026-04-15)

| 模型 | HF 路径 | 实际位置 | 大小 | 状态 |
|------|---------|---------|------|------|
| Qwen2.5-VL-7B-Instruct | `/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct` | symlink → modelscope | ~16GB | OK |
| Qwen3-VL-4B-Instruct | `/data1/ybyang/huggingface/Qwen/Qwen3-VL-4B-Instruct` | direct | ? | incomplete (missing shard 1) |
| OpenVLA-7B | — | — | ~16GB | ModelScope 404, 需其他方式 |

### 在 Hydra config 中引用
```yaml
model_name: "${oc.env:HF_HOME,/data1/ybyang/huggingface}/Qwen/Qwen2.5-VL-7B-Instruct"
```
运行时设置 `export HF_HOME=/data1/ybyang/huggingface`。

### LingBot-VLA-4B (2026-04-20)

`robbyant/lingbot-vla-4b` — HuggingFace ConnectTimeout 确认，需走 ModelScope。

```bash
source /data1/ybyang/vlla/.venvs/lingbot-vla/bin/activate
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Robbyant/lingbot-vla-4b', cache_dir='/data1/ybyang/modelscope')
print(f'Downloaded to: {model_dir}')
"
# Then symlink:
mkdir -p /data1/ybyang/huggingface/robbyant
ln -sf /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b \
       /data1/ybyang/huggingface/robbyant/lingbot-vla-4b
```

| 模型 | HF 路径 | 实际位置 | 大小 | 状态 |
|------|---------|---------|------|------|
| lingbot-vla-4b | `/data1/ybyang/huggingface/robbyant/lingbot-vla-4b` | symlink → modelscope | ~8GB | pending |
| lingbot-vla-4b-posttrain-robotwin | — | — | ~8GB | pending |

## Notes
- Date: 2026-04-20 (updated)
- Environment: xdlab23, modelscope 1.35.3
- HuggingFace 直连确认 ConnectTimeout (2026-04-20 实测)
- ModelScope 下载速度: ~10MB/s (校园网)
- 总下载时间 (Qwen2.5-VL-7B): ~11 分钟
