# GQA (Grouped Query Attention) in Attention Analysis

> VLM 的 Q 和 K head count 不同时，attention score 计算需要 repeat_interleave K 来匹配 Q。

## Problem
Qwen2.5-VL-7B attention 分析 crash：
```
RuntimeError: shape '[1, 1274, 28, 128]' is invalid for input of size 652288
```
试图将 K tensor (hidden_dim=512) reshape 成 28 heads × 128 dim，但实际只有 4 KV heads。

## Cause
Qwen2.5-VL-7B 使用 GQA：
- Q heads: 28, Q output dim: 28 × 128 = 3584
- KV heads: 4, K output dim: 4 × 128 = 512

`_compute_attention_scores` 原始实现假设 Q 和 K 有相同的 head count。

## Solution
```python
num_q_heads = q_tensor.shape[-1] // head_dim  # 28
num_k_heads = k_tensor.shape[-1] // head_dim  # 4

q = q_tensor.view(batch, seq_q, num_q_heads, head_dim).transpose(1, 2)
k = k_tensor.view(batch, seq_k, num_k_heads, head_dim).transpose(1, 2)

# Expand K heads to match Q via repeat_interleave
if num_k_heads != num_q_heads:
    repeat_factor = num_q_heads // num_k_heads  # 28/4 = 7
    k = k.repeat_interleave(repeat_factor, dim=1)
```

加 head_dim divisibility check 防止静默错误：
```python
if q_tensor.shape[-1] % head_dim != 0:
    raise ValueError(f"Q hidden dim {q_tensor.shape[-1]} not divisible by head_dim {head_dim}")
```

## Commands
```bash
# 查看模型的 Q/K head 配置
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration
model = ...
print(f'num_heads: {model.config.text_config.num_attention_heads}')      # 28
print(f'num_kv_heads: {model.config.text_config.num_key_value_heads}')  # 4
"
```

## Notes
- Date: 2026-04-15
- 影响所有使用 GQA 的模型：Qwen2.5-VL, Llama-3, Mistral 等
- `repeat_interleave` 是 HuggingFace 自身 GQA 实现的标准做法
