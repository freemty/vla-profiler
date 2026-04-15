# Qwen2.5-VL Vision Token Spatial Mapping

> 从 processor 输出定位 vision tokens 在序列中的位置和对应的 image patch grid 尺寸。

## Problem

Attention overlay 需要将 token-level attention 映射回图像空间 (H_patch, W_patch)。需要知道：
1. Vision tokens 在 input_ids 序列中的起止位置
2. Vision tokens 对应的 patch grid 维度 (行数 x 列数)
3. 原始图像尺寸 (用于 overlay resize)

## Cause

Qwen2.5-VL 的 vision tokenization pipeline:
- ViT 将图像切成 14x14 patches
- 经 spatial merge (2x2) 后 token 数量减少 4 倍
- Processor 返回 `image_grid_thw` tensor: `[temporal, height_patches, width_patches]`
- Input_ids 中使用 `<|image_pad|>` (token ID 151655) 作为 vision placeholder

## Solution

### 1. 获取 patch grid 维度

```python
# processor 返回的 model_inputs 中包含 image_grid_thw
grid_thw = model_inputs.image_grid_thw.tolist()  # [[1, 58, 86]] for 单图
# temporal=1, h_patches=58, w_patches=86
# 总 vision tokens = 1 * 58 * 86 = 4988
```

### 2. 定位 vision token 在序列中的位置

```python
# 方案 A: 从 tokenizer 动态获取 token ID
vision_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

# 方案 B: 扫描 input_ids 中的连续 vision token 块
input_ids = model_inputs.input_ids[0].tolist()
ranges = []
in_vision = False
start = 0
for idx, tid in enumerate(input_ids):
    if tid == vision_token_id and not in_vision:
        start = idx
        in_vision = True
    elif tid != vision_token_id and in_vision:
        ranges.append((start, idx))
        in_vision = False
if in_vision:
    ranges.append((start, len(input_ids)))
# ranges = [(2, 4990)] for 单图 — vision tokens 从 idx 2 到 4990
```

### 3. 获取原始图像尺寸

```python
# image_inputs 是 PIL Image 列表 (from qwen_vl_utils.process_vision_info)
for img in image_inputs:
    w, h = img.size  # PIL: (width, height)
    # 注意 PIL 的 .size 是 (w, h)，需要转为 (h, w) 存储
```

### 4. 映射 attention 到图像空间

```python
# attn_mean: (seq_q, seq_k) 已平均过 heads
# 取 visual key tokens 的 column-mean = "每个 visual token 被关注的程度"
visual_attn = attn_mean[:, start:end].mean(axis=0)  # (n_visual_tokens,)
heatmap = visual_attn.reshape(h_patches, w_patches)  # (58, 86)
# 然后 cv2.resize 到原始图像尺寸做 overlay
```

## Key Numbers (Qwen2.5-VL-7B, 单图 demo.jpeg 756x1008)

| 指标 | 值 |
|------|-----|
| image_grid_thw | [1, 58, 86] |
| vision tokens | 4988 |
| vision_token_id | 151655 (`<\|image_pad\|>`) |
| vision range in input_ids | (2, 4990) |
| Q heads | 28 |
| KV heads | 4 (GQA) |
| head_dim | 128 |
| total seq_len | ~5100 (视 prompt 长度) |

## Notes
- Date: 2026-04-15
- Environment: xdlab23, Qwen2.5-VL-7B-Instruct, transformers 5.0.0.dev0
- 不同图像分辨率会产生不同的 `image_grid_thw`，取决于 processor 的 min_pixels/max_pixels 设置
- 多图输入时 `image_grid_thw` 有多行，`ranges` 有多个区间
