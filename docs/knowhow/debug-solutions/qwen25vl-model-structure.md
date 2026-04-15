# Qwen2.5-VL Model Structure (transformers 5.0.0.dev0)

> Qwen2.5-VL 的内部模块路径与早期版本不同，需要用 language_model 中间层访问 transformer layers。

## Problem
`self.pipeline.model.model.layers` 抛出 `AttributeError: 'Qwen2_5_VLModel' object has no attribute 'layers'`

## Cause
transformers 5.0.0.dev0 中 Qwen2.5-VL 的模型结构多了一层 `language_model`：
```
model (Qwen2_5_VLForConditionalGeneration)
  └── model (Qwen2_5_VLModel)
      ├── visual (Qwen2_5_VisionTransformerPretrainedModel)  ← vision encoder
      └── language_model (Qwen2_5_VLTextModel)
          ├── embed_tokens
          ├── layers (ModuleList, len=28)  ← transformer layers
          ├── norm
          └── rotary_emb
```

## Solution
```python
# WRONG (old structure)
model.model.layers

# CORRECT (transformers 5.0.0.dev0)
model.model.language_model.layers    # 28 transformer layers
model.model.visual                    # vision encoder (unchanged)
```

## Commands
```bash
# Inspect model structure on server
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'path/to/model', torch_dtype=torch.bfloat16, device_map='auto'
)
for name, child in model.model.named_children():
    print(f'model.model.{name}: {type(child).__name__}')
"
```

## Notes
- Date: 2026-04-15
- Environment: xdlab23, transformers 5.0.0.dev0, Qwen2.5-VL-7B-Instruct
- Also: `attn_implementation="flash_attention_2"` fails if flash_attn not installed — use `"sdpa"` instead
