# LingBot-VLA Integration

> Flow VLA (Pi0-style) 集成到 profiling framework 的关键问题和解决方案

## Problem
将 lingbotvla (Qwen2.5-VL-3B + flow action head) 集成到 vlla profiling framework，遇到一系列 API 兼容性和 PyTorch hook 问题。

## Cause
lingbotvla 依赖旧版 lerobot API (`lerobot.common.policies`)，其 config 系统、attention 实现、forward 调用方式均与标准 HF 模型不同。

## Solutions

### 1. lerobot API 兼容 (common → policies)

lingbotvla 使用 `from lerobot.common.policies.pi0.configuration_pi0 import PI0Config`，但 lerobot ≥0.3.3 改为 `lerobot.policies`。

**Fix:** 创建 compat shim 目录 `lerobot/common/policies/`：
- 直接复制 `configuration_pi0.py` (避免 import chain)
- Monkey-patch `PreTrainedPolicy` 移除 `predict_action_chunk` abstractmethod

### 2. PI0Config 字段过滤

config.json 含 lingbotvla 专有字段 (type, enable_expert_vision 等)，PI0Config dataclass 不接受。

**Fix:** 用 `dataclasses.fields(PI0Config)` 获取有效字段，过滤 kwargs 构造，然后 `setattr` 附加额外属性。

### 3. PyTorch hooks 不被 `.forward()` 直接调用触发

`sample_actions()` 和 `predict_velocity()` 调用 `self.qwenvl_with_expert.forward(...)` 而非 `self.qwenvl_with_expert(...)`。PyTorch hooks 只在 `__call__` 时触发。

**Fix:** Patch 源码将 `.forward(` 替换为 `(`。

### 4. Flex attention dtype 不兼容 (PyTorch 2.8)

PyTorch 2.8 的 flex_attention block_mask 要求 bool dtype，lingbotvla 传入 Long。

**Fix:** `pi0_config.attention_implementation = "eager"` (fa2 也不行，lingbotvla 报 "not implemented")

### 5. Eager attention mask dtype

lingbotvla 的 attention_mask 是 Long，`torch.where` 需要 bool。

**Fix:** 在 `utils.py` 的 `our_eager_attention_forward` 中添加 `attention_mask = attention_mask.bool()`

### 6. 图像输入格式

lingbotvla 期望 pre-patchified tensors `(B, N, num_patches, patch_dim)` 而非 raw pixels `(B, N, 3, H, W)`。224×224 输入: num_patches=256, patch_dim=1176。

## Commands

```bash
# 服务器端 patch (qwenvl_with_expert forward → __call__)
sed -i 's/self.qwenvl_with_expert.forward(/self.qwenvl_with_expert(/' modeling_lingbot_vla.py

# attention mask bool patch
# 在 our_eager_attention_forward 中添加: attention_mask = attention_mask.bool()
```

## BaseVLAController 继承链 (2026-04-20)

`_register_capture_hook` 原本只定义在 `BaseVLMController` 上。`LingBotVLAController` 继承的是 `BaseVLAController` → `BaseController`，调用 `self._register_capture_hook()` 会 AttributeError。

**Fix:** 在 `BaseVLAController` 中也定义 `_register_capture_hook`（相同逻辑）。

**教训:** 新建 Controller 子类时，检查 analysis hooks 方法的继承来源。两条继承链 (VLM vs VLA) 各自需要独立的 hook 方法。

## LIBERO Eval 兼容性问题 (2026-05-13)

### Problem
用 vit-probe env (transformers 4.57, lerobot 0.5.1) 跑 `lingbotvla` 包做 LIBERO eval 时连续报错。

### Error 1: lerobot import 路径变更
```
ModuleNotFoundError: No module named 'lerobot.common'
```
lerobot 0.5.1 把 `lerobot.common.policies.pi0` 改成了 `lerobot.policies.pi0`。lingbotvla 包 8 处硬编码旧路径。

**Fix**: 批量 sed
```bash
find /data1/ybyang/lingbot-vla/lingbotvla -name "*.py" -not -path "*__pycache__*" \
  -exec sed -i "s/from lerobot\.common\.policies/from lerobot.policies/g" {} + \
  -exec sed -i "s/from lerobot\.common\.datasets/from lerobot.datasets/g" {} +
```

### Error 2: transformers 4.57 缺 LossKwargs
```
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```
`modeling_lingbot_vla.py` import `LossKwargs`，这个 class 在 transformers 4.57 里不存在（可能是 4.40 时代的 API）。

**无法 patch**——`LossKwargs` 被用在 class 定义里，不是简单的 import 替换。

### 解法
不用 lingbotvla 包加载模型。用我们自己的 `LingBotVLAController` 的加载路径：
1. 直接用 `lerobot.policies.pi0.configuration_pi0.PI0Config`
2. 用 `safetensors.safe_open` 加载权重
3. 手动构建 `LingbotVlaPolicy`（绕过 `modeling_lingbot_vla.py` 的 import chain）

验证：`from src.controllers.lingbot_vla_controller import LingBotVLAController` 在 vit-probe env 成功。

## Notes
- Date: 2026-04-20 (初版) / 2026-05-13 (LIBERO eval 兼容性追加)
- Environment: xdlab23, vit-probe conda env (transformers 4.57, lerobot 0.5.1)
- Server patches: `/data1/ybyang/lingbot-vla/lingbotvla/` (lerobot import 8 处已 sed 修)
