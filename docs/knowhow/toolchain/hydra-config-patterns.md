# Hydra Config Patterns & Gotchas

> Hydra 1.3 的配置合并、struct mode、嵌套 config 的常见陷阱和解法。

## Problem 1: Config group nesting
`--config-name qwen_vl_7b/profiling` 把 config 嵌套在 `qwen_vl_7b:` key 下，顶层属性访问失败。

## Cause
Hydra 把子目录路径当 config group 处理，自动加一层嵌套。

## Solution
在 entry point 中 unwrap：
```python
@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(raw_cfg: DictConfig):
    OmegaConf.set_struct(raw_cfg, False)
    keys = list(raw_cfg.keys())
    if len(keys) == 1 and isinstance(raw_cfg[keys[0]], DictConfig):
        cfg = raw_cfg[keys[0]]  # unwrap nested group
    else:
        cfg = raw_cfg
```

## Problem 2: Struct mode blocks override
`omegaconf.errors.ConfigAttributeError: Key 'device' is not in struct` — command-line override 被拒绝。

## Solution
在 entry point 早期关闭 struct mode：
```python
OmegaConf.set_struct(cfg, False)
```

## Problem 3: Defaults merge order
子 config 覆盖不生效，base.yaml 的值优先。

## Solution
在子 config 的 defaults 中加 `_self_`：
```yaml
defaults:
  - /base
  - _self_    # 确保本文件的值覆盖 base
```

## Problem 4: OmegaConf DictConfig 传给第三方库
`qwen_vl_utils.process_vision_info(messages)` 报 `TypeError: string indices must be integers` — 因为 messages 是 OmegaConf DictConfig 而非 plain dict。

## Solution
```python
from omegaconf import OmegaConf
if hasattr(messages, '_iter_ex'):
    messages = OmegaConf.to_container(messages, resolve=True)
```

## Problem 5: `${variable}` interpolation 在 unwrap 后丢失
base.yaml 定义 `HF_PATH: ${oc.env:HF_HOME,...}`，子 config 引用 `${HF_PATH}`，但 unwrap 后变量丢失。

## Solution
子 config 直接用 `oc.env` 而不是引用 base 的变量：
```yaml
# WRONG: model_name: "${HF_PATH}/Qwen/..."
# CORRECT:
model_name: "${oc.env:HF_HOME,/data1/ybyang/huggingface}/Qwen/..."
```

## Problem 6: ListConfig not JSON serializable in save_results

`json.dump()` 对包含 OmegaConf ListConfig/DictConfig 的 dict 报 `TypeError: Object of type ListConfig is not JSON serializable`。

## Cause

`cfg.inputs` 从 Hydra config 读出来后仍是 OmegaConf 容器类型，传到 `save_results()` 时没有转换。

## Solution

在 JSON 序列化前检查并转换：
```python
from omegaconf import OmegaConf

raw_messages = inputs.get("messages", [])
messages_plain = (
    OmegaConf.to_container(raw_messages, resolve=True)
    if hasattr(raw_messages, "_iter_ex")
    else raw_messages
)
```

`_iter_ex` 属性是 OmegaConf 容器的可靠检测方式（`isinstance` 需要额外 import）。

## Problem 7: launch_exp.sh device override rejected by struct mode

`device=cuda:0` 作为 CLI override 被 Hydra struct mode 拒绝：`ConfigAttributeError: Key 'device' is not in struct`。

## Cause

`launch_exp.sh` 在命令行传 `device=cuda:0`，但子 config (如 `attention_overlay.yaml`) 经 defaults 合并后启用了 struct mode，`device` key 不在合并后的 struct 中。

## Solution

删除 launch script 中的 `device=cuda:0` override — `base.yaml` 已定义 `device: "cuda:0"`，所有子 config 通过 defaults merge 继承。

## Notes
- Date: 2026-04-15
- Environment: hydra-core 1.3.2, omegaconf 2.3.0
