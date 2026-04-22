# NitroGen Controller Deployment

> NitroGen 500M DiT profiling 部署到 xdlab23 时遇到的 5 个问题及修复

## Problem 1: Hydra "Could not load 'base'"

子 config `configs/nitrogen/profiling.yaml` 的 defaults 缺少 `_self_`，导致 Hydra 无法正确 merge。

**Fix:** 在 defaults 中加 `- _self_`：
```yaml
defaults:
  - /base
  - _self_
```

参见 `docs/knowhow/toolchain/hydra-config-patterns.md` Problem 3。

## Problem 2: "Unsupported hook_mode 'random'"

`controller_config.mode: random` 被 `_create_controller()` 误解析为 `hook_mode`，因为 BaseController 从 controller_config 中读 `mode` 字段作为 hook_mode。

**Fix:** 将 config key 从 `mode: random` 改为 `weight_mode: random`，保留 `mode: profiling` 给 hook_mode。controller 代码中用 `getattr(cfg, "weight_mode", getattr(cfg, "mode", "random"))` 读取。

**教训:** controller_config 中的 `mode` 字段被 framework 层面截取了，自定义含义需用不同的 key name。

## Problem 3: SigLIP 下载超时 (服务器无外网)

xdlab23 无法访问 HuggingFace。`SiglipVisionModel.from_pretrained("google/siglip-large-patch16-256")` 超时。

**Fix:** Random weight 模式下不使用 `from_pretrained()`，改为从 `SiglipVisionConfig` 直接构建：
```python
from transformers import SiglipVisionConfig, SiglipVisionModel
vision_cfg = SiglipVisionConfig(
    hidden_size=768, intermediate_size=3072,
    num_hidden_layers=27, num_attention_heads=16,
    image_size=256, patch_size=16,
)
random_vision = SiglipVisionModel(vision_cfg)
model.vision_encoder = random_vision.vision_model
```

**教训:** 在离线环境做 timing-only profiling 时，所有 `from_pretrained()` 调用都需要替换为 config-based 构建。

## Problem 4: NitroGen imports 作用域问题

Random weight 路径绕过了 `NitroGen.__init__()`（使用 `__new__` + 手动构建），但 `DiT` 和 `SelfAttentionTransformer` 只在 NitroGen 的 module 内部导入。

**Fix:** 在 `init_pipeline()` 中显式导入：
```python
from nitrogen.flow_matching_transformer.modules import DiT, DiTConfig, SelfAttentionTransformer, SelfAttentionTransformerConfig
```

**教训:** 用 `__new__` 绕过 `__init__` 构建模型时，需要手动导入所有子模块类。

## Problem 5: k sweep 不生效 (所有 k 值 ~130ms)

`model_inference()` 中 `num_steps = model.num_inference_timesteps` 硬编码了步数，忽略了 inputs 中传入的 `num_inference_steps`。

**Fix:**
```python
num_steps = inputs.get("num_inference_steps", model.num_inference_timesteps)
```

**教训:** sweep 参数必须从 `inputs` dict 读取，而非从模型默认值读取。这是 k sweep / resolution sweep 等参数扫描实验的通用模式。

## Commands

```bash
# 部署 NitroGen 到 xdlab23 (GitHub 被防火墙封锁)
# 本地 sparse clone
git clone --filter=blob:none --sparse https://github.com/MineDojo/NitroGen.git /tmp/NitroGen
cd /tmp/NitroGen && git sparse-checkout set nitrogen setup.py pyproject.toml
# 打包传输
tar czf /tmp/NitroGen.tar.gz -C /tmp NitroGen
scp -P 66 /tmp/NitroGen.tar.gz xdlab23_yang:/data1/ybyang/
# 服务器端安装
ssh xdlab23_yang "cd /data1/ybyang && tar xzf NitroGen.tar.gz && cd NitroGen && pip install -e ."
```

## Codex Adversarial Review Findings (2026-04-22)

Codex challenge 模式对 NitroGen controller diff 做了对抗性审查，发现以下问题:

### High: `weights_only=False` (L222)
`torch.load(..., weights_only=False)` 允许任意 pickle 执行。内部 profiling 工具风险低，但最佳实践应用 `weights_only=True` + allowlist。

### High: `strict=False` 静默忽略 key mismatch (L224)
`load_state_dict(..., strict=False)` 不报告 missing/unexpected keys。如果 checkpoint 结构不匹配，profiling 会在部分初始化的模型上跑。**建议:** 至少 log 返回的 missing/unexpected keys。

### Medium: YAML config 部分字段未接入 (L94)
`init_pipeline()` 只读顶层字段，`controller_config` 下的 `repo_path`, `dit_*`, `vl_*` 等被忽略。当前实验靠硬编码默认值工作。

### Medium: warmup/iterations 配置不生效 (profiling.yaml:41)
YAML 写 `warmup: 5` + `iterations: 20`，但 runner 用 `base.yaml` 的 `num_warmup_runs=3` / `num_benchmark_runs=10`。

### Medium: 硬编码 `num_visual_tokens=256` (L281)
任何 image/patch size 变化会导致 crash 或静默丢弃 tokens。

**教训:** ML profiling 代码也应做安全审查。`torch.load(weights_only=False)` + `strict=False` 是常见但有风险的 pattern。

## Notes
- Date: 2026-04-22
- Updated: 2026-04-22 (Codex review findings)
- Environment: xdlab23, conda vit-probe, PyTorch 2.x, RTX 5880 Ada
- Model: NitroGen 500M (407M actual: Vision=199M, VL-SA=28M, DiT=174M)
