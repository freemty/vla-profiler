# LingBot-VA (Full WAM) Integration Issues

> LingBot-VA WanTransformer3DModel 构造参数、VAE 配置、timestep shape 等 7 个陷阱。

## Problem

LingBot-VA 使用自定义 WanTransformer3DModel (非 diffusers 版本)，API 与 HuggingFace diffusers 的同名类有显著差异。集成过程中遇到多个构造/推理错误。

## Cause

LingBot-VA fork 了 Wan2.2 代码并大幅修改：加入 action_mode 分支、自定义 FlowMatchScheduler、streaming VAE 包装器。文档极少，只能读源码确定参数。

## Solution

### Issue 1: WanTransformer3DModel 构造参数

LingBot-VA 版本 (wan_va/modules/model.py) 与 diffusers 版本签名不同：

```python
# WRONG — diffusers API
WanTransformer3DModel(num_layers=40, num_attention_heads=40, qk_norm=True)

# CORRECT — LingBot-VA API
WanTransformer3DModel(
    patch_size=[1, 2, 2],
    num_attention_heads=24,    # not 40
    attention_head_dim=128,
    in_channels=48,
    out_channels=48,
    action_dim=30,             # LingBot-VA specific
    text_dim=4096,
    freq_dim=256,
    ffn_dim=14336,             # not 13824
    num_layers=30,             # not 40
    cross_attn_norm=True,
    eps=1e-6,
    rope_max_seq_len=1024,
    attn_mode="torch",         # or "flash"
)
# NO qk_norm parameter — will raise TypeError
```

### Issue 2: AutoencoderKLWan z_dim

Diffusers 0.35.2 使用 `z_dim`，不是 `latent_channels`：

```python
# WRONG
AutoencoderKLWan(latent_channels=48)  # TypeError

# CORRECT
AutoencoderKLWan(z_dim=48, in_channels=3, out_channels=3)
```

### Issue 3: VAE latents_mean length mismatch

Random-init VAE 的 `config.latents_mean` 默认长度为 16，但 z_dim=48。归一化前必须检查：

```python
if len(vae.config.latents_mean) == mu.shape[1]:
    mu = (mu - mean) / std
# else: skip normalization (random-init mode)
```

### Issue 4: init_latent shape mismatch

VAE spatial factor 是 8x，但 latent grid 使用 H//16 (包含 patchify factor)。Random mode 不能用 VAE output 作为 init_latent：

```python
# WRONG — VAE output spatial is H//8, but latent grid expects H//16
init_latent = mu[:, :, 0:1]

# CORRECT — use random tensor at correct shape
init_latent = torch.randn(1, 48, F, latent_H, latent_W, device=device, dtype=dtype)
```

### Issue 5: Timestep batch dimension

Model forward 期望 timesteps shape `[B, F]`，不是 `[F]`：

```python
# WRONG — 1D tensor
timestep_vec = torch.ones([F], ...) * t  # IndexError on dim=1

# CORRECT — 2D with batch dim
timestep_vec = torch.ones([1, F], dtype=torch.float32, device=device) * t
```

### Issue 6: action_mode flag routing

同一个 transformer 服务 video 和 action。`action_mode` 控制：
- Input embedding: `patch_embedding_mlp` (video) vs `action_embedder` (action)
- Timestep: `condition_embedder` (video) vs `condition_embedder_action` (action)
- Output: `proj_out` (video) vs `action_proj_out` (action)

```python
# Video denoising
noise_pred = transformer(input_dict, action_mode=False)

# Action denoising
action_pred = transformer(input_dict, action_mode=True)
```

### Issue 7: FlowMatchScheduler 是自定义的

不是 diffusers 的 FlowMatchEulerDiscreteScheduler。在 `wan_va/utils/utils.py` 中定义：

```python
from wan_va.utils import FlowMatchScheduler
video_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
# Video shift=5.0, Action shift=1.0 — different noise schedules
```

## Commands

```bash
# 验证 WanTransformer3DModel 签名
cd /data1/ybyang/lingbot-va
python -c "from wan_va.modules.model import WanTransformer3DModel; import inspect; print(inspect.signature(WanTransformer3DModel.__init__))"

# 验证 VAE config
python -c "from diffusers import AutoencoderKLWan; import inspect; print(inspect.signature(AutoencoderKLWan.__init__))"
```

## Notes
- Date: 2026-04-21
- Environment: xdlab23, vit-probe conda env (torch 2.9.0, diffusers 0.35.2)
- Related: `src/controllers/lingbot_va_controller.py`, `scripts/profile_lingbot_va.py`
- See also: `docs/knowhow/debug-solutions/lingbotvla-integration.md` (VLA version, different issues)
