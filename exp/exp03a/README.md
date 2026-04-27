# exp03a: LingBot-VLA-4B Inference Profiling

## Motivation

LingBot-VLA 是蚂蚁灵波 (RobbyAnt) 基于 Qwen2.5-VL-3B 的 4B VLA foundation model。
测量其 E/P/D/Action 延迟 breakdown，量化 VLA fine-tuning 相对于 pure VLM 的 overhead。

**对比基线:**
- exp01a: Qwen2.5-VL-7B (pure VLM) — E=253ms/P=156ms/D=18.6ms/tok (single_img)
- exp02a: ACT (LeRobot, ResNet18) — Total ~3ms

**关键问题:**
1. VLA action head 引入多少额外延迟？
2. 3B backbone vs 7B backbone 的 E/P/D scaling？
3. Depth injection 的延迟成本？

## Model

| 变体 | HuggingFace | 描述 |
|------|-------------|------|
| lingbot-vla-4b | robbyant/lingbot-vla-4b | 标准版 (无 depth) |
| lingbot-vla-4b-depth | robbyant/lingbot-vla-4b-depth | Depth 蒸馏版 |

- Backbone: Qwen2.5-VL-3B-Instruct
- Action dim: 最大 75
- Framework: LeRobot 生态
- License: Apache-2.0

## Method

LingBot-VLA 是 flow-matching VLA (非 autoregressive)，阶段结构与纯 VLM 不同:
1. **Encode (E):** Qwen2.5-VL ViT — 处理 pre-patchified 图像 (256 patches × 1176 dim)
2. **Context (C):** qwenvl_with_expert forward — 填充 prefix KV cache (视觉+语言 tokens)
3. **Action (A):** 10-step flow-matching denoise loop — Euler 积分，每步调用 predict_velocity → action_out_proj

与 VLM 的 E/P/D 不同，flow VLA 没有 autoregressive decode，而是固定步数的 denoise loop。

输入条件:
- 单图 (224×224, 1 camera, manipulation)
- 多视角 (2 cameras)

## Hardware

- Server: xdlab23
- GPU: RTX 5880 Ada 48GB
- Env: uv venv `.venvs/lingbot-vla/` (非 conda)

## Commands

LingBot-VLA 走 Hydra (`configs/lingbot_vla_4b/profiling.yaml`)，但因依赖冲突用
uv venv 而非 `vit-probe` conda。入口见 `scripts/setup_lingbot_vla.sh` 建立 env
后，标准 `launch_exp.sh` 即可 (该脚本内部切换 env)。

```bash
# On xdlab23 (from /data1/ybyang/vlla)
bash scripts/launch_exp.sh 0 lingbot_vla_4b/profiling

# Local: pull results
bash scripts/download-results.sh lingbot_vla_4b
```

## Results

**10 runs, mean ± std, CUDA-timed:**

| Input | Encode (E) | Context (C) | Action (A) | Total |
|-------|-----------|-------------|------------|-------|
| single_img | 35.7 ± 0.84ms | 38.3 ± 0.81ms | 0.48 ± 0.01ms | 74.5ms |
| multi_view (2 cam) | 36.3 ± 0.44ms | 38.3 ± 0.24ms | 0.48 ± 0.01ms | 75.0ms |

### 对比分析

| Model | Type | Encode | Context/Prefill | Action/Decode | Total |
|-------|------|--------|----------------|---------------|-------|
| **LingBot-VLA-4B** | Flow VLA (3B) | 35.7ms | 38.3ms | 0.48ms | 74.5ms |
| Qwen2.5-VL-7B (exp01a) | VLM (7B) | 253ms | 156ms (P) | 18.6ms/tok (D) | ~428ms |
| ACT (exp02a) | Lightweight VLA | 2.5ms | — | 0.5ms | ~3ms |

### 关键发现

1. **3B backbone 比 7B 快 ~7x** — Encode 35.7ms vs 253ms (同 ViT 架构，参数量差异)
2. **Context ≈ Encode** — KV cache fill (38ms) 与 vision encoding (36ms) 几乎等价，说明 LLM forward 在 prefix 阶段的计算量与 ViT 相当
3. **Multi-view encode 几乎不增加** (+1.5%) — 因为图像在 ViT 之前已 patchify 聚合，ViT 处理的是合并后的 patch 序列
4. **Flow action 极快 (0.48ms)** — 10-step denoise loop 的 action_out_proj 累计仅 0.48ms，实际 denoise 计算在 context/expert forward 中
5. **Total 74.5ms ≈ 13Hz** — 接近机器人 10Hz 实时控制需求，但仍有 ~25% gap

### 预测 vs 实际

预估 250-350ms，实际 74.5ms — 远超预期。原因:
- 预估基于 7B→3B 的粗略 2x scaling，实际 7x
- Flow VLA 的 fixed-step denoise 比 autoregressive decode 高效得多
- Patchified input 避免了重复 ViT forward

## Status

- [x] 环境搭建 (uv venv + 模型下载)
- [x] 验证 inference (14 次迭代修复)
- [x] PhaseTimer 适配 (自定义 register_profiling_hooks)
- [x] Profiling 运行 (完整 E/C/A)
- [x] 分析 + 对比报告

## Technical Notes

- lingbotvla 使用旧版 `lerobot.common.policies` API，需要 compat shim
- PI0Config 不接受 lingbotvla 的额外字段，用 dataclass field 过滤 + setattr
- `sample_actions()` 原始代码直接调用 `.forward()` 绕过 hooks — 需要 patch 为 `__call__`
- PyTorch 2.8 flex attention 与 lingbotvla 不兼容 — 使用 `eager` attention
- lingbotvla 的 `attention_mask` 是 Long 类型，eager attention 需要 `.bool()` patch
