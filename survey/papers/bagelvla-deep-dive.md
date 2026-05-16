---
title: "BagelVLA 精读 — Interleaved Vision-Language-Action + Residual Flow Guidance"
sources:
  - https://arxiv.org/abs/2602.09849
date_saved: 2026-05-16
tags: [VLA, MoT, flow-matching, world-model, interleaved-planning, RFG, long-horizon]
related:
  - survey/papers/pi-series-evolution.md (π0.7 multi-component pipeline 对照)
  - survey/papers/cosmos-policy-deep-dive.md (unified latent denoising 对照)
  - survey/papers/awesome-wam-survey-2026.md (Joint WAM 分类)
---

# BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation

- **Authors:** Yucheng Hu, Jianke Zhang, et al. (Tsinghua + ByteDance Seed)
- **arXiv:** 2602.09849, Feb 2026

## Core Idea

单一 MoT 框架内交替执行 linguistic planning → visual forecasting → action generation。用 Residual Flow Guidance (RFG) 做单步 denoising 提取 predictive visual features，避免完整 video generation 的延迟。

## Architecture

| 组件 | 参数量 | 功能 | 骨干 |
|------|--------|------|------|
| Understanding Expert | 7B | 语言规划 (subtask) | Qwen2.5-LLM-7B |
| Generation Expert | 7B | 视觉预测 (keyframe) | Qwen2.5-LLM-7B |
| Action Expert | 2B | 动作生成 (flow matching) | Qwen2.5 (MLP 1/5) |

Total ~16B params, MoT 架构 (每次只有 1 个 expert active)。从 Bagel (ByteDance 统一理解+生成模型) 初始化。

## Residual Flow Guidance (RFG)

关键创新：将 keyframe 噪声初始化从 `N(0, I)` 改为 `N(v_t, I)` (当前帧为中心)。Generation Expert 只需学 residual change (动态区域)，单步 denoising 即可提取有意义的 predictive visual features。

三种 scheme 对比 (Calvin, A800):
| Scheme | Latency/chunk | Calvin ABC-D |
|--------|--------------|--------------|
| Complete Denoise (50+10 steps) | 6.04s | 2.48 |
| Joint Denoise (50 steps) | 2.90s | 2.04 |
| Single-step + RFG | **1.23s** | **3.60** |

## Inference Pipeline

1. Understanding Expert (7B AR) → subtask text
2. VAE encode 当前帧 + Gaussian noise → noisy keyframe
3. 统一序列: [views, text, noisy_keyframe, action_noise]
4. Generation Expert 处理 image token (单步), Action Expert 处理 action token (10 步 flow matching)
5. Action Expert attend to Generation Expert 第一步 KV-cache → predictive visual conditioning
6. 异步执行: KV cache 复用 → 72 Hz effective (chunk=48)

## Key Numbers

- **1.2s/chunk on RTX 5090** → 40 Hz effective (chunk=48), 72 Hz (async)
- Calvin ABC-D: **4.41** (vs Pi-Zero 3.65, UP-VLA 4.08)
- RoboTwin Clean: **75.26%** (vs Pi-Zero 46.42%)
- Real long-horizon stack: **73.3%** (vs Pi-Zero 40.0%)
- 没有 LIBERO 评测

## 在我们延迟分类中的位置

单次调用 1200ms，介于 Action DiT (200-407ms) 和 Full WAM (2518ms) 之间。40-72 Hz 声称靠 chunk=48 摊销，实际 re-planning 频率 ~0.83Hz。

## 与我们实验的桥接

1. **RFG = "world model as feature extractor"**: 不做完整 video gen，只借用 generation expert 第一步 denoising 作为 predictive visual conditioning。类似 Cosmos 的 latent frame injection，但用 7B 级 generation expert (远比 Cosmos 2B 重)
2. **MoT 避免 attention 破坏**: 与 exp05a 发现的 "VLA fine-tuning destroys VLM attention" 形成对照 — MoT 架构中各 expert 的 attention 独立
3. **方向 A 的实例**: RFG single-step = algorithmic reduction 的一种 (减少 denoise 步数到 1)
4. **大 chunk 摊销 vs reactivity**: 1.2s 内对环境变化是盲的，对 contact-rich/dynamic tasks 有风险

## Limitations

- 延迟拆解不透明 (text/RFG/action 各占多少未报)
- 没有 LIBERO 评测 (无法与 Cosmos 97.4% / Fast-WAM 94.5% 直接比较)
- 模型未开源 (截至 2026-05)
- 16B 参数量对边缘部署不友好
- 72 Hz 依赖异步 + 大 chunk，re-planning 频率仅 0.83Hz

## Profiling Value

中等优先级。如果开源，可以验证:
- RFG overhead = 1 步 7B generation forward (~150-250ms on RTX 5880)
- 三个 Expert 的精确延迟拆解
- MoT 架构的 KV cache 复用效率
