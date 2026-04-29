---
title: Cosmos Policy — Unified Video-Policy via Latent Frame Injection
authors: NVIDIA Cosmos Team + Stanford
venue: ICLR 2026 (submitted)
arxiv: (TBD — release via research.nvidia.com/labs/cosmos-lab/cosmos-policy)
code: github.com/nvlabs/cosmos-policy
checkpoints: hf.co/collections/nvidia/cosmos-policy
date_saved: 2026-04-28
tags: [VLA, WAM, video-policy, DiT, FastVideo, candidate-A, latent-frame-injection]
---

# Cosmos Policy — Deep Dive

## Why this paper matters for vlla

Cosmos Policy 在 vlla landscape 里开辟了第三个品类: **"unified video-policy"** — 既不是纯 VLA (PI/Helix/OpenVLA 那条线) 也不是 "WM + 独立 policy head" (1XWM/Rhoda/DreamZero 那条线), 而是**用同一个 video DiT 同时承担 imagination 和 action generation**。通过 latent frame injection 把 action 直接编码进 video latent 序列, Cosmos-Predict2-2B 作为骨干在 LIBERO 刷到 98.5%, RoboCasa 67.1% — 两个 benchmark 的 SOTA, 而且用了单阶段 fine-tune (不是 DreamZero 那种 zero-shot LAM 复杂堆栈)。

对 vlla "Fast VLA first" 的 P0 而言, Cosmos Policy 是 **candidate A (Action DiT 加速) 的强候选 workload**: (1) **pure DiT homogeneous** — 整个 action path 就是一个 2B video DiT, 没有 LingBot-VA 那种 MoT/cross-attn/Action Expert 的异构分支, 所以 FastVideo 的 STA/VSA/step distillation 可以**天然无摩擦迁移**; (2) **direct-mode latency 是论文零报告的关键缺口** — 论文只在 planning 模式讨论 5s/chunk, 对 2Hz control 的 direct-mode 估算 ~500ms/chunk 是我们能抢先填的 profiling gap; (3) **SOTA quality floor** 让加速结果可信 — 如果能在保持 LIBERO 98% 的前提下把 direct mode 压到 10Hz+, 这是个干净的 FastVideo-for-robotics 叙事。

## Methodology Skeleton

**One-sentence problem statement**: 把机器人 policy 从"单独 action head"升级为"video DiT 同时生成未来帧 + action", 用 latent frame injection 让 action 和 video 共享同一个 denoising trajectory。

**Core architecture**:
- **Backbone**: Cosmos-Predict2-2B (NVIDIA 开源 video DiT, 2B params, pure transformer, flow matching)
- **Input**: 当前观测帧 → VAE encode → latent frame; 语言指令 → text encoder → condition
- **Action encoding**: action chunk (H steps × D dims) → 线性投影到 latent frame 空间 → 作为额外 latent frame 注入 video token 序列
- **Output**: denoised video latents (含未来预测帧 + action latent) → VAE decode 出视频 + 反投影出 action

**Key equations / design choices**:
- Flow matching loss on **joint latent sequence** (video frames + action latent): `L = E[||v_θ(z_t, t, c) - (z_1 - z_0)||²]` 其中 z 包含 video + action latent, c 是 text + current observation
- **Latent frame injection**: action 被当作"第 N+1 帧"参与 denoising, 不引入新的 head/branch — 这是 homogeneity 的关键
- **两种推理模式**:
  1. **Planning mode**: full denoise (50 steps flow matching), 产出完整 5s 视频 + action chunk, 报告 5s wall-clock per chunk
  2. **Direct mode**: 跳过 video 帧 decode, 只取 action latent, 但 denoise 过程本身不变 (2B DiT × 50 steps)

**Algorithm flow (inference)**:
1. Encode current obs (VAE encode) → z_obs
2. Sample noise z_T, 拼接 [z_obs, z_T_video_frames, z_T_action]
3. For t = T → 0: z_{t-1} = z_t - v_θ(z_t, t, text)·Δt (Euler / 50 steps)
4. Extract action latent → 反投影到 action chunk (H steps)
5. (Optional) VAE decode video latents → 可视化未来帧

## Assumptions & Limitations

**Architectural assumptions**:
- **[Restrictive]** Action space 能被线性投影到 video latent 空间而不损失精度 — 对连续低维 action (7-DoF arm) 合理, 对 discrete / high-DoF (dexterous hand) 未验证
- **[Restrictive]** 单一 video DiT 同时优化 video prediction 和 action generation 不会出现"哪个先收敛挤占另一个"的 capacity 竞争 — 论文用 2B 大模型规避, 但小模型设定下未知
- **[Standard]** VAE encoder 对机器人第一视角图像足够 informative — Cosmos-Predict2 本来就在大规模视频上预训练, 可信

**Training assumptions**:
- **[Restrictive]** 单阶段 fine-tune 足够 — 对比 DreamZero 需要 LAM 蒸馏多阶段, 这是重要简化但依赖 2B 骨干的泛化能力
- **[Standard]** Flow matching 比 diffusion 更稳 — 符合 2025-2026 年领域共识

**Inference-time limitations (论文未直接承认)**:
- **Direct mode 仍然需要完整 50-step denoise** — 即使不 decode video, DiT forward 成本不变, 这是我们估算 ~500ms/chunk 的来源 (2B DiT × 50 step, 类比 exp06a NitroGen 174M = 7.2ms/step → 2B ≈ 80ms/step × 50 = 4000ms, 但 Cosmos-Predict2 用过 caching 和 flash attn, 实际可能压到 400-600ms)
- **Chunk-level latency only** — 论文只报 chunk 粒度, 没拆 encode/denoise/decode, 这是 profiling gap
- **No per-step breakdown** — 50 steps 每步是否均匀? DiT layer activation variance? 论文无数据

**Cross-ref with vlla exp summary**:
- 对比 **exp07a Pi-Zero** (total 200ms, action 82%): Cosmos Policy direct mode 估算 ~500ms → **2.5x 慢于 Pi-Zero**, 但 quality 更高 (LIBERO 98.5% vs Pi-Zero 未直接可比)
- 对比 **exp04a Fast-WAM** (total 407ms @10step, action 89%): 和 Cosmos Policy 量级相当, 但 Fast-WAM 已经是 10-step 省版, Cosmos 50-step
- 对比 **exp06a NitroGen 174M DiT** (7.2ms/step): per-step scaling 预期 2B ≈ 80-120ms/step, 50-step ~4-6s — **除非有 caching**

## Bridge Analysis

**Where it sits in vlla landscape.md**:
- **Industrial WAM landscape** 归类: runtime video-gen 阵营 (和 1XWM/Rhoda/DreamZero 同类), 但 **uniquely homogeneous** — 没有独立 policy head
- **VLA/WAM efficiency survey** 归类: 是 "video DiT 加速"的**理想 target**, 因为没有异构分支需要特殊处理
- **π series evolution** 对照: π0.7 走 "High-Level + WM + VLA + Action Expert 四件套"的 pipeline 路径, Cosmos Policy 反向走"单 DiT 做所有事"的 monolithic 路径 — **两条正交的 system 路线**

**Borrowable points (to vlla)**:
1. **Latent frame injection** 本身是优雅的 design pattern — 将来如果自己做 action head, 可以借用这个"action as extra latent frame"的 formulation 避免异构
2. **Cosmos-Predict2-2B 开源** — 直接可用的 2B video DiT 骨干, 比训练自己的 backbone 省几百 GPU-days
3. **LIBERO/RoboCasa eval setup** — 官方脚本可直接接入 exp07 profiling framework

**Differences to be careful about (对照 vlla 已有架构)**:
- **BaseVLAController 目前假设 encode/context/action 三阶段** (Pi-Zero, LingBot-VA 都是) — Cosmos Policy 是 "encode → 50-step joint denoise → extract action", 没有独立 context 阶段, **需要 PhaseTimer 改造**
- **Homogeneous DiT ≠ 简单** — 虽然 architecture 同质, 但 action latent 和 video latent 在同一序列中 denoise, attention 模式和 pure video DiT 不同 (有 action-aware spatial bias?), **exp05 类 attention analysis 需要重做**

**exp09 候选提议 (从 candidate A 细化)**:
> **exp09a: Cosmos Policy 1-step distillation + best-of-N value filtering**
> - Baseline: Cosmos Policy 50-step direct mode, 测 LIBERO 98.5% / direct latency
> - Step 1: FastVideo-style step distillation (50→4→1), 记 quality 掉点曲线
> - Step 2: best-of-N (N=4-8) sampling + learned value head 做 action selection, 补回 quality
> - 目标: **LIBERO 97%+ (掉 1.5 pt 可接受) @ 10Hz+ direct mode** (5x 加速)
> - 科研 story: "video DiT 的加速术能无损迁移到 policy DiT 吗?"

## Open Questions

1. ~~**Direct mode 到底多慢?**~~ **已回答 (exp09a, full sweep)**:
   - **Action-only 5-step: 659ms / 1.5Hz** (parallel_gen=False)
   - **Full (action+future+value): 1363ms / 0.73Hz** (parallel_gen=True, +698ms overhead)
   - **Linear fit (R²=0.9975)**: fixed cost **265ms** (VAE encode + extract) + **76.8ms/step** (DiT denoise)
   - **1-step distilled 预测: 342ms / 2.9Hz** (固定成本占主导)
   - 之前估算 "~500ms" 和 "272ms/step" 都不准确: 前者忽略了 parallel_gen 开销, 后者是 total/steps 而非斜率
2. ~~**5 steps 可压到多少?**~~ **部分回答**: step sweep 显示 1-step 和 2-step 几乎一样 (378ms vs 381ms), 说明 fixed cost 占主导。1-step distillation → 342ms/2.9Hz 是可达的; 但真实 quality 损失需要 LIBERO eval 验证。
3. **Action latent 维度多少?** 对 action head 线性投影的 rank 有影响, 代码应该有
4. ~~**Cosmos-Predict2-2B 在 RTX 5880 Ada 48GB 上显存够吗?**~~ **已回答 (exp09a)**: Peak VRAM = 8816 MB. 48GB 绰绰有余。
5. **和 LingBot-VA 的 full WAM (697ms video + 1708ms action = 2518ms) 比, Cosmos Policy 是否真的更快?** 需要同机测
6. **Attention pattern 是不是和 exp05 Gini 崩塌一致 (VLA fine-tune 后 sink 消失)?** 未验证
7. **Latent frame injection 是否适用于 dexterous hand (高维 action)?** 论文未覆盖
