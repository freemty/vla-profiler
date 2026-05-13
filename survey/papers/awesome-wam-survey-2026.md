---
title: "Awesome-WAM Survey (Fudan OpenMOSS) — WAM 领域首篇系统性 Survey"
authors: Siyin Wang, Junhao Shi, Zhaoyang Fu et al. (Fudan + NUS)
arxiv: "2605.12090"
date_saved: 2026-05-13
tags: [WAM, survey, taxonomy, cascaded, joint, diffusion, autoregressive, profiling-gap]
related:
  - survey/papers/va-world-models.md (我们自己的 VA+WAM survey)
  - survey/papers/industrial-wam-landscape-2026.md (工业 WAM 分类)
  - survey/papers/robotics-fm-landscape-2026-q2.md (五家头部横评)
  - survey/papers/cosmos-policy-deep-dive.md (Joint/Unified DiT 代表)
  - survey/papers/rhoda-dva-generalist-gen1-deep-dive.md (Cascaded/Explicit 代表)
---

# Awesome-WAM Survey — WAM 领域首篇系统性 Survey

## Why this matters

arXiv:2605.12090 (2026-05-12), 复旦 OpenMOSS + NUS。**64 篇论文**，第一次给 WAM 一个硬定义 + 完整分类体系。之前 WAM 这个词在各论文里含义模糊 (有时指 world model, 有时指 video-action model)，这篇 survey 把它钉死了。

---

## 1. WAM 硬定义

**World Action Model**: 模型必须同时做 forward state prediction + action generation — 目标分布是 `p(o', a | o, l)`。

| 概念 | 目标分布 | 是否 WAM? |
|------|---------|----------|
| VLA | `p(a \| o, l)` — 无预测 | ❌ reactive |
| WM | `p(o' \| o, a)` — 无 action output | ❌ predictive but not executable |
| WAM | `p(o', a \| o, l)` — 预测 + action 联合 | ✅ |

**边界条件**: future-state prediction 必须是 policy 的一部分，不能只是 auxiliary loss 或外部 simulator。

---

## 2. 两大架构族

### Cascaded vs Joint 的判定标准

**核心区分: WM 和 Action Model 能不能拆开独立训练/替换？**

| | Cascaded | Joint |
|--|---------|-------|
| **训练** | WM 和 Action Model 分别独立训练 | Video 分支和 Action 分支联合训练，共享梯度 |
| **耦合** | 推理时串联，但模块可独立替换 | 架构内部有 cross-attn / shared latent / 同一 DiT 耦合 |
| **判定** | 能拆 = Cascaded | 不能拆 = Joint |

**易混淆的例子**:
- **π0.7 → Cascaded**: WM 单独生成 goal image → VLA + Action Expert 独立消费。各组件分开训练，可独立替换。
- **Rhoda DVA → Cascaded**: Video Model 从零单独预训练 → IDM 单独训练 (~10hr, 甚至 random motions)。两个模型完全独立。
- **DreamZero → Joint/Unified DiT**: Video frames 和 action tokens 在同一个 DiT 的同一次 denoising 里联合生成，同一个 forward pass。
- **LingBot-VA → Joint/Multi-DiT**: Video DiT 和 Action DiT 是两个 head，但通过 cross-attention 耦合联合训练，不能独立替换。
- **Fast-WAM → Joint/Multi-DiT**: 同理，imagination 和 action 通过 hidden state 耦合联合训练。

### 2.1 Cascaded WAM: WM → Action Model (两阶段)

WM 先生成未来表征 → **独立**的 action model 解码为 executable commands。两个模块分别训练，可独立替换。

**Explicit (pixel-space carriers)**:
- Learned action extraction (IDM):
  - UniPi (NeurIPS 2023), VLP (ICLR 2024), Veo-Act, VAG, **pi0.7** (goal image → VLA), **Rhoda DVA** (video → IDM)
  - 优点: interpretable (可以看 video/image), 易 debug
  - 缺点: video denoising 延迟高, IDM 引入额外 error
- Geometric extraction (flow/depth/4D):
  - AVDC (ICLR 2024), Im2Flow2Act (CoRL 2024), Dream2Flow, NovaFlow, 4DGen
  - 优点: 几何信号 → action 的映射更 compact

**Implicit (latent representations)**:
- VPP (ICML 2025), mimic-video, LAPA (ICLR 2025), villa-X (ICLR 2026), S-VAM, MWM, OmniVTA
- 优点: 不需要 full pixel generation → **可能更快**
- 缺点: 不可解释, latent 质量难评估

### 2.2 Joint WAM: 未来 + Action 在同一模型联合生成

Video/latent 预测和 action 生成**联合训练**，架构内部耦合，不可独立拆分。

**Autoregressive Generation**:
- Explicit-Decoupled: GR-1 (ICLR 2024), GR-2 (ByteDance)
- Unified-Discrete: CoT-VLA (CVPR 2025), WorldVLA (阿里), F1 (InternRobotics), RynnVLA-002 (阿里)
- Predictive-Latent: VLA-JEPA

**Diffusion-based Generation**:
- Unified DiT (single denoising engine, video+action 同 latent space):
  - **Cosmos Policy**, **DreamZero**, X-WAM, GigaWorld-Policy, PAD, VideoVLA, UWM, FLARE
- Multi-DiT (separate denoisers, coupled via cross-attn or hidden state):
  - Cross-Attention: **LingBot-VA**, DUST, Motus, CoVAR, LDA-1B, AdaWorldPolicy, AIM, MotuBrain, DexWorldModel
  - Hidden State: **Fast-WAM**, DiT4DiT, WAV
  - Shared Representation: PhysGen

---

## 3. 和我们 profiling 的映射

| Survey 分类 | 我们之前的叫法 | 已测模型 | Latency 特征 |
|-------------|-------------|---------|-------------|
| Cascaded/Explicit/Learned | "Pure WAM" | LingBot-VA (exp04b: 2518ms/0.4Hz), Fast-WAM (exp04c: 257ms/3.9Hz) | Video denoising 主导, 秒级 |
| Cascaded/Explicit/Learned | pi0.7 (goal image) | π0 (exp07a: 200ms/5Hz) | Action Expert 主导 (82%) |
| Joint/Unified DiT | "Unified video-policy" | Cosmos Policy (exp09a: 342ms/2.9Hz 1-step) | DiT denoising, 百ms级 |
| Joint/Multi-DiT/Cross-Attn | — | **未测** | 预计类 LingBot-VA (MoT cross-attn overhead) |
| Cascaded/Implicit | — | **未测** | 预计更快 (无 pixel generation) |
| Joint/AR | — | **未测** | 预计类 VLA (autoregressive token generation) |

### 覆盖率

- ✅ 已测: 4/6 大类 (Cascaded Explicit + Joint Unified DiT)
- ❌ 缺失: Joint/Multi-DiT 和 Cascaded/Implicit — 这两类在 2026 年论文数量爆发
- **最大 gap**: Cascaded/Implicit 路线如果延迟确实更低, 可能是 profiling 的高价值目标

---

## 4. Open Challenges (survey 列出的 6 条)

| # | Challenge | 对我们的关系 |
|---|-----------|-------------|
| 1 | Architectural coupling — 缺乏匹配条件下的系统对比 | 我们的 profiling 是唯一做跨模型 E/C/A breakdown 的 |
| 2 | Multimodal physical state — RGB 不够, 缺 tactile/force | Genesis GENE-26.5 走这条路 (触觉 + 力反馈) |
| 3 | Data mixture design — 各数据源的边际贡献不清楚 | 和我们无直接关系 |
| 4 | Long-horizon planning — drift, compounding error | Rhoda 用 Leapfrog + action conditioning 缓解 |
| 5 | **Inference latency and efficiency** | **= 我们的核心方向** |
| 6 | Evaluation — 缺 joint metrics for causal consistency | 可能是 profiling 的下游应用 |

Survey 原文 Challenge 5:
> "Diffusion and autoregressive prediction remain too slow for many closed-loop settings without aggressive compression."

---

## 5. 数据生态 (survey 的数据分类)

| 数据源 | 特点 | 代表 |
|--------|------|------|
| Robot teleoperation | 高频对齐 state-action pairs, 低 sim2real gap | OXE, DROID, BridgeData |
| Portable human demo | UMI-style, 低成本多样性 | UMI, DROID, HumanPlus |
| Simulation | 可控物理, 可 scale, privileged info | Isaac Lab, Genesis World, ManiSkill |
| Human/ego-centric video | 海量但无 action label | Ego4D, Something-Something, YouTube |

**五家头部的数据策略正好覆盖了这四类**:
- Skild = Simulation 为主
- Rhoda = Human/ego-centric video 为主
- Generalist = Portable human demo (穿戴设备) 为主
- PI = Robot teleoperation 为主
- Genesis = 三层混合 (Glove + ego + 3rd-person)

---

## 6. Evaluation Protocols (survey 整理)

### World Modeling 评估
- Visual fidelity: PSNR, SSIM, LPIPS, DreamSim, FVD
- Physical commonsense: VideoPhy, PhyGenBench, Physics-IQ, WorldScore
- Action plausibility: WorldSimBench, IDM Turing Test

### Action Policy 评估
- General manipulation: LIBERO, RoboCasa, CALVIN, GemBench, Meta-World, RLBench, ManiSkill
- Bimanual/humanoid: RoboTwin, BiGym, HumanoidBench
- Mobile manipulation: ManipulaTHOR, HomeRobot, BEHAVIOR-1K
- Contact/deformation: SoftGym, PlasticineLab, TacSL
- Real-device: RoboArena, RoboChallenge, Maniparena

**注意: 没有任何 latency/throughput benchmark** — survey 自己在 Challenge 5 也承认这个 gap。

---

## 7. Paper List 亮点 (64 篇, 按时间线)

### 关键节点
- **2023**: UniPi (NeurIPS) 开创 video→IDM 范式
- **2024**: GR-1 (ICLR) + LAPA (ICLR) — joint autoregressive 起步
- **2025**: VPP, UWM, Fast-WAM, mimic-video, Motus — 爆发期
- **2026**: Cosmos Policy, DreamZero, pi0.7, X-WAM, GigaWorld-Policy, MotuBrain, MWM, S-VAM, OmniVTA — 几乎每周一篇

### 我们 survey 里已有深读但未出现在这篇 survey 里的
- **Rhoda DVA** — 未发 arXiv, 只有 blog (survey 不收)
- **Generalist GEN-1** — 同上, 只有 blog
- **Skild Brain** — 同上
- **Genesis GENE-26.5** — 同上

这进一步说明**工业头部的技术细节只在 blog 里**, 学术 survey 覆盖不到。我们的 `robotics-fm-landscape-2026-q2.md` 填补了这个 gap。

---

## 8. 对 vlla 项目的具体价值

1. **分类框架可以直接复用** — Cascaded vs Joint, Explicit vs Implicit, Learned vs Geometric 这套 taxonomy 比我们之前的 "Pure WAM / Hybrid / Pure Action" 更精细
2. **Paper list 作为 profiling 候选池** — 64 篇中有 code+model 的可以加入 exp11+ 的候选
3. **Evaluation 部分整理了所有 benchmark** — 可以用于我们的 Pareto frontier (latency × accuracy)
4. **Challenge 5 是我们的 citation target** — 如果我们发 paper, 这篇 survey 是 motivation section 的天然引用

---

*2026-05-13. Project page + GitHub README 全文读取 (web-fetcher), arXiv abstract 确认。*
