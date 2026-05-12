---
title: "Rhoda DVA + Generalist GEN-1 对比精读 — 两条 'Post-VLA' 路线"
sources:
  - https://www.rhoda.ai/research/direct-video-action (2026-03, Rhoda AI Team)
  - https://generalistai.com/blog/apr-02-2026-GEN-1 (2026-04-02, Generalist AI Team)
  - https://generalistai.com/blog/apr-07-2026-beyond-world-models (2026-04-07/pub 05-05, Pete Florence)
date_saved: 2026-05-11
tags: [DVA, WAM, GEN-1, from-scratch, inverse-dynamics, leapfrog-inference, harmonic-reasoning, post-VLA]
related:
  - survey/papers/industrial-wam-landscape-2026.md (之前 homepage 级覆盖, 现在升级)
  - survey/papers/pi-series-evolution.md (GEN-1 初覆盖 §5)
  - survey/papers/cosmos-policy-deep-dive.md (unified video-policy 第三条路)
  - survey/papers/uva-cosmos-policy-comparative.md (decoupled vs monolithic)
---

# Rhoda DVA + Generalist GEN-1 — 两条 "Post-VLA" 路线精读

## 0. 为什么放在一起读

2026 Q1，Rhoda 和 Generalist 几乎同时发布了各自的技术博客，**都明确拒绝 "VLA" 标签**，但走的路线截然不同：

| 维度 | Rhoda DVA | Generalist GEN-1 |
|------|-----------|-------------------|
| **核心范式** | 纯 WAM — video prediction → IDM → actions | 从零训练 multimodal model — "不是 VLA 也不是 WM" |
| **推理时有 video generation？** | ✅ 完整 video denoising (claim: first) | ❌ 无 video generation，直接出 action |
| **预训练数据** | Web video (未公开规模) | 500K+ hr physical interaction (穿戴设备, 非 teleop) |
| **Task-specific data** | 10-20 hr robot data | ~1 hr robot data |
| **训练起点** | Causal video model from scratch | Multimodal model from scratch (99% params) |
| **自我定位** | "Direct Video-Action Model" | "既不是 VLA 也不是 WM" → "system" |
| **inference 创新** | Leapfrog Inference (overlap predict + execute) | Harmonic Reasoning (未披露细节) + custom paged attention |
| **部署规模** | 2 tasks, 1.5hr / 160min 无干预 | 6 tasks, 99% 成功率, 1800+ 连续 trials |
| **Hz / latency** | **未报** | **未报** (但 claim "3x faster than SOTA") |
| **融资** | $450M Series A, $1.7B 估值 | 未公开 (Pete Florence 联创 → Google → Generalist) |

**共同点**: 都从零训练 (不 fine-tune 现有 VLM)、都极度 data-efficient、都已部署到真实工业场景、都**不报 Hz 数字**。

---

## 1. Rhoda DVA — 技术架构详解

### 1.1 三件套流水线

```
Camera feed ─┬→ [Causal Video Model] → predicted future frames
             │                           │
             │                           ▼
             │                    [Inverse Dynamics Model] → actions
             │                           │
             └──── (action conditioning) ◄┘
                         ▲
                    Leapfrog loop
```

**Video Model**: 因果 (causal) video DiT, 从零预训练 (不从 bidirectional 蒸馏)。明确对标并批评了两类替代方案：
- Existing causal methods: 编码全序列但只在末尾几帧计算 loss → 训练低效
- Diffusion Forcing (Chen et al. 2024): 每帧独立加噪到随机 level → 长 context 退化

**Context Amortization** (核心创新): 在序列的**每个位置**预测未来帧 (类比 LLM next-token prediction at every position)。好处:
1. 训练 mask = 推理 mask (noise-free context → predict future) — 无 train/infer mismatch
2. loss 在更多帧上计算 → 训练效率更高
3. 可训练数百帧 context

推理时: KV-cache 复用已编码 context, 只在新帧到达时增量编码。

### 1.2 Inverse Dynamics Model (IDM)

- **非因果** (non-causal): 给定 predicted video → 反推 end-effector motion
- 极小模型, ~10 hr data 够训
- 甚至可以用 **random motions** 训练 (不需高质量 demonstrations)
- 跨 task 复用: 同一 embodiment type 的不同 task 共享 IDM

**关键 insight**: decision-making 已由 video model 完成 → IDM 只做 "translation"，不需要建模 complex behavior。

### 1.3 Leapfrog Inference

```
time ──────────────────────────────────────────►
      ┌──────── predict ────────┐
      │  (covers next latency)  │
      ▼                         │
 ┌─execute prev─┐  ┌─execute curr─┐  ┌─execute next─┐
 └──────────────┘  └──────────────┘  └──────────────┘
                   ┌──────── predict ────────┐
                   │  (covers next latency)  │
                   ▼                         │
```

- 预测足够远的未来 → overlap model inference & action execution
- **Action conditioning**: 每次 prediction 以**当前正在执行的 action**为条件 → 保证轨迹连续 (避免 jerk/oscillation)
- 本质是 **latency-hiding 策略** — 和 GPU pipeline parallelism 异曲同工

### 1.4 部署 Demo

| Task | Robot Data | Duration | 特点 |
|------|-----------|----------|------|
| Decanting (拆箱分拣) | 11 hr | 1.5 hr 无干预 | Bimanual, 10kg, deformable objects |
| Container Breakdown | 17 hr | 160 min 无干预 | 50 lbs, partial observability |
| Shell Game | — | — | 长 context memory demo (数百帧追踪) |
| Returns Processing | — | — | Long-horizon, end-to-end without subtask scaffolding |
| One-shot Demo Following | — | — | In-context learning from single human demo |
| One-shot Drawing | — | — | Stroke order preservation |

### 1.5 引用网络 (对 vlla landscape 有用的部分)

Rhoda 把自己放在 4 个 bucket 的交叉点，并明确标注了每个 bucket 的代表：

| Bucket | 代表 | Rhoda 评价 |
|--------|------|-----------|
| Video synthesis for robot data | GR00T N1, DreamGen, UniSim | 不是 DVA (video 用于 sim/data, 不做 policy) |
| Video model fine-tuned to predict actions | GR-2, UVA, Cosmos Policy, DreamZero | "fine-tuned" — Rhoda 是 "from scratch" |
| Open-loop video model control | UniPi, PLEX, 1X WM, TesserAct | "open-loop" — Rhoda 是 "closed-loop" |
| Closed-loop distilled from non-causal | Video Prediction Policy, mimic-video, LingBot VA | "distilled from non-causal" — Rhoda 是 "native causal" |

**Claim: first to (1) pre-train causal video model from scratch AND (2) do full video denoising in real-time closed-loop.**

---

## 2. Generalist GEN-1 — 技术架构详解

### 2.1 "99% Params From Scratch"

GEN-1 不基于任何现有 VLM/Video model — 全部从零训练。这和整个 VLA 领域的主流做法 (基于 PaliGemma / Qwen-VL / LLaMA 等 fine-tune) 直接对立。

预训练数据: **500K+ 小时** physical interaction data，采自低成本穿戴设备 (非 teleop)。**预训练零 robot data** → fine-tune 时才第一次接触 robot embodiment。

### 2.2 Mastery = Reliability + Speed + Improvisation

GEN-1 定义的评估框架:

| 维度 | GEN-1 表现 | 对比 |
|------|-----------|------|
| **Reliability** | 6 tasks 平均 99% (1 hr data) | GEN-0: 64%, from-scratch: 19% |
| **Speed** | Box fold 12.1s, Phone pack 15.5s | π0/GEN-0: ~34s (box fold), ~3x faster |
| **Improvisation** | 自发 bimanual regrasp, extrinsic dexterity | 超出训练分布的恢复行为 |

### 2.3 Scaling Laws → Commercial Viability

```
GEN-0 (Nov 2025): Scaling laws exist in robotics (64% avg)
  │
  ▼  +data +compute +algorithmic advances
GEN-1 (Apr 2026): Commercial viability threshold (99% avg)
```

类比 LLM: GPT-2 → GPT-3 跨越。每一代 unlock 更复杂的 task set。

### 2.4 系统组件 (未完全披露)

GEN-1 blog 提到但未详述的系统组件:
- **Harmonic Reasoning**: 新 inference-time technique → 加速 + 精度 (名称暗示频域/周期分解?)
- **Custom paged attention kernels**: 为 real-time inference 定制 (PagedAttention variant?)
- **Post-training techniques**: RL + multimodal human guidance
- **Learning from experience (RL)**: 速度提升的关键组件之一

### 2.5 "Beyond World Models" — Pete Florence 方法论宣言

核心论点 (3 条):

**1. Goals > Labels**: "goal-driven > idea-driven" (引用 Schulman)。WM 是 2026 的 bandwagon, VLA 是 2023-2025 的 bandwagon。Generalist 共同发明了 VLA (RT-2) 和 robotics WM (VLP)，但刻意不用任何标签。

**2. "And" not "Or"**: 不是 VLA 或 WM 的选择题 → "how much of each" → "objectives & constraints"。Chinchilla 是这类思维的典范。GEN-1 **已经 1年+ 实验 combining VLA + WM + beyond**。

**3. Supply side will change**: "数据不够" 是暂时约束。VLM pretrain 是 data 不够时的拐杖 (crutch)。500K+ hr 后, 拐杖还需要吗？

**隐含攻击目标**:
- π0/OpenVLA 路线 (fine-tune 现有 VLM) → "crutch"
- 1X/Rhoda 路线 (WM bandwagon) → "idea-driven"
- 整个 "VLA vs WM" 辩论 → "limiting 'or' question"

---

## 3. 对比分析 — 三层 "Post-VLA" 光谱

2026 Q1 后，VLA landscape 不再是 "VLA vs WAM" 的二分，而是三条技术路线：

```
Pure WAM ◄──────────────────── Hybrid ──────────────────────► Pure Action Model
(runtime video gen)                                      (no video gen at inference)

Rhoda DVA          Cosmos Policy / UVA          Generalist GEN-1
1X WM              DreamZero                    π0.7 (pipeline)
LingBot-VA         Fast-WAM                     OpenVLA-OFT

video denoise      video+action joint           direct action output
  秒级 A              百ms级 A                      ms-十ms级 A
```

### 3.1 Latency 含义 (for our profiling)

| 路线 | Action latency driver | 我们的实测 | 对应 Hz |
|------|----------------------|-----------|---------|
| Pure WAM (Rhoda/1X) | Video denoising (steps × per-step) | exp04b: V=697ms, A=1708ms → 0.4Hz | <1 Hz raw |
| Hybrid (Cosmos/Fast-WAM) | DiT denoising (fewer steps) | exp09a: 342ms (1-step) | 2-5 Hz |
| Pure Action (GEN-1/π0) | Action head forward | exp07a: 164ms (300M Expert) | 5-15 Hz |

**Rhoda 的 Leapfrog Inference 和 Generalist 的 Harmonic Reasoning 都是 latency-hiding 策略** — 它们不降低单次推理延迟，而是通过 overlap/pipeline 让有效控制频率高于 raw inference Hz。这正是 system-level optimization 的领域。

### 3.2 Data Efficiency 对比

| 模型 | Task-specific data | Pretrain data | Success |
|------|-------------------|---------------|---------|
| Rhoda DVA | 10-20 hr | Web video (未公开) | 1.5hr+ 无干预 |
| Generalist GEN-1 | ~1 hr | 500K+ hr (穿戴设备) | 99% on 6 tasks |
| π0 | "large-scale teleop" | VLM pretrain (PaliGemma) | 参考 |
| OpenVLA-OFT | OXE 970K eps | VLM pretrain (Prismatic) | 97.1% LIBERO |
| Cosmos Policy | per-task episodes | Video DiT (Cosmos-Predict2) | 98.5% LIBERO |

GEN-1 的 1 hr data 如果成立, 是数量级碾压。但注意: **预训练数据的采集成本被隐藏了** — 500K hr 穿戴设备数据不是免费的。

### 3.3 "从零训练" 的隐含信息

Rhoda 和 Generalist **都**选择从零训练 (不基于现有 VLM/Video model)。这不是巧合:

1. **完全控制架构** — 不受 VLM 架构约束 (e.g., Qwen2-VL 的 ViT resolution, LLaMA 的 context length)
2. **避免 domain gap** — VLM pretrain 的 distribution (internet text+image) 和 robot control 差距大
3. **IP 护城河** — 从零训练的模型不受上游 license 约束

代价: 极高的 pretrain compute + 数据采集投入。**这两家都是 well-funded startup** ($450M / undisclosed but ex-Google scale)。学术界很难复制。

---

## 4. 对 vlla 研究的具体影响

### 4.1 更新 industrial-wam-landscape 分类

之前写 landscape 时 Rhoda 条目说 "无技术 blog、research page 404"。现在 DVA blog 填补了这个缺口。需要更新:
- ✅ 架构: Causal Video Model + IDM + Leapfrog Inference (三件套)
- ✅ 训练: Context Amortization (from scratch, 非蒸馏)
- ✅ 部署: 2 industrial tasks, multi-hour autonomous operation
- ❌ 仍未报: model size, Hz, latency, 训练数据规模

### 4.2 Profiling gap 更加明确

**三家都不报 Hz:**
- Rhoda: 只说 "multiple times per second" (closed-loop), 零数字
- Generalist: 只说 "3x faster task completion" (speed of task, not inference Hz)
- π0.7: 不公开 inference spec

**我们的 profiling 数据 (exp04b/07a/09a) 是目前公开的几乎唯一 E/C/A breakdown**。这篇博客进一步确认了这个 gap 的存在。

### 4.3 Leapfrog Inference → serving systems 方向

Leapfrog Inference 是 **system-level latency hiding**:
- 预测 horizon > inference latency → overlap
- Action conditioning 保证连续性
- 本质等价于 DistServe 的 PD disaggregation → 不同的 "阶段" 可以 overlap/pipeline

这和我们之前 EPDA spec 的 motivation 完全一致: serving 层的优化不是只加速单次 forward, 而是**重排各阶段的执行拓扑**。

### 4.4 Harmonic Reasoning — 待关注

Generalist 只提了名字没说内容。从名称推测可能涉及:
- 频域分解 (harmonic analysis → 不同频率的 action 分开处理?)
- 或 inference-time search/planning (类 MCTS?)
- 或 multi-resolution prediction (类 π0.7 的 high-level + low-level?)

**如果后续有论文披露, 这可能是 inference-time efficiency 的新方向。**

---

## 5. 关键引语 (逐字)

### Rhoda DVA
> *"To the best of our knowledge, our model is the first to **pre-train a causal video model from scratch** and also the first to **perform full video denoising during real-time closed-loop robot control**."*

> *"Non-causal video-to-action translation is a much more constrained problem. The complex decision making has already been handled at the video generation stage."*

> *"Each video prediction is long enough to cover the next prediction's latency."* (Leapfrog)

### Generalist — GEN-1
> *"While we may call GEN-1 a model, it is even more accurate to refer to GEN-1 as a **system**."*

> *"In GEN-1, approximately **99% of the parameters are trained from scratch**."*

> *"GEN-1 can achieve comparable performance to GEN-0 with **10x less task-specific data**."*

> *"Without pretraining: 19%. GEN-0: 64%. GEN-1: **99%**."*

### Generalist — Beyond World Models
> *"World models are having their moment in early 2026. VLAs had theirs from 2023 to 2025. **Bandwagons are part of the nature of academic research.**"*

> *"We co-invented VLAs, have been publishing on world models in robotics since 2023 ... So why no label?"*

> *"What's after the crutch? Will you still want the crutch?"* (关于 VLM pretrain)

> *"For over a year, we have been experimenting with **combining ideas from across what you might call VLAs, world models, and beyond**."*

---

## 6. 引用网络 (合并)

### Rhoda DVA 引用 (22 refs)
- [1] GR00T N1 (2503.14734), [2] DreamGen (2505.12705), [3] UniSim (2310.06114)
- [4-12] Video model → action 系列: GR-2, UVA, Cosmos Policy, DreamZero, PAWM, LAPA, VideoGen=Policy 等
- [13-18] Open-loop WM: UniPi, PLEX, VideoWorld, This&That, TesserAct, **1X WM**
- [19-21] Distilled closed-loop: Video Prediction Policy, mimic-video, **LingBot VA**
- [22] Diffusion Forcing (2407.01392)

### Generalist GEN-1 引用 (13 refs)
- [1] GEN-0 scaling laws (prior), [2] GPT scaling (Kaplan 2020)
- [5] PaLM-E (2303.00189), [6] RT-2 / VLA (Brohan 2023), [7] VLP (Du 2023)
- [8] Chinchilla (Hoffmann 2022), [10-12] GEN-0/π0/π*0.6 速度对比

---

*2026-05-11. 三篇 blog 全文读取 (web-fetcher), 与 industrial-wam-landscape + pi-series-evolution + cosmos-policy-deep-dive 交叉参照。*
