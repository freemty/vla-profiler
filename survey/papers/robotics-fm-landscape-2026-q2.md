---
title: "Robotics Foundation Model Landscape — 2026 Q1-Q2 五家头部横评"
date_saved: 2026-05-11
sources:
  - https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain (2025-07-29)
  - https://www.skild.ai/blogs/omni-bodied (2025-09-24)
  - https://www.skild.ai/blogs/learning-by-watching (2026-01-12)
  - https://www.skild.ai/blogs/reindustrial-revolution (2026-03-19)
  - https://www.rhoda.ai/research/direct-video-action (2026-03)
  - https://generalistai.com/blog/apr-02-2026-GEN-1 (2026-04-02)
  - https://generalistai.com/blog/apr-07-2026-beyond-world-models (2026-04-07, pub 2026-05-05)
  - https://www.genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level (2026-05-07)
  - survey/papers/pi-series-evolution.md (PI π 系列已有深读)
  - survey/papers/rhoda-dva-generalist-gen1-deep-dive.md (Rhoda+Generalist 已有深读)
tags: [landscape, foundation-model, VLA, WAM, from-scratch, system, dexterity, profiling-gap]
---

# Robotics Foundation Model Landscape — 五家头部横评 (2026 Q1-Q2)

## 0. 为什么写这篇

2026 Q1-Q2 出现了一个罕见的信息窗口：五家最 well-funded 的 robotics FM 公司几乎同时发布了详细技术博客。这些信息之前散落在 homepage 级一句话或完全缺失 (Skild blog 404, Rhoda research page 404)。现在足够做一次**完整的技术路线对比**。

关联:
- `survey/papers/rhoda-dva-generalist-gen1-deep-dive.md` — Rhoda DVA + Generalist GEN-1 双路线精读
- `survey/papers/pi-series-evolution.md` — PI π 系列 + GEN-1 初覆盖
- `survey/papers/industrial-wam-landscape-2026.md` — WAM vs VLA binary 分类

---

## 1. 总览矩阵

| 公司 | 旗舰模型 | 发布时间 | 路线 | 核心赌注 |
|------|---------|---------|------|---------|
| **Skild AI** | Skild Brain | 2025-07 (iterating) | Sim-pretrained hierarchical | 形态多样性 |
| **Rhoda AI** | DVA | 2026-03 | Causal video → IDM → action | 物理理解 (video) |
| **Generalist AI** | GEN-1 | 2026-04-02 | From-scratch multimodal | 训练范式 + 数据规模 |
| **Physical Intelligence** | π0.7 | 2026-04 | Pipeline (planner+WM+VLA+expert) | 系统分工 |
| **Genesis AI** | GENE-26.5 | 2026-05-07 | Flow matching + 自研 hand | 硬件精度 |

| 公司 | 融资 | 估值 | 创始团队 |
|------|------|------|---------|
| Skild | $300M A + $1.4B C | $14B | Deepak Pathak + Abhinav Gupta + Lerrel Pinto (CMU) |
| Rhoda | $450M A | $1.7B | Stealth → 2026-03 exit |
| Generalist | 未公开 | 未公开 | Pete Florence (Google → RT-2/PaLM-E co-inventor) |
| PI | $2.4B+ | $39B (报道) | Sergey Levine + Karol Hausman + Chelsea Finn (Google) |
| Genesis | 未公开 | 未公开 | 2026-05 首次公开 |

---

## 2. 架构对比

### 2.1 Skild Brain — Hierarchical Omni-Bodied

```
[High-level Policy]  (低频: manipulation/navigation 指令)
        │
        ▼
[Low-level Policy]   (高频: joint angles / motor torques)
```

- **Pretrain**: 100K+ 不同 morphology 的 sim robot + internet human video
- **Post-train**: <1 hr task-specific robot data
- **核心能力**: zero-shot 控制从未见过的 robot, in-context learning (从失败中学习)
- **关键批评**: 明确称 "fine-tune VLM + 1% robot data" 为 "Potemkin village"
- **部署**: ABB Robotics + Universal Robots + NVIDIA 合作, Foxconn Blackwell GPU 装配线
- **Sim**: Isaac Lab / Isaac Sim + NVIDIA Cosmos (数据增强)

### 2.2 Rhoda DVA — Causal Video → IDM → Action

```
Camera → [Causal Video Model] → predicted frames → [IDM] → actions
              ▲                                        │
              └──── action conditioning (Leapfrog) ────┘
```

- **Video Model**: 从零预训练 causal video DiT, Context Amortization 训练
- **IDM**: 极小模型, ~10 hr data, 甚至可以用 random motions 训练
- **Leapfrog Inference**: predict horizon > inference latency → overlap execute & predict
- **Claim**: first to pre-train causal video from scratch + first full video denoising in real-time closed-loop
- **部署**: Decanting (11hr, 1.5hr 无干预) + Container Breakdown (17hr, 160min 无干预)
- **深读**: `survey/papers/rhoda-dva-generalist-gen1-deep-dive.md`

### 2.3 Generalist GEN-1 — From-Scratch Multimodal System

```
[Multimodal Foundation Model]  (99% params from scratch)
   + Harmonic Reasoning (inference-time, 未披露)
   + Custom Paged Attention
   + Post-training (RL + multimodal human guidance)
```

- **Pretrain**: 500K+ hr physical interaction data (低成本穿戴设备, 非 teleop), 零 robot data
- **Post-train**: ~1 hr per task → 99% 成功率 (6 tasks)
- **Speed**: 3x faster than SOTA (box fold 12s vs π0/GEN-0 34s)
- **Mastery 框架**: Reliability (99%) × Speed (3x) × Improvisation (emergent recovery)
- **方法论宣言**: "Goals > Labels", VLM pretrain 是 "crutch", WM 是 "bandwagon"
- **深读**: `survey/papers/rhoda-dva-generalist-gen1-deep-dive.md`

### 2.4 PI π0.7 — Multi-Component Pipeline

```
[High-Level Planner] → [World Model (goal image)] → [VLA] → [Action Expert]
```

- **最新公开**: π0.7 用 4 件套 pipeline, 不再是单体 VLA
- **原始 π0**: PaliGemma 2B VLM + flow matching action head
- **关键转折**: 2026-04 同月承认 "model is a system" — 和 Generalist 不谋而合
- **部署**: 叠衣服、清桌子、装箱等家务/服务场景
- **深读**: `survey/papers/pi-series-evolution.md`

### 2.5 Genesis GENE-26.5 — Flow Matching + 自研 Hand

```
[Robotics-Native Foundation Model]
   Input: language + vision + proprioception + tactile + action (五模态)
   Method: flow matching joint distribution over trajectories
   Output: action trajectories
        │
        ▼
[Custom Control Middleware]  (3ms E2E latency, 500Hz, impedance control)
        │
        ▼
[Genesis Hand 1.0]  (20-DOF, 1:1 人手尺寸, soft-contact skin)
```

- **Pretrain**: Glove data + egocentric video + third-person video (200K+ hr)
- **Post-train**: <1 hr (<200 episodes for <20s skills)
- **核心创新**: hardware co-design — 自研 hand (20-DOF, 1:1 人手) + 自研 control (3ms) + 自研 sim (Genesis World)
- **Scaling laws**: open-loop (loss ∝ model size) + closed-loop (sim eval, zero sim training data)
- **评估**: Genesis World sim → 单数据点 = 200 setups × 150+ hr robot time
- **Demo**: 做饭 (4min, 20+ subtasks), 解魔方 (claim: first bimanual), 实验室移液 (mm级), 弹钢琴, 线束绑扎

---

## 3. 维度对比

### 3.1 训练数据

| 公司 | Pretrain 数据源 | Pretrain 规模 | Task-specific |
|------|---------------|-------------|---------------|
| Skild | Sim (100K morphology) + human video | 未公开 (trillions of trajectories 目标) | <1 hr |
| Rhoda | Web video | 未公开 | 10-20 hr |
| Generalist | 穿戴设备 physical interaction | 500K+ hr | ~1 hr |
| PI | VLM pretrain (PaliGemma) + teleop | large-scale teleop | large-scale |
| Genesis | Glove + ego + 3rd-person video | 200K+ hr | <1 hr |

**PI 是唯一仍依赖大规模 teleop 的**。其余四家都在压缩 task-specific data 到 <1hr。

### 3.2 模型起源

| 公司 | 基于现有 VLM? | 具体做法 |
|------|-------------|---------|
| Skild | ❌ | Sim pretrain from scratch, 明确批评 VLM 为 "Potemkin village" |
| Rhoda | ❌ | Causal video model from scratch (非蒸馏) |
| Generalist | ❌ | 99% params from scratch, 明确称 VLM 为 "crutch" |
| PI | ⚠️ | π0 基于 PaliGemma; π0.7 不清楚 |
| Genesis | ⚠️ | 可吸收 VLM/WM 作为 prior, 非必须 |

### 3.3 推理时 Video Generation

| 公司 | 推理时生成 video? | 影响 |
|------|-----------------|------|
| Skild | ❌ | 直接出 action (hierarchical) |
| Rhoda | ✅ 完整 video denoising | 秒级 latency, 需 Leapfrog hiding |
| Generalist | ❌ | 直接出 action |
| PI | ⚠️ WM 生成 goal image (非 full video) | 轻量 imagination |
| Genesis | ❌ | Flow matching 直接出 action trajectory |

### 3.4 硬件差异化

| 公司 | 自研硬件 | 控制延迟 |
|------|---------|---------|
| Skild | ❌ 用 OEM (ABB/UR) | 未报 |
| Rhoda | 未公开 | 未报 |
| Generalist | 自研 robot hands + data collection devices | 未报 |
| PI | 未公开 | 未报 |
| Genesis | ✅ Genesis Hand 1.0 (20-DOF) + control middleware | **3ms** E2E, 500Hz |

### 3.5 Demo Tasks 谱系

```
粗操作 (locomotion)          中等 (pick-place/fold)         精细 (dexterity)
├─ Skild: 跨形态走路         ├─ Generalist: 叠衣服 (99%)    ├─ Genesis: 打蛋/切菜
├─ Skild: 断腿/卡轮恢复     ├─ Generalist: 折纸箱 (99%)    ├─ Genesis: 双手解魔方
│                            ├─ Rhoda: 拆箱 (1.5hr)         ├─ Genesis: 移液 (mm级)
│                            ├─ Rhoda: 纸箱拆解 (160min)     ├─ Genesis: 弹钢琴
│                            ├─ PI: 叠衣/清桌/装箱           ├─ Genesis: 线束绑扎
│                            ├─ Skild: Blackwell 装配        │
│                            └─ Rhoda: 退货处理              │

长 horizon ─────────────────────────────────────────────────────►
  Skild 装配 (~min)  Generalist 循环 (1800x)  Rhoda 拆箱 (160min)  Genesis 做饭 (4min, 20+ subtasks)
```

---

## 4. 五个赌注

每家赌的 bottleneck 不同，解法也不同：

| 公司 | 赌什么是 bottleneck | 解法 | 反例/风险 |
|------|-------------------|------|----------|
| **Skild** | 形态多样性 (没有通用 body) | 100K morphology sim pretrain | Sim-to-real gap 可能限制上限 |
| **Rhoda** | 物理理解不够 (VLA 没 physics) | Video prediction as policy | Latency overhead (video denoising 秒级) |
| **Generalist** | 数据规模不够 + 训练范式错 | 500K hr from scratch + RL | 需巨量 compute, 无法复用社区 VLM |
| **PI** | 单模型能力上限 | Pipeline (planner+WM+VLA+expert) | 系统复杂度高, 组件间接口成本 |
| **Genesis** | 硬件限制操作精度 | 自研 hand + 3ms control + sim eval | 硬件 scale 慢, 依赖自研供应链 |

**这五个赌注不互斥，但优先级截然不同。** 最终赢家可能是能最快收敛到 "enough of each" 的公司。

---

## 5. 共同趋势

1. **都拒绝 "fine-tune VLM" 范式** — 只有 PI 的 π0 仍基于 PaliGemma
2. **都自称 "system" 而非 "model"** — GEN-1, GENE-26.5, Skild Brain 都明确
3. **全都不报 inference Hz** — profiling gap confirmed
4. **Data efficiency 竞赛** — task-specific data 从 large-scale teleop 压到 <1hr
5. **都在做 from scratch** — 不再复用社区 VLM 权重

---

## 6. 对 vlla 研究的影响

### 6.1 Profiling 价值更高
五家竞相 claim performance 但**零公开 latency breakdown**。我们的 exp04b/07a/09a 数据是目前公开的几乎唯一 E/C/A breakdown。

### 6.2 Pipeline 架构是 serving 优化的天然场景
- PI 的四件套 (Planner → WM → VLA → Expert)
- Rhoda 的三件套 (Video Model → IDM → Leapfrog Scheduler)
- Genesis 的 Model → Control Middleware → Hand
每种都可以做阶段分离 + overlap + pipelining — 这是 DistServe-style disaggregation 的直接迁移。

### 6.3 "From scratch" 趋势扩大 scope
如果主流不再 fine-tune VLM，纯 "VLM 的 vLLM" 可能覆盖面不足。Serving framework 需要扩展到 robotics-native model (flow matching, causal video DiT, hierarchical policies)。

### 6.4 Control latency vs Model latency
Genesis 的 3ms control latency vs 我们测的 50-500ms model inference → **bottleneck 明确在 model，不在 control stack**。这强化了 model-level profiling 和 serving 优化的必要性。

---

*2026-05-11. 五家 blog 全文读取 (web-fetcher), 与 industrial-wam-landscape + pi-series-evolution + rhoda-dva-generalist-gen1-deep-dive 交叉参照。*
