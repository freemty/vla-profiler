# Physical Intelligence π 系列演进 + Generalist GEN-1 对照 (2024-10 → 2026-04)

> **目的**: 把 Physical Intelligence 的 π 系列（π0 / π0.5 / π*0.6 / π0.7）以及同期 Generalist AI 的 GEN-1 串成一张"工业 VLA 技术路线图"，识别出从"单体 VLA"到"多模型 pipeline + 系统观"的方向转变——这直接影响 exp08 候选 D' 的 scope 定义。
> **日期**: 2026-04-27
> **方法**: 主进程 `curl` 官方 blog 原文 + 作者列表 + 关键段落摘录，零 subagent 代笔
> **关联**: `survey/papers/vla-wam-serving-systems-2026.md`、`survey/papers/vla-acceleration-tricks-2026.md`

---

## 0. 一张图总览

```
2024-10  π0 ─────────► single VLA (PaliGemma + Gemma 300M Action Expert, flow)
         │              "first generalist policy"
         │
2025-04  π0.5 ────────► + heterogeneous co-training, open-world generalization
         │              (VLM backbone + 多源数据融合)
         │
2025-11  π*0.6 ───────► + Recap (RL + corrections) on top of π0.6
         │              "mastery via experience"，长程可靠性
         │
2026-04  π0.7 ────────► ★ 从 "model" 变 "pipeline system" ★
         │              High-Level Policy + 轻量 World Model (subgoal image)
         │              + VLA 主干 + Action Expert + 多模态 prompt
         │              compositional generalization 首现
         │
2026-04  GEN-1 ───────► (Generalist AI，非 PI，但同期友商)
                        "model 其实是 system" — explicit 承认
                        Pretrain on 50万小时 human wearable data (无 robot data)
                        1h fine-tune → 99% 成功 + 3x 速度
```

**关键转折**: **2026-04 同月，两家头部都明确承认"VLA 不再是单个 model"**—— PI 用 "pipeline 4 件套"（High-Level + WM + VLA + Action Expert），Generalist 用 "model 其实是 system" 原话。这把 serving 层的复杂度拔高了一个量级。

---

## 1. π0 (2024-10-31) — 单体生成式 VLA

**论文**: "π₀: A Vision-Language-Action Flow Model for General Robot Control"

**架构**: PaliGemma (SigLIP ViT-So400m/14 + Gemma 2B) + Gemma 300M Action Expert，dual-stream + flow matching。这是**我们 exp07a 测的那个**（canonical 200.5ms/5Hz, Action 82%）。

**贡献**: 第一个"generalist robot policy"——单模型跨 robot 平台，可 prompt 可 ft。

**作者**: 18 人（核心：Black / Driess / Finn / Hausman / Ichter / Levine / Pertsch / Shi / Vuong）。

---

## 2. π0.5 (2025-04-22) — open-world 泛化

**论文**: "π 0.5: a VLA with Open-World Generalization"

**核心方法**: **heterogeneous co-training**——在一个 VLA 里同时 co-train: 多个 robot 的 teleoperation + 人类视频 + 网络视觉-语言数据 + web-scale 数据。

**结果**: 在**训练中从未出现过的新家居环境**里做 clean kitchen / bedroom 等长程任务。"不是每次成功，但已有 human-like 的 flexibility"。

**与我们项目的关系**: `allenzren/open-pi-zero` 社区复现包含 π0.5 变体。如果 exp07a 升级到 π0.5 backbone，canonical latency 不会大变（架构同 π0），但语义能力差距大。

---

## 3. π*0.6 (2025-11-17) — RL + coaching，通向 mastery

**论文**: "π*0.6: a VLA that Learns from Experience"

**核心方法**: **Recap** (RL with Experience and Corrections via Advantage-conditioned Policies)——三步学习模拟人类徒弟：
1. Demonstrations（supervised）
2. Corrections（coach 纠错）
3. Autonomous experience（practice makes perfect）

**结果**: 在"最难任务"上 throughput 翻倍 / 失败率减半。能跑一天 espresso 5:30am-11:30pm、新家折 50 件 novel laundry、工厂里装 59 个真实包装盒——**连续数小时无干预**。

**作者**: 53 人（π0 → π0.5 → π*0.6 名单一直在扩张，π*0.6 首现 Amin / Aniceto / Balakrishna / Conley / DiCarlo / Godden / Goryachev / Hancock / Hussein / Jen / Kuchi / Lamb 等工程化大扩招）。

**系统含义**: 为了 Recap 能跑，必须有一个能**连续 autonomous 采数据 → 训练 → 重新部署**的闭环系统。这是 PI 从"model-centric"首次明显偏向"system-centric"的信号。

---

## 4. π0.7 (2026-04-16) ★ 变成 pipeline system

**blog**: "π0.7: a Steerable Model with Emergent Capabilities" (https://www.physicalintelligence.company/blog/pi07)

**作者**: 85 人（跨 π0 全部原班 + π*0.6 工程团 + Ai, Aniceto, Balke, Bokinsky, Cao, Charbonnier, Choudhary, Collins, Dhaka, Eq..）

### 4.1 Architecture (blog figure 描述)

```
                ┌──── Inference Time ─────┐
TASK INSTR ──►  High-Level Policy ──► SUBTASK INSTR ──┐
                      │                                │
                      ▼                                │
                 World Model ──► SUBGOAL image ────────┤
                                                       ▼
  Observation ──┐                            ┌──► π 0.7 VLA
  Memory       ─┼────────────────────────────┤   (Vision-Language-Action)
  Prompt       ─┘                            │      │
                                             │      ▼
                   Desired Metadata ─────────┘  Action Expert
                   (quality, speed)
```

**四件套**:
1. **High-Level Policy** — task → subtask 分解
2. **World Model** — 轻量，给 subtask 生成**视觉 subgoal image**（注意：**不是 video imagination**，是单张 anchor image，远比 Fast-WAM 的 video denoise 轻量）
3. **π0.7 VLA 主干** — 吃 obs + memory + multi-modal prompt + subgoal + metadata
4. **Action Expert** — 出动作（沿用 π0 dual-stream）

### 4.2 Prompt 里塞的四种 conditioning

- 自然语言（task + sub-step）
- **Metadata**（quality / speed）—— 让**次优 autonomous 数据**用 "low quality" 标记后也能吃
- **Control modality label**（joint vs end-effector）
- **Visual subgoal image**（训练时来自数据，推理时 WM 生成）

### 4.3 Emergent capabilities

- **Compositional generalization**: 从没做过 air fryer，step-by-step language coaching 能完成，然后 coaching 可蒸馏进 high-level policy 做全自动
- **Cross-embodiment transfer**: 给没 laundry 数据的新机器人直接折衣服
- **Specialist-level dexterity**: 不 ft 就达到 π*0.6 specialist

### 4.4 对 VLA serving 的系统性冲击

| 变化 | π0 时代 | π0.7 时代 |
|------|---------|----------|
| Forward 对象 | 1 个 VLA | 4 个组件（HLP + WM + VLA + AE） |
| Timing 模型 | E/C/A 三阶段 | E/C/A + **Subgoal Gen 新一阶段** |
| Prompt 结构 | 文本 instruction | 多模态 (text + subgoal image + metadata) |
| 数据质量假设 | 必须高质量 teleop | **低质量数据也能吃**（用 metadata 标注） |
| Serving system 要求 | 单 model runtime | **any-to-any pipeline**（正是 vLLM-Omni 设计目标） |

**结论**: **候选 D' (VLA SLO benchmark) 必须把"4 组件 pipeline"作为标配**，否则 benchmark 打不到 π0.7 层级的真实场景。单纯测 "π0 end-to-end latency" 已经是 2024 的场景了。

---

## 5. GEN-1 (2026-04-02, Generalist AI)

**blog**: https://generalistai.com/blog/apr-02-2026-GEN-1

**主要 claim**（blog 原文）:
- 在之前 SOTA 只有 64% success rate 的 task 上，**average success rate 99%+**
- **3x faster** than prior SOTA on same tasks
- 每个结果只需 **~1 hour of robot data** (新 task 或新 embodiment)
- pretraining 数据集 **50 万小时**，**全部来自人类穿戴设备（无 robot data）**
- GEN-1 = GEN-0 (2025) 的 further scaling + 算法改进

### 5.1 明确承认 "model 是 system"

Blog 原文：

> *"While we may call GEN-1 a model, it is even more accurate to refer to GEN-1 as a **system**. Just as with frontier LLM chatbots and APIs, there are many system-level components across inference and model harnessing that critically advance its performance beyond being just a set of model weights."*

**这是迄今最明确的"VLA serving 需要 system 层"的工业表态**。GEN-1 的组件：pretrain 进步 + post-train + RL + multimodal human guidance + **新 inference-time techniques**。最后这一项与我们的 exp08 方向直接相关（虽然 blog 没披露细节）。

### 5.2 Mastery = 可靠 + 速度 + 即兴

blog 定义的 3 个 mastery 维度：
- **Reliability**: 传统机器人可靠但不 end-to-end；end-to-end 模型之前从没做到跨任务/系统/环境稳定可靠
- **Speed**: "demo 视频太慢" 的老大难。速度上来后世界不再 quasi-static（摩擦动力学变化、运动模糊、对 precision/reactivity/inference 提要求）——**inference latency 是 mastery 的三要素之一**
- **Improvisation**: unexpected 场景中创造性恢复，靠 "physical commonsense"

### 5.3 数据源哲学

**不要 teleoperation 数据**（"enormous teleoperation datasets expensive and difficult to scale"），pretrain 全用**人类穿戴设备**数据（half a million hours）。与 π 系列（主要靠 teleop + 混源 co-train）路线**截然不同**。

---

## 6. 对项目的三条具体行动

### 6.1 Controller 不动，但 profiling 方法论要升级

我们的 `PiZeroController` 目前测的是 **π0 (10-step flow, 200.5ms)**，是 2024 年代的 single-model picture。

**升级路径**（优先级从低到高）:
- (a) exp07a 加个 "subgoal generation" 占位阶段作为 sensitivity 分析（即使 WM 未集成，也可以用 image diffusion proxy 估计 ~50-100ms overhead）
- (b) 候选 D' 的 VLA SLO benchmark 必须把 **Pipeline mode**（π0.7-like 4 件套）作为 top-level tier，而不是只测 single-model
- (c) 如果 π0.7 weights 开源，可以在 `.venvs/pizero` 里直接加载做 canonical 扩展

### 6.2 π*0.6 → OxyGen → 我们的数据链

**有趣的 triangulation**:
- OxyGen (2603.14371, 2026-03) 声称跑 π0.5 multi-task 3.7x 加速
- π*0.6 (2025-11) 需要 continuous autonomous-collection 闭环
- **两者合起来暗示**: OxyGen-style 的 continuous batching 正是 Recap 训练循环急需的基础设施

候选 D' 可以定位为："OxyGen 只做了 KV 共享，没做 PD disaggregation + streaming；π*0.6 的 autonomous 闭环需要这些。"

### 6.3 GEN-1 vs π0.7 的对比是 exp08 motivation figure 的好材料

两家都在 2026-04 同月说 "model is a system"。这段话可以直接当 **advisor meeting 的 opening slide**：
- 不是我们项目"先见"——是工业 consensus，正在发生
- 我们的 job 就是把这些 system 的**时序/资源/bottleneck 特征**量化出来

---

## 7. 引用清单

### Physical Intelligence π series
- **π0** (2024-10-31) — https://www.physicalintelligence.company/blog/pi0 — Paper: `pi0.pdf` on blog
- **π0.5** (2025-04-22) — https://www.physicalintelligence.company/blog/pi05 — Paper: `pi05.pdf`
- **π*0.6** (2025-11-17) — https://www.physicalintelligence.company/blog/pistar06 — Paper: `pistar06.pdf`
- **π0.7** (2026-04-16) — https://www.physicalintelligence.company/blog/pi07 — Paper: `pi07.pdf`

### Generalist AI
- **GEN-1** (2026-04-02) — https://generalistai.com/blog/apr-02-2026-GEN-1
- **GEN-0** (2025) — prior work referenced in GEN-1 blog
- Leadership ref: Kaplan-McCandlish scaling laws (2021), GPT-2 (2019), GPT-3 (2020), RT-2 (2023), PaLM-E (2023), Video Language Planning (Du et al., 2023)

### 内部关联文档
- `survey/papers/vla-wam-serving-systems-2026.md` — OxyGen / VLAgents 是 VLA-dedicated serving 的唯二真系统
- `survey/papers/vla-acceleration-tricks-2026.md` — 9 篇 model-level 加速
- `survey/papers/multimodal-serving-systems-2026.md` — vLLM-Omni / SGLang Diffusion
- `exp/exp07a/README.md` — 基于 π0 架构的 canonical profiling

---

*2026-04-27. 通过 `curl` 5 篇官方 blog 汇总。零 subagent 代笔，作者列表 + 关键段落从 HTML 解析得来，可反查。*
