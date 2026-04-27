# 第一次 Meeting 大纲 — 与张昊

> 定位：请教式 meeting，不是汇报式。带着 survey context + 数据 + 一个核心问题去，听他的判断。

## 开场 (2 min)

自我介绍 + 表达目前状态：
- 拿到 offer 后立刻开始调研和实验，已有 13 个实验 + 6 份 survey
- 对领域版图有大致了解，但对**方向选择**有一个需要他判断的核心问题

## Q0 — 核心问题 (15 min)

> **"机器人的通用操作能力，是否必须依赖一个端侧放不下的大模型？"**

这是一个 technical bet。答案决定 VLA serving 是真需求还是伪需求，进而决定我整个 PhD 的方向。

### The Bet

```
模型大小:  100M  →  1B  →  3B  →  7B  →  14B+系统
泛化能力:   窄     中     ?     广     极广
部署位置:  边缘    边缘   边缘   边缘勉强   必须云端

                         ↑
               存不存在一个"能力悬崖"？
               3B 以下能做到"足够通用"吗？
```

| 如果 YES (大模型不可避免) | 如果 NO (小模型够) |
|--------------------------|-------------------|
| VLA serving 是 vLLM 级别的刚需 | VLA serving 是伪问题 |
| PhD = "做 VLA 的 vLLM" | PhD = model compression / on-device |
| exp08 是 serving system 的 motivation | exp08 是 dead end |
| Hao 的 vLLM 经验直接迁移 | 需要走蒸馏/量化路线 |

### 我的 survey 看到的证据

**YES 侧 (大模型趋势)**:
- 每次泛化能力跳跃都伴随模型变大：ACT 5M (窄) → OpenVLA 7B → π0.7 四件套 → DreamZero 14B (zero-shot)
- PI 和 Generalist 2026-04 同月承认 "model is a system" → 不是一个模型变大, 是多个组队 → 总量更大
- 多机器人编队 (亚马逊 75 万 Kiva, 工厂产线) → 如果大模型, 则 N:M 集中推理服务器
- OxyGen (2603.14371) 是唯一一篇 VLA continuous batching — 说明有人在赌 YES

**NO 侧 (小模型够用)**:
- OpenVLA-OFT 7B 量化到 INT4 → 可能 Jetson AGX Orin 跑 10-20Hz, 很多任务够了
- NanoVLA / BitVLA 存在 (虽然目前窄)
- **VLA 蒸馏还没被认真做过** — LLM 70B→7B 保留 80-90%能力已证明, VLA 没人系统做
- 机器人动作空间远小于语言 — 只需预测 6-DOF 关节角, 不是生成莎士比亚
- 关键：**还没有人做过 VLA 的 model-size vs generalization scaling curve**

### 想请教

1. 您觉得机器人通用操作，最终会落在多大的模型上？存不存在"3B 以下够用"的可能？
2. 您做 vLLM 时 LLM serving 的需求已经很明确了吗？还是也有赌的成分？
3. 如果 VLA serving 是真需求, 它更像 **LLM serving** (stateless 请求) 还是 **游戏服务器** (有状态 session)？
4. VLA model-size scaling curve 这件事本身是不是值得做？(作为判断 bet 的实证基础)

### 为什么这个问题重要

这不只是 exp08 的 motivation, 是**整个 PhD 方向级别的 bet**:
- 如果 YES → 做 serving system (vLLM for VLA), exp08 是 motivation
- 如果 NO → 做 model-level 加速 (蒸馏/量化/caching), exp08 只是 mechanism study
- 如果 unclear → 做 VLA scaling curve 本身就是 contribution (帮领域判断这个 bet)

## Q1 — 方向选择请教 (10 min)

根据 Q0 的回答，展示候选方向请他点评：

| 候选 | 一句话 | 风险 |
|------|--------|------|
| **C: DiT caching for VLA** | FastVideo 蒸馏迁移到 VLA action denoising | 可能太 incremental |
| **D': Mechanism study + VLA SLO** | GPU contention 分析 + VLA benchmark | vLLM-Omni 已占 framework 空间 |
| **新: VLA Serving System** | 做"VLA 的 vLLM" | 需求可能不存在 (Q0 核心) |

**诚实地承认**：
- vLLM-Omni (4.5k★, arXiv:2602.02204) 已有 EPDA framework → 我们不该重复做
- exp08 的实验数据有价值 (非对称 contention, M4 模型)，但 packaging 方式取决于方向

## Q2 — 展示实验数据 (5 min)

不用说太多原理，直接展示**最有冲击力的数字**：

```
exp01a-07a: 7 个模型的延迟 Pareto 图
            ACT 3ms → LingBot-VLA 75ms → Pi-Zero 200ms → Full WAM 2518ms

exp08:      两个阶段放一张卡 → D 慢 3.5 倍、P 慢 2.9 倍
            但 E 和 A 几乎不受影响
            → 能预测 (M4 R²=0.94)
            → 实际部署建议: {E,A} 同卡, {P,D} 必须拆
```

**关键一句话**："DistServe 说 P+D 要拆，我们把这个结论扩展到了 VLA 的 4 个阶段，发现分法不一样 — E 和 A 可以合在一起。"

## Q3 — 听他建议 (5 min)

开放式收尾：
1. 您觉得我目前缺什么能力？需要先补什么课？
2. 入学前这几个月，您建议我做什么？
3. 组里有没有做相关方向的 senior student 可以指导我？

## 准备 checklist

- [ ] `slides/epda-roofline-motivation.html` 更新: 加 Q0 的 serving 讨论页
- [ ] 打印 exp08b 6-pair heatmap (一页纸)
- [ ] 准备 7-model Pareto 图 (一页纸)
- [ ] 读完 DistServe 论文（至少前 3 节，能说清 goodput 和 PD disaggregation）
- [ ] 准备问 vLLM-Omni: 您组里是否有人参与/关注？

## 时间控制

总 35min 左右。**Q0 是最重要的环节**，如果只聊一件事就聊它。
实验数据是锦上添花，不是主菜 — 导师看的不是你跑了多少实验，是你的问题意识和判断力。
