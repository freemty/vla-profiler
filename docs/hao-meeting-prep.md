# 第一次 Meeting 大纲 — 与张昊

> 定位：请教式 meeting，不是汇报式。带着 survey context + 数据 + 一个核心问题去，听他的判断。

## 开场 (2 min)

自我介绍 + 表达目前状态：
- 拿到 offer 后立刻开始调研和实验，已有 13 个实验 + 6 份 survey
- 对领域版图有大致了解，但对**方向选择**有一个需要他判断的核心问题

## Q0 — 核心问题 (15 min)

> "VLA/Action Model 的 serving 是一个真需求吗？如果是，它长什么样？"

### 请教的 context

带着 survey 结论去提问，不是空手问：

**现状**（从我的 survey 中）：
- 今天的 VLA 部署 = **一个机器人独占一张卡**，不存在 serving 问题
- LLM serving (vLLM) 之所以成立，是因为千万用户共享 GPU → batching/scheduling/PagedAttention 有真实需求
- VLA 目前**没有**这个场景。OxyGen (2603.14371) 是唯一一篇做 VLA continuous batching 的，但很初步

**但趋势在变**：
- 多机器人编队：亚马逊 75 万 Kiva、工厂产线 → 集中推理服务器 → N 机器人 : M 张卡
- 模型变大：π0.7 = 四件套 (HLP + WM + VLA + Expert)、DreamZero = 14B → 边缘放不下
- PI 和 Generalist 2026-04 同月承认 "model is a system" → system = 多组件 = 需要调度
- 自动驾驶 VLM 在走云-端混合 (Wayve LINGO-2)

**想请教**：
1. 您觉得 VLA serving 的真需求会在**什么时间点**出现？1 年？3 年？5 年？
2. 它的形态更像 **LLM serving** (多用户共享 stateless 请求) 还是 **游戏服务器** (有状态 session、持久连接)？
3. 如果现在投入做 VLA serving 研究，是**太早** (需求不存在) 还是**正好** (提前占坑)？
4. 您做 vLLM 时，LLM serving 的需求已经很明确了吗？还是也有赌的成分？

### 为什么这个问题重要

答案直接决定接下来的路线选择：
- 如果 serving 是真需求 → 做 EPDA serving system (候选 D')
- 如果 serving 太早 → 做 model-level 加速 (候选 C: DiT caching) 或 mechanism study
- 如果形态不同于 LLM → 可能需要全新的设计（不是扩展 vLLM）

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
