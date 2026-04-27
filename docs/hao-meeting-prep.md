# 第一次 Meeting 大纲 — 与张昊

> **定调：Fast VLA first, serving later.**
> 定位：请教式 meeting。带着 survey context + profiling 数据 + 自己的判断去，请他校准。

## 我的判断 (meeting 前先亮出来)

VLA 推理现在卡在**单请求太慢**，不是并发不够：

```
ACT          3ms    300Hz   ✅ 足够，但只能做窄任务
LingBot-VLA  75ms    13Hz   ⚠️ 刚刚够简单抓取
Pi-Zero     200ms     5Hz   ❌ 多数实时场景不够
DreamZero  2518ms   0.4Hz   ❌ 完全不可用
```

机器人需要 10-50Hz。VLA 单次推理还差 2-10x。**这是 FastVideo 而不是 vLLM 的问题阶段。**

同时：VLA 推理没有统一框架 — 5 个模型 5 套 env 5 个入口，连 profiling 都没有标准方式。领域处于 "wild west"。

**Serving (高并发/多用户调度) 是模型够快之后才会浮现的需求。** 就像 FastVideo 先把视频生成从分钟压到秒，serving 需求才跟着来。

## 开场 (2 min)

- 拿到 offer 后立刻启动调研 — 13 个实验 (7 模型 profiling + 6 contention 测量) + 6 份 survey
- 目前对 VLA inference 版图有大致认识，带了一个判断来请您校准

## Q0 — 我的判断对不对？(10 min)

> **"VLA 现在更像 2022 年的 video generation（单请求太慢，需要 FastVideo 式加速），还是 2023 年的 LLM（并发爆发，需要 vLLM 式 serving）？"**

**我认为是前者。理由：**
1. 单次推理离实时差 2-10x（数据见上）
2. 部署现状是一机一卡，不存在 batching 需求
3. OxyGen 是唯一做 VLA serving 的，但 VLA 连单次推理都没优化好
4. vLLM-Omni / SGLang Diffusion 已经在做 multimodal serving → 我们不该重复

**想请教：**
1. 您同意 "fast first, serving later" 吗？还是您看到了我没看到的并发需求？
2. 您做 FastVideo 时的核心 insight（STA/蒸馏）哪些能直接迁移到 VLA？
3. VLA 推理加速最该先攻哪个阶段？我的数据说 **Action (DiT denoising) 占 80-90%** — 您同意从这里入手吗？

## Q1 — 方向选择 (10 min)

### 背后的 Technical Bet

> **"机器人通用操作是否必须依赖端侧放不下的大模型？"**

| 如果大模型不可避免 | 如果小模型够用 |
|-------------------|--------------|
| Fast VLA 做完后 serving 自然成为下一步 | Fast VLA 本身就是终局 |
| PhD 路线: 加速 → serving → co-design | PhD 路线: 蒸馏 + 量化 + on-device |

目前没有 VLA model-size vs generalization 的 scaling curve — 这件事本身是不是值得做？

### 候选路线 (请他排序)

| 候选 | 一句话 | 我的评估 |
|------|--------|---------|
| **A: VLA 单次推理加速** | FastVideo 思路迁移到 VLA action denoising (STA/蒸馏/step caching) | ⭐ 最优先 — 直接 bottleneck |
| **B: VLA inference benchmark** | 统一 profiling 框架 + SLO benchmark (填 wild-west 空白) | 配套 A，低风险 |
| **C: DiT caching for VLA** | 逐层 activation variance → 最优 cache 策略 | A 的子方向 |
| D: Serving system | EPDA disaggregation | 太早 — 等 A/B 做完再说 |
| E: Mechanism study | exp08 contention 分析发 workshop paper | side project |

## Q2 — 展示数据 (5 min)

**7-model Pareto 图** — 一张图展示全部 profiling：

```
           延迟 →
ACT  ■ 3ms
NitroGen  ■■ 18ms (k=1)
LingBot-VLA  ■■■■■ 75ms
NitroGen  ■■■■■■■■ 126ms (k=16)
Pi-Zero  ■■■■■■■■■■■■ 200ms
Fast-WAM  ■■■■■■■■■■■■■■■■■■■■■ 407ms
Full WAM  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 2518ms
```

**一个核心发现**：Action 阶段 (DiT denoising) 占 80-94% 延迟 — 这是加速最该攻的地方。

**DiT scaling curve** (exp06a + exp07a):
```
174M (NitroGen)  →  7.2ms/step
300M (Pi-Zero)   → 16.5ms/step  (cross-attn 使 300M 比纯 DiT 贵 2.3x)
350M (Fast-WAM)  → 32ms/step
```

**关键一句话**："Action 占 80%+，而且 DiT 越大越贵呈超线性增长 — FastVideo 的 STA/蒸馏在这里有巨大空间。"

**exp08 附带提一句** (不展开)：
- EPDA 四阶段 co-location 时 D/P 脆弱 2.4-2.9x，E/A 鲁棒
- 将来做 serving 时可直接用，目前 park

## Q3 — 听建议 (5 min)

1. 入学前这几个月，您建议我先做什么？直接上手 VLA 加速实验还是先补系统课？
2. 组里谁在做 VLA 或 FastVideo 相关的？可以对接？
3. 您觉得我目前最缺什么能力？

## 准备 checklist

- [ ] 画 7-model Pareto 图 (一页, 有 Hz 刻度)
- [ ] 画 DiT scaling curve 图 (174M / 300M / 350M per-step, 含 cross-attn 标注)
- [ ] 读 FastVideo 论文 (STA + VSA + 蒸馏, 至少前 4 节)
- [ ] 读 DistServe 前 3 节 (能说清 PD disaggregation, 作为"为什么 serving later"的背景)
- [ ] `slides/epda-roofline-motivation.html` 更新: 加 "Fast VLA First" framing
- [ ] 准备 exp08 一页总结 (备用, 只在他问时展开)

## 时间控制

总 ~30min。**Q0 + Q1 是主菜 (20min)**。数据是论据不是主角。
展示的重点是 **你的判断力和问题意识**，不是你跑了多少实验。
