# 第一次 Meeting 大纲 — 与张昊

> 叙事线：**从 3ms 到 2.5 秒 — 我把 action model design space 测了一遍。**
> 定位：展示系统性覆盖 + 从数据中提炼 insight 的能力。

---

## 开场 (2 min)

拿到 offer 后立刻启动调研：7 个模型 profiling + 6 份 survey + 6 个 contention 实验。

一句话："我发现 **action model 正在分化出四种范式**，而瓶颈几乎全在 Action 阶段。今天想 walk through 这个版图，请您校准。"

---

## Act 1 — 从 3ms 到 2.5s：四次跳跃 (10 min)

> 核心图: Action Model Design Space (一张图，四个范式从左到右，每个标注实验编号)

走一遍光谱，每一步讲清：**加了什么能力？付了什么延迟代价？**

### 跳跃 1: Single-Forward → 3ms, 300Hz (exp02a)

ACT: ResNet18 + CVAE，一次 forward 出 action chunk。

- **够快** — 但只能做窄任务 (单臂抓取)，泛化能力约等于零
- Action 几乎不花时间 (0.5ms)，80% 延迟在 CNN encode

→ 这是 **VLA 延迟下界**。

### 跳跃 2: 加 VLM Backbone + Flow Head → 74ms, 13Hz (exp03a)

LingBot-VLA: 把 CNN 换成 3B VLM (Qwen2.5-VL-3B)，action 用 10-step flow head。

- 获得语言理解 + 视觉泛化，代价是 backbone 变重 (E=36ms, C=38ms)
- 但 **action head 几乎免费** — 0.05ms/step, 10 步共 0.48ms
- 瓶颈从 action 翻转为 **backbone** (E+C 占 99%)

→ 轻量 flow head 不是问题。**问题在下一步。**

### 跳跃 3: 独立 Action DiT → 瓶颈翻转 (exp06a, exp07a, exp04a)

把 action head 从小 MLP 换成独立 DiT transformer — 领域主流趋势。

三个 DiT，三个量级:
```
NitroGen  174M DiT   7.2ms/step   @k=1: 18ms (56Hz)    exp06a
Pi-Zero   300M DiT  16.5ms/step   @k=10: 200ms (5Hz)   exp07a
Fast-WAM  350M DiT  32ms/step     @k=10: 407ms (2.5Hz) exp04a
```

**这一步发生了瓶颈翻转** — Action DiT 从 <1% 跃升到 **80-94%** 的总延迟占比。

两个关键发现：
- **DiT scaling 超线性**: 174M→350M 参数 2x，延迟 **4.4x**。从 compute-bound 进入 memory-BW-bound
- **Cross-attn 是隐藏 tax**: Pi-Zero 300M 理论 ~12ms/step (线性外推)，实测 16.5ms — cross-attn to VLM KV 加了 **35% overhead**。Fast-WAM 更极端，30 层每层 MoT cross-attn

→ **这是主战场。** FastVideo 的 STA/蒸馏在这里有巨大空间。

### 跳跃 4: 加 Video Imagination → 延迟爆炸 (exp04b)

LingBot-VA: 用同一个 5B DiT 先生成 video (20 步)，再生成 action (50 步)。

- 2518ms, 0.4Hz — 比 skip-imagination 慢 **6x**
- 获得 zero-shot 泛化 (DreamZero 路线)，但完全不可实时

→ Full WAM 是 **能力上界**，但需要 10x+ 加速才有实用价值。

### 小结: 一张图看全景

```
       3ms          74ms      18-407ms        2518ms
        |             |          |              |
     [ACT]    [LingBot-VLA]  [DiT VLAs]   [Full WAM]
     300Hz        13Hz       2.5-56Hz       0.4Hz
                                ↑
                           主战场: Action DiT
                           占 80-94% 延迟
```

机器人需要 **10-50Hz**。只有跳跃 2 和部分跳跃 3 够用。差距: **2-10x**。

---

## Act 2 — VLM 侧的一个意外发现 (3 min)

> 如果时间紧可以跳过，但这个发现很有趣。

VLM 的 attention 呈极端稀疏 (Gini >0.91)，Pos 2 是 universal sink — 说明 **VLM token pruning 有很大空间** (exp01b)。

但 VLA fine-tuning 之后，**attention 被彻底重塑**: Gini 崩塌到 0.07，sink 迁移，entropy 变 flat (exp05a)。

消歧实验确认这是 fine-tuning 效应，非 model size (exp05b)。

→ **VLM pruning 不可直接迁移到 VLA。** 如果要做 VLA token 压缩，需要在 VLA 训练后的 attention pattern 上重新设计。

---

## Act 3 — 空白与判断 (5 min)

### 版图里缺什么

1. **没有 Action DiT 加速** — FastVideo 的 STA / 蒸馏 / step caching 没人迁移过来
2. **没有统一 benchmark** — 5 个模型 5 套 env，连 profiling 都没标准方式
3. **没有 model-size scaling curve** — 不知道"够用"的最小 DiT 是多大
4. **Cross-attn overhead 没人专门优化** — 贡献 35-440% 额外延迟

### 我的优先级

```
A: Action DiT 加速              ← 直接 bottleneck, FastVideo 迁移
B: VLA inference benchmark      ← 填 wild-west 空白, 配套 A
C: DiT caching for VLA          ← A 的子方向
D: Serving system               ← 太早, 单请求还差 2-10x
```

一句话: **"Fast VLA first, serving later."**

VLA 现在更像 2022 的 video generation (单请求太慢, FastVideo 阶段)，不是 2023 的 LLM (并发爆发, vLLM 阶段)。

---

## Act 4 — 请教 (5 min)

1. 我测了 4 种范式 7 个模型 — **漏了什么关键架构？**
2. FastVideo 的 STA / 蒸馏 — **哪些能直接迁移到 VLA Action DiT？**
3. 入学前这几个月 — **先上手加速实验，还是先补系统课？**
4. 组里谁在做相关的？

---

## 备用材料 (他问才展开)

- **DiT Scaling Curve 图**: 174M / 300M / 350M per-step，标注 cross-attn overhead
- **exp08 contention 一页**: EPDA 四阶段共存，D/P 脆弱 2.4-2.9x，E/A 鲁棒 — serving 方向的预研
- **DistServe PD disaggregation**: 作为 "为什么 serving later" 的理论支撑

## 准备 checklist

- [ ] 画 **Action Model Design Space 图** (四范式 × per-step cost / total latency / Hz, 标实验编号)
- [ ] 画 **DiT Scaling Curve** (174M / 300M / 350M, 含 cross-attn 标注)
- [ ] 读 FastVideo 论文前 4 节 (STA + VSA + 蒸馏)
- [ ] 过一遍 slides/ 确保能快速切到任意一张数据图
- [ ] 1min 版 exp08 口头总结 (不做 slides, 口述即可)

## 时间控制

总 ~25min。**Act 1 是主菜 (10min)** — 走四次跳跃, 数据是论据。Act 3 亮判断。Act 4 听建议。

节奏: **讲故事, 不念表格。** 表格是 backup, 叙事线是 "每一步加了什么能力、付了什么代价"。
