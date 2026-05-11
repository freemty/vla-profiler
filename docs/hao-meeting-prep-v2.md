# 第一次 Meeting 大纲 v2 — 与张昊

> **叙事线升级**: 不再只是 "我测了 7 个模型"，而是 "我把整个 VLA/VA 生态测了一遍，看到了工业界的路线分野，找到了系统优化的结构性机会"。
> **新增**: 10 家 startup landscape + OFT 速度王者 + design space 二维图

---

## 新 Slide Deck 结构 (12-14 slides)

### Slide 0: Title
- "From 3ms to 2.5s — Mapping the VLA/VA Design Space"
- 日期 + UCSD / Hao AI Lab

---

### Slide 1: 工业版图 — 两条路线的对立 (NEW)

> **核心观点**: 2026-Q1，VLA 和 VA (World Action Model) 两条路线已经**商业信念级对立**。

**两栏布局**:

左栏 — **VLA 阵营** (推理时无 video gen):
| 公司 | 旗舰 | 信号 |
|------|------|------|
| Physical Intelligence | π0.7 (pipeline system) | 85人，4组件 |
| Generalist AI | GEN-1 ("model is a system") | 50万hr wearable pretrain |
| Figure | Helix (S1/S2, 7-9Hz) | 200Hz visuomotor loop |
| Google DeepMind | Gemini Robotics 1.5 | — |
| NVIDIA | GR00T N1.5 | Cosmos 仅用于 pretrain |

右栏 — **VA / WAM 阵营** (推理时生成 video):
| 公司 | 旗舰 | 信号 |
|------|------|------|
| 1X Technologies | 1XWM | "derives actions from video gen" |
| Rhoda AI | FutureVision ($450M, $1.7B val) | "video-predictive control" |
| NVIDIA GEAR | DreamZero (7Hz on GB200) | Wan2.1 14B DiT |
| Cosmos Policy | Unified video-policy (LIBERO 98.5%) | 2B DiT latent frame injection |

**底部 callout**: 
- 1XWM blog 原文点名 "PI0.6, Helix, Groot N1.5" 为对立面
- 2026-04 同月，PI 和 Generalist 都承认 "model is a system"
- 延迟差异: **VLA ~5Hz vs WAM ~0.4Hz** — 工业选择完全对齐延迟现实

---

### Slide 2: Design Space 二维散点图 (NEW)

> **横轴**: Total Latency (ms, log scale)
> **纵轴**: Action Head Type (从简单到复杂: MLP → Flow → DiT → DiT+Video)
> **气泡大小**: Backbone 参数量
> **颜色**: 路线 (VLA = 蓝, WAM = 红, OFT = 绿)

数据点 (全部来自我们实测):
```
ACT              3ms     Single-forward MLP    0.1B    exp02a
OpenVLA-OFT    109ms     Parallel MLP (OFT)    7B      exp11a ★NEW
StarVLA-OFT     63ms     Parallel MLP (OFT)    3B      exp11b ★NEW
LingBot-VLA     75ms     10-step Flow          3B      exp03a
Pi-Zero        200ms     10-step DiT (300M)    2.7B    exp07a
Fast-WAM       407ms     10-step DiT (350M)    5B      exp04a
Cosmos Policy  659ms     5-step DiT (2B)       2B      exp09a
LingBot-VA    2518ms     DiT + Video gen       5B      exp04b
```

**标注**:
- 10Hz 线 (100ms) — 机器人实时阈值
- 5Hz 线 (200ms) — 勉强可用
- OFT 路线突破: **瓶颈从 Action (80-94%) 翻转为 Backbone (84-99%)**

---

### Slide 3: OFT — 瓶颈翻转的新范式 (NEW)

> **核心观点**: OFT parallel MLP 把 action 延迟从 165ms 压到 0.13ms (1270x)。但瓶颈没消失，只是转移了。

**三栏对比**:
| | Pi-Zero (flow) | OpenVLA-OFT (7B) | StarVLA-OFT (3B) |
|---|---|---|---|
| E (encode) | 9ms (5%) | 17ms (15%) | 35ms (55%) |
| C (context) | 26ms (13%) | 92ms (**84%**) | 28ms (45%) |
| A (action) | **165ms (82%)** | 0.24ms (0.2%) | 0.13ms (0.2%) |
| Total | 200ms / 5Hz | 109ms / 9Hz | 63ms / **16Hz** |

**Insight**: 
- OFT 消灭了 A 瓶颈 → 现在是纯 backbone efficiency 问题
- 3B backbone (StarVLA) 比 7B (OpenVLA) 快 1.7x → **backbone size 是 OFT 的唯一 knob**
- **OFT + 小 backbone = 目前唯一跨过 10Hz 线的开源 VLA** (StarVLA 15.8Hz)

---

### Slide 4: 四次跳跃 (原 Act 1，更新数据)

原有内容保留，新增第 5 个跳跃:

### 跳跃 5 (NEW): 把 Action Head 砍掉 → OFT 回归 10Hz+

OpenVLA-OFT / StarVLA-OFT 证明：如果 action head 足够轻（parallel MLP），**瓶颈完全回到 backbone**。

```
          跳跃 1     跳跃 2      跳跃 3       跳跃 4      跳跃 5
          ACT      Flow VLA    DiT VLA      Full WAM     OFT VLA
          3ms      74ms        200-407ms    2518ms       63-109ms
          300Hz    13Hz        2.5-5Hz      0.4Hz        9-16Hz
          ↑                                              ↑
       下界                                           实时回归
```

---

### Slide 5: Action DiT Scaling (原 slide, 加新数据点)

DiT scaling curve 更新:
```
OFT MLP:  0.13-0.24ms/step  (不在 curve 上，太快了)
174M DiT: 7.2ms/step         exp06a
300M DiT: 16.5ms/step        exp07a  (含 cross-attn tax)
350M DiT: 32ms/step          exp04a
2B DiT:   76.8ms/step        exp09a  (Cosmos Policy)
```

**新 insight**: OFT 证明 action head 可以从 32ms/step 回到 0.13ms — 代价是从 iterative refinement 退回 single-pass regression。Quality-latency trade-off 是核心问题。

---

### Slide 6: VLM Attention 被重塑 (原 Act 2, 不变)

---

### Slide 7: 版图空白 + 优先级 (原 Act 3, 更新)

原有 4 条空白 + 新增:

5. **OFT 的 quality ceiling 在哪？** — 97% LIBERO 看起来够，但跨 embodiment / 长程任务没验证
6. **Backbone 加速 (非 action)** — OFT 把问题转化为纯 VLM 推理加速，vLLM/flash-attn/quantization 全部适用

**优先级更新**:
```
A: Action DiT 加速        ← 对 flow/DiT VLA 主流有效
A': Backbone 压缩 (OFT)  ← 对 OFT 路线有效 (知识蒸馏/量化)
B: VLA inference bench    ← 统一 profiling，填空白
C: DiT caching for VLA   ← A 的子方向
```

一句话: **"两条加速路径 — 压 action head (A), 或砍掉 action head + 压 backbone (A')"**

---

### Slide 8: 请教 (原 Act 4, 更新问题)

1. VLA 还是 VA — **您看好哪条路线？** (PI/Figure 都不做推理时 video gen)
2. OFT 把问题变成纯 backbone efficiency — **这跟 vLLM/FastVideo 的关系是什么？**
3. FastVideo STA/蒸馏 — **能直接迁移到 VLA Action DiT 吗？**
4. 我应该先做 A (Action DiT 加速) 还是 A' (OFT + backbone 压缩)？
5. 组里谁在做相关的？

---

### Slide 9: Reproducibility (原 backup, 不变)

---

### Slide 10: EPDA Contention (原 backup, 不变)

---

### Slide 11: 一页带走 (Closing)

```
9 models × 11 experiments = 第一张 VLA/VA 延迟全景图

发现:
  · Action DiT = 主流 VLA 的 80-94% 瓶颈 (跳跃 3)
  · OFT 消灭了 action 瓶颈，问题变成 backbone (跳跃 5)
  · 工业两派: VLA (~5Hz) vs WAM (~0.4Hz)
  · VLM attention pruning 不可迁移到 VLA

下一步: Fast VLA first — 压 Action DiT (A) 或压 Backbone (A')
```

---

## 与旧版差异

| 维度 | v1 (2026-04-28) | v2 (2026-05-11) |
|------|-----------------|-----------------|
| 模型数 | 7 | **9** (+OpenVLA-OFT, StarVLA-OFT) |
| 实验数 | 6+3 contention | **11** (+exp09a, exp11a, exp11b) |
| 叙事 | 四次跳跃 | **五次跳跃** (加 OFT 回归实时) |
| 缺少 | 工业版图 context | **10 家 startup landscape slide** |
| 缺少 | Design space 全景图 | **二维散点图 (latency × action type)** |
| 缺少 | OFT 路线 | **OFT 瓶颈翻转 slide** |
| 优先级 | A > B > C > D | **A + A' 双路径** |
| 问题 | 4 个 | **5 个** (加: VLA vs VA 您看好哪条) |

## 准备 checklist

- [ ] 用现有 slide 视觉风格重写 12 slides
- [ ] Design Space 散点图 (SVG 或 CSS, 含 10Hz/5Hz 线)
- [ ] 10 家 startup 两栏表
- [ ] OFT 三栏对比表 (E/C/A breakdown)
- [ ] 更新 exp summary 表 (9 models)
- [ ] 口头 1-2 句 exp08 备用
