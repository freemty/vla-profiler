# Hao Meeting v2 — Slide 文案（精简版）

> 每页：标题 + 一句话 + 数据。解释口头讲。

---

## [Slide 0] Title

From 3ms to 2.5s
**9 模型 · 11 实验 · 10 家公司**

---

## [Slide 1] VLA vs VA：谁做推理时 video gen？

（两栏表，不需要正文）

| VLA（不做 video gen） | VA / WAM（推理时生成 video） |
|---|---|
| PI — π0.7 | 1X — 1XWM |
| Generalist — GEN-1 | Rhoda — FutureVision ($450M) |
| Figure — Helix | NVIDIA — DreamZero |
| Google — Gemini Robotics | Cosmos Policy (LIBERO 98.5%) |
| NVIDIA — GR00T N1.5 | |

**底部一句**: VLA ~5Hz vs WAM ~0.4Hz。

---

## [Slide 2] 七个系统

（卡片网格，每张卡只留：名字 + 一句架构 + 我们的数据）

| 系统 | 一句话 | 我们的数据 |
|------|--------|-----------|
| π0.7 | 4 组件 pipeline，300M Action Expert | 200ms / 5Hz |
| GEN-1 | 黑箱，500K hr 穿戴数据 pretrain | 无 |
| GENE-26.5 | 20-DoF 灵巧手 + sensor glove + sim | 无 |
| Cosmos Policy | 2B DiT，action = latent frame | 659ms / 1.5Hz |
| DreamZero | 14B video WM → 蒸馏 policy | 无 |
| LingBot-VA | 加 video gen → 慢 34x | 2518ms / 0.4Hz |
| Rhoda | "Direct Video Action"，零技术披露 | 无 |

---

## [Slide 3] 延迟 × 复杂度

（散点图，不需要文字。底部一句）

**红线 = 10Hz。只有 OFT 过线。**

---

## [Slide 4] OFT：瓶颈翻转

**Action 165ms → 0.13ms。瓶颈从 action 转到 backbone。**

| | Pi-Zero | OpenVLA-OFT (7B) | StarVLA-OFT (3B) |
|---|---|---|---|
| E | 9ms (5%) | 17ms (15%) | 35ms (55%) |
| C | 26ms (13%) | 92ms (84%) | 28ms (45%) |
| A | 165ms (82%) | 0.24ms | 0.13ms |
| **Total** | **200ms / 5Hz** | **109ms / 9Hz** | **63ms / 16Hz** |

---

## [Slide 5] 五次跳跃

| | 范式 | 延迟 | Hz |
|---|---|---|---|
| 1 | ACT (single forward) | 3ms | 300 |
| 2 | VLM + flow head | 74ms | 13 |
| 3 | Action DiT | 200-407ms | 2.5-5 |
| 4 | Full WAM | 2518ms | 0.4 |
| **5** | **OFT** | **63-109ms** | **9-16** |

**跳跃 3 最重。跳跃 5 砍迭代换速度。**

---

## [Slide 6] 2x params → 4.4x latency

| | Params | Per-Step | Cross-Attn Tax |
|---|---|---|---|
| OFT MLP | ~2M | 0.13ms | — |
| NitroGen | 174M | 7.2ms | 无 |
| Pi-Zero Expert | 300M | 16.5ms | +35% |
| Fast-WAM | 350M | 32ms | +100% |
| Cosmos | 2B | 76.8ms | monolithic |

**2x params → 4.4x latency。Cross-attn 是隐藏税。**

---

## [Slide 7] Attention 被 VLA 训练重塑

| | VLM | VLA fine-tune 后 |
|---|---|---|
| Gini | >0.91 | 0.07 |
| Sink | Pos 2 (12-28x) | Pos 64 |
| Entropy | V-shape | flat |

**VLM pruning 不可迁移到 VLA。**

---

## [Slide 8] 两条路

| Path A | Path A' |
|---|---|
| 压 Action DiT | 砍 action head + 压 backbone |
| FastVideo STA / 蒸馏 / caching | OFT + flash-attn / 量化 |
| → Pi-Zero, Fast-WAM, Cosmos | → OpenVLA-OFT, StarVLA-OFT |

**单请求延迟差 2-10x。先加速，再 serving。**

---

## [Slide 9] 问 Hao

1. VLA vs VA，您看好哪条？
2. OFT 变成 backbone 问题 — 跟 vLLM / FastVideo 什么关系？
3. FastVideo STA / 蒸馏能迁移到 VLA DiT 吗？
4. 先做 Path A 还是 A'？
5. 组里谁在做相关的？

---

## [Slide 10] Backup — 可复现性

| Model | Random | Real | Δ |
|---|---|---|---|
| NitroGen | 7.2ms/step | 7.1ms/step | <2% |
| Pi-Zero | 200ms | 225ms | +12% |
| Fast-WAM LIBERO | — | 94.5% (paper 93.7%) | match |

---

## [Slide 11] Backup — EPDA 干扰

inflation(X|Y) = 1 + v·a, R²=0.94

{E,A} 安全共卡。{P,D} 必须分开。

---

## [Slide 12] 带走

- 9 模型 × 11 实验
- Action DiT = 80-94% 延迟
- OFT 翻转瓶颈 → backbone
- VLA ~5Hz vs WAM ~0.4Hz
- VLM pruning ≠ VLA pruning
- **Fast VLA first**
