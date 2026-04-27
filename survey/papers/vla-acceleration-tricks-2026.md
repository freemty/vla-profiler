# VLA Acceleration Tricks — 9 篇 model-level 加速论文汇总 (2026)

> **目的**: 对 `vla-wam-serving-systems-2026.md` §B.1 列举的 9 篇"算法/模型侧加速"做轻量综述，一次性吃掉，避免逐篇 deep-dive。
> **定位**: 这 9 篇都不是 serving system（那是 §A 的 OxyGen/VLAgents），也不是 world model（那是 WAM 一支），而是**在单模型 forward 内**通过改架构/改 decoding 来压缩延迟的工作。
> **日期**: 2026-04-27
> **方法**: 主进程 `curl arxiv.org/abs/{id}` 读 abstract，一手信息入表
> **对 exp08/候选 D' 的价值**: 见 §4——只有 2 篇值得深读 (PD-VLA / OpenVLA-OFT)，其余作为 landscape baseline

---

## 1. 按加速机制分类

四大族：**(A) parallel decoding** / **(B) one-step flow / early observation** / **(C) model shrinking** / **(D) fine-tuning recipe**。Fast-WAM 作为"skip-imagination"独立一类已有 deep-dive。

### (A) Parallel decoding — 把序列化变并行

| 论文 | arXiv | 机制 | 关键数字 |
|------|-------|------|---------|
| **PD-VLA** | 2503.02310 | 首个 "parallel decoding for VLA + action chunking" 框架。把 AR decode 重写为非线性系统用并行 solver 解 | "significant speedup", action chunk 线性膨胀问题的正面回答 |
| **Discrete Diffusion VLA** | 2508.20072 | 把 AR 和 MLP/diffusion action head 统一到**单 transformer + discrete diffusion**。可并行解 action chunks | 保留 diffusion 的 progressive refinement, 消除 fragmented pathway |

### (B) One-step flow / streaming — 压 denoise 步数或流水线

| 论文 | arXiv | 机制 | 关键数字 |
|------|-------|------|---------|
| **SnapFlow** | 2604.05656 | 对 pi0/pi0.5/SmolVLA 用 progressive self-distillation, 把 10 ODE step 压到 1 step | Denoise 原占 **80%** e2e latency (与 exp07a Pi-Zero 82% 吻合), 1-step 无回退 |
| **FASTER** | 2603.19199 | 重新定义 "reaction time" (VLA 对环境变化响应), 做 async inference with trajectory smoothness + reactive latency 双目标 | 系统性分析 action chunking 的 reaction 因子 |
| **StreamingVLA** | 2603.28565 | 把 observation/generation/execution 三段从串行改流水线 + action flow matching + adaptive early observation | 消除 stage-wait halting |

### (C) Model shrinking — 改架构或规模

| 论文 | arXiv | 机制 | 关键数字 |
|------|-------|------|---------|
| **A1** | 2604.05672 | 完整开源 "truncated" VLA framework——裁 VLM backbone + 小 diffusion/flow head | Commodity GPU real-time 目标 |
| **NanoVLA** | 2510.25122 | Routing decoupled VLU → "nano-size generalist policy"，为 Jetson Orin Nano 级设备设计 | Edge device first class |

### (D) Fine-tuning recipe — 不改架构改训练

| 论文 | arXiv | 机制 | 关键数字 |
|------|-------|------|---------|
| **OpenVLA-OFT** | 2502.19645 | 系统研究 VLA fine-tuning 的 speed/success 联合优化——哪些层冻结、什么 LR schedule、哪些数据比例 | Novel robot setup 下有效 ft 配方 |

### (E) 特殊类 — skip-imagination WAM

| 论文 | arXiv | 机制 | 我们已覆盖 |
|------|-------|------|----------|
| **Fast-WAM** | 2603.16666 | WAM 是否真需要 test-time video imagination 的系统性 ablation | **exp04a 已 profile**，见 `hao-style-synthesis.md` 和 `exp/exp04a/README.md` |

---

## 2. 三个横向观察

### 2.1 A 阶段是"全员公认"的瓶颈
SnapFlow 写明 "denoising alone accounts for 80% of e2e inference"——**与 exp07a 实测 Pi-Zero Action 82% 完全吻合**。FASTER / StreamingVLA / Discrete Diffusion VLA 的动机段都指向同一数据点。

**对 exp08 意义**: "A 阶段 dominate" 不再是我们的独家观察，而是 2026 共识。exp08 要继续有差异化，必须往 **coloc 干扰 / EPD disaggregation / SLO benchmark** 的 system 层推——单纯 "测出 A 是瓶颈" 已无新意。

### 2.2 "Parallel" 和 "One-step" 是两条正交加速轴
- Parallel (PD-VLA / Discrete Diffusion VLA): 减少**每步内的序列化**
- One-step (SnapFlow / FASTER): 减少**步数**
- 两者可以叠加 (未见论文组合)，exp08 如果做 kernel contention model 要把二者分开建模

### 2.3 Edge-first 的论文在走"专用 small 模型"路线
NanoVLA / A1 都是"给 Jetson 级硬件定制 VLA"。这和候选 D' (VLA SLO benchmark) 的场景一致：**SLO benchmark 应包含 edge 硬件档位** (Jetson Orin Nano / Thor) 而不只是 RTX 5880 Ada。

---

## 3. 与项目已有 controller 的映射

| 我们的 controller | 对应加速论文 | 可迁移性 |
|-------------------|------------|---------|
| `PiZeroController` (pi0, 10-step flow, 82% A) | **SnapFlow** (pi0 1-step distill) | **高** — 可以复现 SnapFlow 的蒸馏流程并测 canonical 新延迟 |
| `LingBotVLAController` (Qwen-VL-3B + flow head) | PD-VLA (parallel decoding) | **中** — 论文针对 AR decode, LingBot 用 flow head，机制需 adapt |
| `NitroGenController` (174M DiT, 7.2ms/step) | StreamingVLA (pipeline stages) | **低** — NitroGen 已经很快，stream gain 有限 |
| `OpenVLAController` | **OpenVLA-OFT** | **高** — 直接对应，可按论文配方 ft 后测 success rate trade-off |
| `LingBotVAController` (full WAM, 2518ms) | Fast-WAM (skip-imagination) | **exp04a 已做** |

---

## 4. 哪些值得深读 (action items)

**(建议仅 2 篇 deep-dive)**:

1. **PD-VLA (2503.02310)** — 因为它的 parallel decoding 会直接影响 exp08a 对 coloc 干扰的解读。如果 action chunk 内部可以 parallel decode, kernel-launch contention model 必须把这类 kernel fusion 考虑进去
2. **OpenVLA-OFT (2502.19645)** — 我们已有 OpenVLAController 但未训，按这篇的 speed/success recipe ft 是候选 D' (VLA SLO benchmark) 最轻量的入场路径

**其余 7 篇** (FASTER / StreamingVLA / SnapFlow / A1 / NanoVLA / Discrete Diffusion VLA / Fast-WAM): 本文作为 landscape baseline 够用，**不单独写 deep-dive**；遇到具体实验需要时按 arxiv ID 回查。

---

## 5. 引用清单

- PD-VLA — 2503.02310 — https://arxiv.org/abs/2503.02310
- Discrete Diffusion VLA — 2508.20072 — https://arxiv.org/abs/2508.20072
- SnapFlow — 2604.05656 — https://arxiv.org/abs/2604.05656
- FASTER — 2603.19199 — https://arxiv.org/abs/2603.19199
- StreamingVLA — 2603.28565 — https://arxiv.org/abs/2603.28565
- A1 — 2604.05672 — https://arxiv.org/abs/2604.05672
- NanoVLA — 2510.25122 — https://arxiv.org/abs/2510.25122
- OpenVLA-OFT — 2502.19645 — https://arxiv.org/abs/2502.19645
- Fast-WAM — 2603.16666 — https://arxiv.org/abs/2603.16666 (已有 `exp/exp04a/`)

---

*2026-04-27. 基于 9 篇 arXiv abstract 的 consolidated landscape 汇总，避免 9 篇逐一 deep-dive 的 diminishing return。所有 arXiv ID 与 §A `vla-wam-serving-systems-2026.md` 的 18/18 校验一致。*
