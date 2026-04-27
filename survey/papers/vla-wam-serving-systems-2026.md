# VLA / WAM 专用 Serving / Efficiency Systems 现状调研 (2026-04-27)

> **目的**: 精确定位"有没有专门做 VLA/WAM 的 serving/inference 系统"——不是通用 multimodal serving (已有 `multimodal-serving-systems-2026.md` 覆盖 vLLM-Omni/SGLang Diffusion)，不是 algorithmic acceleration，不是 benchmark。
> **结论**: **真空白基本成立**。截至 2026-04 只有 2 篇预印本真正把 VLA 当 serving first-class citizen (**OxyGen** + **VLAgents**)，且都不覆盖 WAM/DiT 流。工业栈 (GR00T / openpi / LeRobot) 全是"单 policy WebSocket"级别，无 batcher/scheduler。
> **方法**: subagent 多源调研 → arXiv ID 主进程 `curl` 逐条校验 → 18/18 命中，零幻觉
> **前置**: `multimodal-serving-systems-2026.md` (通用层已关闭)，`hao-style-synthesis.md` 候选 D'，`exp/exp08a/FINDINGS.md`

---

## 1. 真正的 VLA-dedicated serving (Section A)

| 系统 | arXiv | 月份 | 定位 | 覆盖面 | 对 exp08 的价值 |
|------|-------|------|------|--------|----------------|
| **OxyGen** | 2603.14371 | 2026-03 | VLA 版 vLLM 雏形——continuous batching + 跨任务 **KV shared**, MoT 架构 (π₀.₅)，3.7× throughput，200 tok/s + 70 Hz 并发 | **仅 KV + batching** 一层 | **最硬核的 VLA serving 论文**，离"DistServe of VLA"最近。是候选 D' 的天然 baseline |
| **VLAgents** | 2601.11250 | 2026-01 | Policy server 协议层——Gymnasium-style 统一接口 + zero-copy shared memory (sim) / compressed streaming (remote)，声称优于 OpenVLA/openpi/LeRobot 自带 server | 协议 + 传输层 | 偏"VLA 的 torchserve"，不做 scheduling/batching 深度 |

**两者都不覆盖的空白** (exp08 真可做的事):
- **Streaming VAE for video-WAM serving** (DreamZero/NitroGen 流)
- **Action-goodput SLO** (类 DistServe TTFT/TPOT)
- **VLA 的 PD / EPD / EPDA disaggregation**
- **Speculative action rollout** (world model 预测 + 动作投机)

---

## 2. 邻接但不符合 "VLA serving system" 定义 (Section B)

### 2.1 算法/模型侧加速——不是 serving system

| 系统 | arXiv | 为什么不算 |
|------|-------|-----------|
| Fast-WAM | 2603.16666 | Skip-imagination 是 model redesign，不是 scheduler |
| FASTER | 2603.19199 | "Rethinking Real-Time Flow VLAs" — model-level |
| StreamingVLA | 2603.28565 | Streaming + action flow matching 仍是 model arch |
| SnapFlow | 2604.05656 | One-step flow 蒸馏 |
| A1 | 2604.05672 | Truncated VLA, open-source efficient model |
| NanoVLA | 2510.25122 | Routing decoupled understanding，nano-size |
| PD-VLA | 2503.02310 | Parallel decoding + action chunking |
| Discrete Diffusion VLA | 2508.20072 | Discrete diffusion for action decoding |
| OpenVLA-OFT | 2502.19645 | Fine-tuning speed/success 优化 |

### 2.2 Edge-cloud partitioning——层级太浅

| 系统 | arXiv | 为什么不算 |
|------|-------|-----------|
| RoboECC | 2603.20711 | 多因素 edge-cloud 协同部署策略，无 runtime scheduler/batcher |
| RAPID | 2603.07949 | Redundancy-aware 切割点选择，偏 placement 不做 serving |

### 2.3 Performance / benchmark——未来系统的 motivation material

| 系统 | arXiv | 用途 |
|------|-------|------|
| VLA-Perf | 2602.18397 | "How Fast Can I Run My VLA?" 性能建模 (motivation figure) |
| Characterizing VLA on Edge | 2603.02271 | Jetson Orin/Thor 特征——action 阶段 dominate 的首手证据 |
| Embodied Foundation Models at the Edge (Survey) | 2603.16952 | Deployment constraints 综述 |
| Survey on Efficient VLAs | 2510.24795 | 效率方法综述，包括 quantization/pruning/distillation |
| vla-eval | 2603.13966 | Evaluation harness (WebSocket + Docker)，偏 eval 工具 |

### 2.4 工业栈——单 policy endpoint 级别

| 仓库 | URL | 实际做的事 | 距 serving system 差距 |
|------|-----|----------|----------------------|
| NVIDIA Isaac-GR00T | github.com/NVIDIA/Isaac-GR00T | ZMQ policy server + TensorRT export | **无 batcher**，无 multi-client scheduler |
| Physical Intelligence openpi | github.com/Physical-Intelligence/openpi | `scripts/serve_policy.py` websocket 单进程 | 同上 |
| HuggingFace LeRobot | github.com/huggingface/lerobot | `serve_policy` websocket | 同上 |

### 2.5 Hao AI Lab (UCSD) 自身

2025-2026 publications list **不含 VLA/robotics 方向**——全部是 LLM serving (vLLM)、video diffusion (FastVideo)、diffusion LLM。DistServe 和 FastVideo 是 **方法论 source**，不是 VLA serving system。

---

## 3. 空白图 (Section C)

以"VLA/WAM closed-loop serving"为 first-class citizen 排一下成熟度：

```
成熟 ├──────────────────────────────── 空白
     │
     │  OxyGen (KV + batching)       ← 2026-03, 覆盖 1/5
     │  VLAgents (protocol)          ← 2026-01, 覆盖 1/5
     │
     │  [ 以下全部空白 ]
     │  ❌ VLA 的 PD/EPD/EPDA disagg
     │  ❌ Streaming VAE for video-WAM
     │  ❌ Action-goodput SLO
     │  ❌ Speculative action rollout
     │  ❌ Multi-robot cluster scheduling
     │  ❌ WAM-aware DiT caching serving
     │
     └────────────────────────────────►
         几乎没有 "DistServe of VLA" 级别的完整 system
```

**两个观察**:
1. **OxyGen 是 genuinely 先手工作** — 离"DistServe of VLA"最近但只做 KV 一层
2. **Hao 本人没做 VLA serving** — 这是项目"向 Hao 方向靠拢"的结构性机会（他已经做了 LLM+video 两个 serving 系统，VLA 是 next natural step）

---

## 4. 对 v0.8.0 exp08 scope 的补充

v0.8.0 已经识别了 B1-B4 四个真空白（VLA SLO benchmark / kernel contention model / FastVideo 迁 VLA / Visual KV 压缩）。本次调研补充两项：

- **B5: Streaming VAE WAM serving** — 对标 FastVideo 但目标是 video-WAM (LingBot-VA / Fast-WAM / DreamZero 流)。exp04b canonical (2518ms, 0.40Hz) 是天然 starting point
- **B6: 扩展 OxyGen 到 PD 维度** — OxyGen 只做 KV shared，没做 prefill-decode 分离；可以以它为 baseline, 加 DistServe 思想做 VLA-PD
- **B2 重新表述**: GPU kernel-level contention model 现在有了 **OxyGen 作为 reference implementation** (它的 continuous batching 已经在做 kernel 级调度优化)——mechanism study 可以直接和 OxyGen runtime 的实测数据对齐

**方向建议**: 候选 D' 的 related-work 段落应显式承认 OxyGen 是 concurrent work, 并定位 exp08 为"对 OxyGen 没覆盖的 WAM/PD/SLO 层做补充"。

---

## 5. 引用清单 (全部通过 `curl arxiv.org/abs/{id}` 校验，18/18 命中)

### Section A (VLA-dedicated serving)
- **OxyGen** — 2603.14371 — https://arxiv.org/abs/2603.14371 — "Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism"
- **VLAgents** — 2601.11250 — https://arxiv.org/abs/2601.11250 — "A Policy Server for Efficient VLA Inference"

### Section B.1 (algorithmic acceleration, not serving)
- Fast-WAM — 2603.16666 — https://arxiv.org/abs/2603.16666
- FASTER — 2603.19199 — https://arxiv.org/abs/2603.19199
- StreamingVLA — 2603.28565 — https://arxiv.org/abs/2603.28565
- SnapFlow — 2604.05656 — https://arxiv.org/abs/2604.05656
- A1 — 2604.05672 — https://arxiv.org/abs/2604.05672
- NanoVLA — 2510.25122 — https://arxiv.org/abs/2510.25122
- PD-VLA — 2503.02310 — https://arxiv.org/abs/2503.02310
- Discrete Diffusion VLA — 2508.20072 — https://arxiv.org/abs/2508.20072
- OpenVLA-OFT — 2502.19645 — https://arxiv.org/abs/2502.19645

### Section B.2 (edge-cloud partitioning)
- RoboECC — 2603.20711 — https://arxiv.org/abs/2603.20711
- RAPID — 2603.07949 — https://arxiv.org/abs/2603.07949

### Section B.3 (performance & surveys)
- VLA-Perf — 2602.18397 — https://arxiv.org/abs/2602.18397
- Characterizing VLA on Edge — 2603.02271 — https://arxiv.org/abs/2603.02271
- Embodied Foundation Models at Edge — 2603.16952 — https://arxiv.org/abs/2603.16952
- Survey on Efficient VLAs — 2510.24795 — https://arxiv.org/abs/2510.24795
- vla-eval — 2603.13966 — https://arxiv.org/abs/2603.13966

### Section B.4 (industrial stacks)
- NVIDIA Isaac-GR00T — https://github.com/NVIDIA/Isaac-GR00T
- Physical Intelligence openpi — https://github.com/Physical-Intelligence/openpi
- HuggingFace LeRobot — https://github.com/huggingface/lerobot

### Section B.5 (Hao AI Lab)
- Hao AI Lab Publications — https://haoailab.com/publications/

---

*2026-04-27. 调研方法: general-purpose subagent + 主进程 arXiv ID 校验。所有 arXiv ID 经 `curl arxiv.org/abs/{id}` 验证，18/18 title 与描述一致。*
