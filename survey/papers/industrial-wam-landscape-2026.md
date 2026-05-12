# Industrial WAM Landscape (2024-2026)

> **核心问题**: 哪些工业机构真正在做 **runtime video generation → action**（而不是把 video WM 用于 pretrain/sim）？这条"video-at-inference"的技术路线是 VLA lineage 的对立面，与我们 exp04a/04b 测过的 FastWAM/LingBot-VA 同派系。
> **日期**: 2026-04-27
> **方法**: 主进程 `curl` 官方 blog 原文（1X / Rhoda / GAIA / Genie / World Labs / NVIDIA GEAR / PI / Generalist）+ subagent 扩展 + 关键引用全部 `curl` 反验
> **关联**: `survey/papers/pi-series-evolution.md` (VLA lineage), `survey/papers/vla-wam-serving-systems-2026.md`, `survey/papers/dreamdojo-dreamzero-deep-dive.md`

---

## 0. 分类准则（硬性二分）

**Binary**: 模型是否在**推理时生成像素**以决定动作？

- ✅ **Runtime video-gen WAM**: 推理时 forward 一次会产生 video / 未来帧，动作基于此
- ❌ **Video-pretrain + VLA inference**: video 只在预训练/仿真阶段出现，推理时是普通 VLA forward

这个区分直接决定了 serving 层的 latency/资源特性——前者 A 阶段是 **video denoise 秒级**（exp04b 实测 V=697ms），后者 A 阶段是 **action head 毫秒-数十毫秒**（exp07a 164ms）。

---

## 1. ✅ Runtime video-gen WAM（少数派）

### 1.1 1X Technologies — **1XWM** ⭐ 旗舰

- **URL**: https://www.1x.tech/discover/1x-world-model (2026-01-12, AI Team)
- **原话（逐字）**:
  > *"While VLAs directly predict action trajectories from static image-language input, our world-model based policy **derives robot actions from text-conditioned video generation**."*
  > *"From the same starting image sequence, our world model can **imagine multiple futures** from different robot action proposals."*
- **架构**: Video-pretrained world model，集成进 NEO 机器人做 policy。显式批评 VLA 路线："VLMs benefit from internet-scale knowledge, but are trained on objectives that emphasize visual and semantic understanding over prediction of physical dynamics"
- **关键差异**: 不依赖大规模 robot data 或 teleop demonstration，靠"internet-scale video + human-like embodiment"
- **阵型对比**: 显式列出对立面 `PI0.6 / Helix / Groot N1.5` 为 VLA 派
- **对我们项目**: **这是工业界最明确 runtime-video-gen 的 case**。候选 D' SLO benchmark 应把 "1XWM-style" 作为 worst-case tier，推理时 video gen 的 latency/显存特征与 exp04b 实测一致

### 1.2 Rhoda AI — **DVA (Direct Video-Action Model)** ⭐ 技术细节已公开

- **URL**: https://www.rhoda.ai/research/direct-video-action (2026-03 tech blog) + https://www.rhoda.ai/news (stealth exit 2026-03-10)
- **深读**: `survey/papers/rhoda-dva-generalist-gen1-deep-dive.md`
- **原话（逐字反验）**:
  > *"To the best of our knowledge, our model is the first to **pre-train a causal video model from scratch** and also the first to **perform full video denoising during real-time closed-loop robot control**."*
  > *"Non-causal video-to-action translation is a much more constrained problem. The complex decision making has already been handled at the video generation stage."*
- **架构 (三件套)**: Causal Video Model (from scratch, Context Amortization 训练) → Inverse Dynamics Model (~10hr data, 极小) → Actions。推理时做完整 video denoising。
- **Leapfrog Inference**: predict horizon > inference latency → overlap model inference & action execution。Action conditioning 保轨迹连续。
- **部署**: Decanting (11hr data, 1.5hr 无干预, bimanual 10kg) + Container Breakdown (17hr data, 160min 无干预, 50lbs)。长 context memory (数百帧), one-shot demo following。
- **融资**: $450M Series A, Bloomberg 报道 $1.7B 估值（2026-03-10）
- **仍未报**: model size, Hz, latency 具体数字。只说 "multiple times per second"
- **分类理由**: 推理时完整 video denoising + IDM action translation = 最纯粹的 runtime video-gen WAM

### 1.3 NVIDIA GEAR — **DreamZero**

- 已有深读: `survey/papers/dreamdojo-dreamzero-deep-dive.md`
- Wan2.1 14B DiT 推理时运行，~7Hz on GB200

### 1.4 World Labs — **RTFM (Real-Time Frame Model)**

- **URL**: https://www.worldlabs.ai/blog (2025-10-16)
- **描述**: "generates video in real-time as you interact with it"
- **注意**: 定位是**空间/创作 product**，不是 robotics，但技术 sibling（Marble WM 背后的 interactive generation engine）
- **相关旗舰**: Marble (2025-11, multimodal WM)，Spark 2.0 (2026-04-14, streamable 3DGS)

---

## 2. ❌ Video-pretrain + VLA-inference（主流多数）

| 公司 | 旗舰 | 推理架构 | Video 在哪用 | 源 |
|------|------|---------|-------------|-----|
| Physical Intelligence | π0 / π0.5 / π*0.6 / **π0.7** | VLA (+ 轻量 subgoal image gen for π0.7) | Co-training 数据源 | `pi-series-evolution.md` |
| Generalist AI | **GEN-1** | VLA system with "inference-time techniques"；50 万小时 human wearable pretrain | Pretrain only | generalistai.com |
| Figure.ai | **Helix** | VLA S1/S2 (7-9Hz VLM + 200Hz visuomotor)，**推理无 video gen** | N/A | figure.ai/news/helix |
| Google DeepMind | **Gemini Robotics 1.5** | VLA（Genie 3 是独立 sim 产品） | N/A | deepmind.google |
| Wayve | AD policy + **GAIA-1/GAIA-2** | 自动驾驶端到端 policy 和 GAIA world model 是两条线 | Sim/eval | wayve.ai/thinking/gaia-2 |
| NVIDIA | **Cosmos** WFM (14B/13B/7B) + **GR00T N1/N1.5** | Cosmos 是 pretrain/sim/data-aug，GR00T 是 VLA | Pretrain + data aug | research.nvidia.com/labs/gear |
| Google DeepMind | **Genie 3** | Interactive world gen，非 robotics closed-loop | Product (games/sim) | deepmind.google |

**关键观察**: **VLA 派（PI, Figure, Google, PI, Generalist）明确不做推理时 video gen**。Wayve 和 NVIDIA 有 WM 但只用在 pretrain/sim。

---

## 3. 信息缺失 / 未确认

| 目标 | 问题 | 当前证据 |
|------|------|---------|
| **Tesla FSD** | 推理时是否有 world model video-gen？ | Elon tweets 提到 "world model"，但 AI Day'22/'23 只描述 occupancy nets + planner。**无 primary source 证实 runtime video-gen** — 按规则判定为 "无证据，不采信" |
| **Unitree 宇树** | 有 WM 吗？ | 官网 + arXiv 无，公开出货是 RL-control policy。 `UnifoLM-WMA` 为社区名称，未找到 Unitree primary publication |
| **Skild AI** | 架构？ | 官网 blog 404；公开表述"general brain"，无 WM 细节 |
| **Sanctuary / Agility / Apptronik** | 均只有 homepage，无架构 disclosure |
| **RAI Institute (ex-BD)** | 无 WAM architecture 公开 |
| **中国工业界** (星海图 Galbot / 银河通用 / 智元 AgiBot / 星动纪元) | 公开材料显示 **全是 VLA/diffusion policy 路线**，无匹配 1XWM/Rhoda 的 runtime-video-gen 旗舰 | —— |
| **智源 BAAI RoboBrain** | VLM-for-robotics，无 runtime video WM 证据 |

---

## 4. 工业版图的三条观察

### 4.1 "Runtime video-gen WAM" 是**少数派 + 美国/NVIDIA 主导**
截至 2026-04，明确 claim 推理时 video-gen 做 action 的工业旗舰只有 4 家：**1X, Rhoda, NVIDIA DreamZero, World Labs RTFM**（最后一家非 robotics）。**中国无匹配旗舰**，主流走 VLA/diffusion policy 路线。

### 4.2 PI / Generalist / Figure 三大 VLA 阵营 vs 1X / Rhoda 两大 WAM 阵营，**2026-Q1 开始旗帜鲜明地对立**
1XWM blog 原文点名 "PI0.6, Helix, Groot N1.5" 作为"VLA 路线"的对立面。Rhoda 说 "conventional VLA pipelines struggle to achieve"（官网原话）。这是**商业信念层面的分野**，不是单纯的学术偏好。

### 4.3 **Serving 层代价差异极大，工业分野完全对齐 exp04b 实测**
| 路线 | Action phase 实测（RTX 5880 Ada） | 典型 Hz |
|------|---------------------------------|---------|
| VLA (Pi-Zero, exp07a) | A=165ms (stable) | ~5 |
| WAM-runtime-gen (LingBot-VA full, exp04b) | V=697ms + A=1708ms | **0.4** |

**0.4Hz 的 gap 是 VLA vs WAM 路线选择的本质原因**——工业大多数走 VLA，就是因为 WAM 推理代价还没压到 real-time 水位。DreamZero 7Hz on GB200 是目前最接近阈值的，但需要 H200/B200 级硬件。

---

## 5. 对项目 exp08 / 候选 D' 的行动含义

1. **候选 D' 的 SLO benchmark 必须同时包含两派**:
   - VLA tier (π0 / π0.7 pipeline / Helix / GEN-1) — 主流，majority weight
   - WAM tier (1XWM / Rhoda / DreamZero) — 少数但 latency tail 极长，是 SLO-violation 最严重的 case
2. **1XWM / Rhoda 若开源，是 exp04b 后下一个 canonical 数据点**。目前只有 WAM 学术模型（Fast-WAM / LingBot-VA）能跑
3. **exp04b 的 0.4Hz 不再孤立**——1XWM / Rhoda 正在把这条路线商业化，只是工程上用更大的硬件（H100/B200/GB200）把 Hz 顶到实用区间。这给我们的 exp04b 数据新的 relevance
4. **"Model is a system" 对 VLA 派成立（π0.7 + GEN-1），对 WAM 派同样成立**：1XWM 是 "video gen model + action decoder + history memory" 的复合 system，不是单 model。两条 lineage 的 serving 复杂度都在上升

---

## 6. 引用清单（primary source）

### Runtime video-gen WAM
- 1X Technologies 1XWM — https://www.1x.tech/discover/1x-world-model (2026-01-12)
- 1X World Model (origin post) — https://www.1x.tech/discover/world-model-self-learning (2024-09-17, Monas/Jang)
- Rhoda AI — https://www.rhoda.ai/ + https://www.rhoda.ai/news (2026-03-10 stealth exit, $450M Series A, $1.7B val per Bloomberg)
- NVIDIA DreamZero (GEAR) — `dreamdojo-dreamzero-deep-dive.md`
- World Labs RTFM — https://www.worldlabs.ai/blog (2025-10-16)
- World Labs Marble — https://www.worldlabs.ai/blog (2025-11-12)

### Video-pretrain + VLA-inference
- PI π series — `pi-series-evolution.md`
- Generalist GEN-1 — https://generalistai.com/blog/apr-02-2026-GEN-1
- Figure Helix — https://www.figure.ai/news/helix
- Wayve GAIA-2 — https://wayve.ai/thinking/gaia-2/
- Google Genie 3 + Gemini Robotics 1.5 — https://deepmind.google/research/
- NVIDIA Cosmos — https://github.com/NVIDIA/Cosmos + https://research.nvidia.com/labs/gear/
- NVIDIA GR00T N1.5 — research.nvidia.com/labs/gear

### Unverified / no evidence
- Tesla FSD runtime WM: no primary source
- Unitree WM: no primary source
- Skild / Sanctuary / Agility / Apptronik / RAI: homepage-level only

---

*2026-04-27. 主进程 curl + subagent 复核 + 关键引用反验（Rhoda "video-predictive control" / 1XWM "text-conditioned video generation" 均逐字确认）。Tesla / Chinese industry 方向保守判定为"无证据"。*
