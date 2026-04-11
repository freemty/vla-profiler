# VA (Vision-Action) 与 World Action Model 深度 Survey

**编写日期:** 2026-04-11
**研究背景:** UCSD 张昊实验室 -- VLM/VLA Real-Time Systems
**定位:** 补充 landscape.md 中 VA 与 World Model 方向的深度分析，聚焦 inference efficiency 与 systems 视角

---

## 目录

1. [VA (Vision-Action) 模型全景](#1-va-vision-action-模型全景)
2. [World Action Model / World Model for Action](#2-world-action-model--world-model-for-action)
3. [World Model for VLA -- 融合路线](#3-world-model-for-vla----融合路线)
4. [Inference Efficiency 对比分析](#4-inference-efficiency-对比分析)
5. [Systems 视角分析](#5-systems-视角分析)
6. [与张昊实验室技术栈的迁移分析](#6-与张昊实验室技术栈的迁移分析)
7. [Research Gaps 与机会](#7-research-gaps-与机会)

---

## 1. VA (Vision-Action) 模型全景

### 1.1 定义与核心设计哲学

VA (Vision-Action) 模型是直接从视觉观测映射到动作空间的策略网络，**不经过自然语言中间表示**。其设计哲学是：对于很多机器人操作任务，language conditioning 是不必要的开销 -- 人类工人在执行重复性操作时也不需要语言指令。

VA 模型的核心特征：
- **无 LLM backbone：** 不包含大语言模型，参数量通常在 10M - 1B 范围
- **端到端 visuomotor：** 从像素/点云直接到关节角度/末端执行器位姿
- **任务指定方式：** 通过 goal image、task embedding、demo conditioning 而非 language instruction
- **Inference 极快：** 5-50ms 的典型延迟，比 VLA 快 10-100x

### 1.2 主要架构族谱

#### 1.2.1 Diffusion Policy 系列

**Diffusion Policy (Chi et al., 2023)** (arXiv: 2303.04137) 是 VA 领域的里程碑工作。核心思想：将 behavior cloning 问题建模为条件去噪扩散过程，从噪声中逐步生成 action trajectory。

**架构：**
- **条件编码器：** ResNet 或 ViT 处理图像观测
- **去噪网络：** U-Net (CNN-based) 或 DiT (Transformer-based)
- **动作表示：** 连续 action chunk (通常 T=16 步)
- **Training：** DDPM loss，预测噪声 epsilon
- **Inference：** DDPM/DDIM 去噪 (原始 100 步，后续优化到 10-1 步)

**Inference 特征：**
- 原始 DDPM：100 步去噪 x ~2ms/step = ~200ms (在 GPU 上)
- DDIM 采样：10-20 步 = 20-40ms
- 但 action chunking (T=16) 使得等效控制频率 = 1/(inference_time/T)

**关键后续改进：**

| 模型 | 年份 | 核心改进 | Inference 延迟 | arXiv |
|------|------|---------|--------------|-------|
| Diffusion Policy (原始) | 2023 | DDPM action generation | ~200ms (100步) | 2303.04137 |
| DP (DDIM) | 2023 | DDIM 加速采样 | ~30ms (10步) | 同上 |
| FlowPolicy | 2024 | Consistency flow matching | ~5-10ms (1-2步) | 2412.04987 |
| DM1 | 2025 | MeanFlow + dispersive reg. | ~5ms (1步) | 2510.07865 |
| ManiFlow | 2025 | Consistency flow training | ~5ms (1-2步) | 2509.01819 |
| Falcon | 2025 | Partial denoising + action reuse | 显著降低 | 2503.00339 |
| Sparse ActionGen | 2026 | Real-time pruning of diffusion | 25% 计算量 | 2601.12894 |
| Action-to-Action Flow | 2026 | Informed initialization | **0.56ms** (1步) | 2602.07322 |
| One-Step Flow Policy | 2026 | Self-distillation for 1-step | ~3ms (1步) | - |
| Ada3Drift | 2026 | Adaptive training-time drifting | 1步 | - |

**趋势总结：** 从 2023 年的 100-step DDPM 到 2026 年的 sub-1ms single-step flow，Diffusion Policy 系列的 inference 延迟压缩了 **200-400x**。这是一个极其重要的进展 -- 它意味着 diffusion-based VA 模型的延迟已经完全能满足高频控制的需求 (>100 Hz)。

#### 1.2.2 3D Diffusion Policy (DP3) 系列

**3D Diffusion Policy (Ze et al., 2024)** (arXiv: 2403.03954) 将 diffusion policy 扩展到 3D 点云输入，核心动机是 2D 图像缺少深度信息导致操作精度受限。

**架构：**
- **3D 编码器：** PointNet++ / Point Transformer 处理点云
- **去噪网络：** MLP 或 DiT，条件为 3D feature
- **优势：** 对视角变化、光照变化更鲁棒；空间推理更精确

**关键后续：**

| 模型 | 年份 | 特点 | arXiv |
|------|------|------|-------|
| DP3 | 2024 | 原始 3D diffusion policy | 2403.03954 |
| 3D Flow Diffusion Policy | 2025 | 结合 3D flow 与 diffusion | - |
| EfficientFlow (E3Flow) | 2026 | SE(3)-equivariant flow policy | - |
| PocketDP3 | 2026 | 极致压缩的 pocket-scale 3D policy | - |
| ISS Policy | 2025 | Implicit scene supervision for 3D | - |

**Inference 特征：** 3D policy 额外引入点云预处理开销 (~5-15ms for depth → point cloud + FPS sampling)。总延迟通常比 2D diffusion policy 高 1.5-2x，但 action 精度显著提升。

**与 systems 的关系：** 3D policy 的 bottleneck 在点云处理而非去噪过程。深度传感器的帧率 (通常 30 Hz) 和点云降采样是硬件层面的限制。

#### 1.2.3 ACT (Action Chunking with Transformers)

**ACT (Zhao et al., 2023)** (arXiv: 2304.13705) 是另一条重要的 VA 技术路线，使用 CVAE (Conditional Variational Autoencoder) + Transformer 架构。

**架构：**
- **编码器：** ResNet backbone 提取图像特征
- **CVAE encoder (训练时)：** 将 action sequence 编码为 latent z
- **CVAE decoder：** Transformer decoder，以 visual features + z 为条件生成 action chunk
- **Inference 时：** z 从 prior (标准正态) 采样，单次 forward pass 生成 action chunk

**Inference 特征：**
- **核心优势：** 单次 forward pass，无需迭代去噪
- 典型延迟：10-30ms (取决于 backbone 和 chunk size)
- 比 diffusion policy 的 inference 简单得多
- 但 action 质量在多模态分布上劣于 diffusion (CVAE 的 mode collapse 问题)

**关键后续：**

| 模型 | 年份 | 改进 | arXiv |
|------|------|------|-------|
| ACT (原始) | 2023 | CVAE + Transformer | 2304.13705 |
| MoE-ACT | 2026 | Mixture-of-Experts 扩展多任务 | - |
| ConceptACT | 2026 | Episode-level concepts | - |
| FTACT | 2025 | Force/torque aware | - |
| Haptic-ACT | 2024 | 触觉感知 | - |

**ACT vs Diffusion Policy 的系统性对比：**
- ACT 胜在 inference simplicity (单次 forward)，适合极低延迟场景
- Diffusion Policy 胜在 multi-modal action distribution 建模，适合复杂操作
- 随着 1-step flow 的成熟，两者的 inference 延迟差距已基本消失
- ACT 的 CVAE 框架天然支持 uncertainty estimation (通过 z 的方差)

#### 1.2.4 Behavior Transformer (BeT)

**BeT (Shafiullah et al., 2022)** (arXiv: 2206.11251) 使用 GPT-style transformer 做 autoregressive action prediction。

**架构：**
- **离散化：** 将 action space 通过 k-means clustering 离散化为 action bins
- **Backbone：** Minkowski GPT (GPT-2 variant)
- **生成：** Autoregressive，先预测 action bin index，再预测 bin 内 offset

**Inference 特征：**
- Autoregressive 但维度远低于 VLA (不包含 language tokens)
- 离散化 bins 通常只有 256-1024 个
- 单步延迟约 5-20ms
- 缺点：离散化引入 quantization error (与 VLA 的 token 化 action 类似)

#### 1.2.5 其他重要 VA 架构

| 类型 | 代表模型 | 核心思想 | Inference 特征 |
|------|---------|---------|---------------|
| Normalizing Flows | Flow Policy (Li et al., 2025) | 可逆变换，精确 likelihood | 单次 forward，~5ms |
| Masked Generation | MGP (Zhuang et al., 2025) | Masked token prediction | 并行 decode，~10ms |
| Koopman Operator | Koopman Policy (Han et al., 2026) | 线性化动力学 | 超快 (~1ms) |
| Hybrid Open-loop | Hybrid-Diffusion (2025) | Diffusion + open-loop routines | 分情况 |
| State Space | AnoleVLA (2026) | Mamba-based，线性复杂度 | O(n) inference | 2603.15046 |

### 1.3 VA 模型的 Inference Pipeline 解析

典型的 VA inference pipeline 分为三个阶段：

```
[Camera Input] → [Vision Encoding] → [Action Generation] → [Action Execution]
    ~5ms            ~5-20ms            ~1-200ms (变化大)     ~5ms (controller)
```

**阶段 1: Vision Encoding**
- 2D Policy: ResNet (~5ms), ViT (~10-15ms)
- 3D Policy: Depth→PointCloud (~5ms) + PointNet++ (~10ms)
- 这个阶段在 VA 和 VLA 中差异不大

**阶段 2: Action Generation (核心差异)**
- Single-pass 方法 (ACT, BeT): 单次 forward, ~5-20ms
- Diffusion (100 步): ~200ms -- 太慢
- Flow matching (1 步): ~1-5ms -- 已解决
- **关键洞察：** Action generation 不再是 bottleneck (对于 1-step flow)

**阶段 3: Action Execution**
- Action chunk 通过 interpolation 发送给 low-level controller
- 控制器通常运行在 1kHz (1ms)，远快于 policy inference
- Action chunking 使得 policy 可以以低频运行而不影响 smooth 控制

### 1.4 VA 的适用边界：何时够用，何时需要 VLA？

这是一个实践中极其重要但文献中讨论不足的问题。

**VA 足够的场景：**
1. **固定任务库：** 工厂产线上的重复性操作 (pick-and-place, assembly)
2. **Goal-conditioned 操作：** 给定 goal image 即可定义任务
3. **高频控制任务：** 灵巧手操作、接触丰富的操作 (latency < 10ms 是必须的)
4. **数据充足的窄领域：** 有足够的 demo 数据覆盖 task distribution
5. **Edge 部署：** 计算资源受限，无法承载 LLM backbone

**必须用 VLA 的场景：**
1. **开放指令：** 用户可以用自然语言描述任意任务
2. **Multi-step reasoning：** "先把碗放到桌子上，然后把勺子放进碗里"
3. **Semantic grounding：** "把红色的杯子递给我" (需要 language-vision grounding)
4. **Zero-shot / Few-shot adaptation：** 遇到训练时没见过的任务
5. **Cross-embodiment transfer：** 一个模型用于多种 robot 形态

**"灰色地带" -- VA 和 VLA 之间：**
- 带简单 task embedding 的 VA (如 one-hot conditioning) 可以处理有限的多任务场景
- 带 CLIP 特征 conditioning 的 VA 可以做一定程度的 language grounding，但远不及 VLA
- Dual-system 架构 (GR00T N1, Dual Process VLA) 将 VLA 的 reasoning 与 VA 的 low-level control 解耦

---

## 2. World Action Model / World Model for Action

### 2.1 概念厘清

"World Model" 在 robotics 语境下有多个层次的含义，必须严格区分：

```
World Model 谱系
|
|-- Level 0: 物理引擎 (MuJoCo, Isaac Sim)
|   |-- 精确但需要完整场景建模，非学习型
|
|-- Level 1: Latent Dynamics Model (Dreamer 系列)
|   |-- 学习 state→state 的隐空间转移
|   |-- 用于 model-based RL 的 imagination
|
|-- Level 2: Video World Model (GAIA-1, Genie, UniSim)
|   |-- 学习 video prediction as world simulation
|   |-- 用于 visual planning 和 data augmentation
|
|-- Level 3: World Action Model (DreamZero, Cosmos Policy)
|   |-- 直接将 world model 与 action generation 统一
|   |-- "想象未来" 和 "决定行动" 在一个框架中
|
|-- Level 4: VLM as World Model (LeCun's JEPA vision)
|   |-- 大规模预训练的 VLM 隐式包含 world knowledge
|   |-- 通过 prompting/fine-tuning 提取 action
```

### 2.2 Level 1: Latent Dynamics Model

#### Dreamer 系列 (Hafner et al.)

这是 model-based RL 中最有影响力的 world model 系列。

| 模型 | 年份 | 核心创新 | Inference 特点 | 论文 |
|------|------|---------|--------------|------|
| Dreamer (v1) | 2020 | RSSM (Recurrent State Space Model) | Latent rollout ~0.5ms/step | arXiv: 1912.01603 |
| Dreamer v2 | 2021 | Discrete latent + KL balancing | 更稳定的 rollout | arXiv: 2010.02193 |
| Dreamer v3 | 2023 | Symlog predictions, fixed hyperparams | 跨域通用 | arXiv: 2301.04104 |
| DayDreamer | 2022 | Dreamer 在真实机器人上的应用 | 同 Dreamer v2 | arXiv: 2206.14176 |

**RSSM 架构核心：**
```
Latent state: s_t = (h_t, z_t)
  h_t: deterministic recurrent state (GRU)
  z_t: stochastic discrete latent (32 x 32 categories)

Transition:   h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
Prior:        z_t ~ p(z_t | h_t)
Posterior:    z_t ~ q(z_t | h_t, o_t)
Decoder:      o_t ~ p(o_t | h_t, z_t)
Reward:       r_t ~ p(r_t | h_t, z_t)
```

**Inference Pipeline (对于 action selection)：**
1. **Encode 当前观测:** o_t → z_t (via posterior) -- ~1ms
2. **Imagination rollout:** 在 latent space 中展开 H 步 future -- ~0.5ms/step x H steps
3. **Action selection:** Actor-critic 在 imagined trajectory 上选 action -- ~1ms
4. **总延迟:** ~1 + 0.5*H + 1 ms。当 H=15，约 10ms

**核心 efficiency 优势：** Latent rollout 不需要 decode 到 pixel space (除非需要可视化)，因此每步只需 GRU forward + MLP，极快。

**DayDreamer (2022, arXiv: 2206.14176) 的真实世界验证：**
- 在 A1 四足机器人上实现了 online learning from scratch
- 仅 1 小时的真实世界交互就学会了 locomotion
- 验证了 latent world model 在真实机器人上的可行性
- **但:** 限于 low-dimensional 状态空间和简单 action space

#### DIAMOND (Alonso et al., 2024)

**DIAMOND** (arXiv: 2405.12399) 使用 diffusion model 作为 world model，在 Atari 上达到人类水平。

- **创新：** 用 diffusion 建模 observation transition (而非 latent transition)
- **代价：** 每个 world model step 需要完整的 diffusion sampling (~100ms)
- **意义：** 证明了 diffusion 可以做高质量 world simulation
- **限制：** Inference 延迟太高，不适合 real-time 控制

### 2.3 Level 2: Video World Model

#### 关键工作

| 模型 | 年份 | 团队 | 核心思想 | 规模 | 论文 |
|------|------|------|---------|------|------|
| GAIA-1 | 2023 | Wayve | 自动驾驶 video world model | 6.5B | - |
| Genie | 2024 | DeepMind | Interactive world from video | 11B | arXiv: 2402.15391 |
| Genie 2 | 2024 | DeepMind | Scalable 3D world generation | 更大 | - |
| UniSim | 2023 | Google | Universal simulator via video | Large | arXiv: 2310.06680 |
| Cosmos | 2025 | NVIDIA | World foundation model | 多规模 | - |
| Sora | 2024 | OpenAI | General video generation | 未公开 | - |

**Video World Model 的 inference 特征：**

这些模型的 inference 极其昂贵，因为：
1. **生成分辨率高：** 通常 256x256 或更高，每帧数千到数万 token
2. **时间步多：** 预测 1-5 秒 future video = 30-150 frames
3. **生成过程本身需要迭代：** Diffusion-based 生成需要 20-100 步去噪
4. **总计算量：** 生成 1 秒 future video 可能需要 1-10 秒的 GPU 时间

**因此，Video World Model 直接用于 real-time 控制是不可行的。** 它们的价值在于：
- Training data augmentation (离线)
- 长期 planning (可以容忍高延迟)
- 评估/筛选 action plan 的质量 (imagine-then-select)

#### Genie (DeepMind, 2024, arXiv: 2402.15391)

Genie 值得特别关注，因为它提出了 **action-controllable video generation** 的概念：

- 从无标注视频中学习 latent action space
- 用户可以通过 latent action 控制生成的视频中的 agent 行为
- 这本质上是一个 **interactive world simulator**
- **推理时：** 给定当前帧 + latent action → 生成下一帧
- **延迟：** 单帧生成 ~50-200ms (取决于分辨率)

**对 robotics 的启示：** 如果能将 Genie 式的 interactive world model 与 low-level action decoder 结合，就得到一个可以 "在想象中规划" 的系统。

### 2.4 Level 3: World Action Model (WAM)

这是 2025-2026 年最令人兴奋的方向之一。

#### DreamZero (Ye et al., 2026, arXiv: 2602.15922)

**DreamZero** 是 World Action Model (WAM) 的代表工作，由 NVIDIA 主导。

**核心主张：** "World Action Models are Zero-shot Policies" -- 经过大规模视频数据预训练的 world model，可以直接作为 zero-shot policy 使用，无需任何 robot-specific fine-tuning。

**架构：**
1. **Video generation backbone：** DiT-based conditional video generator
2. **Action decoder：** 从生成的 video latent 中提取 action
3. **Pipeline：**
   - 输入当前观测 + task description
   - World model 生成未来 video (在 latent space)
   - Action decoder 将 video latent → robot action
4. **控制频率：** 7Hz 实时控制 (143ms per inference)

**Inference 分析：**
```
DreamZero Inference Pipeline:
[Observation encode]  →  [Video prediction (latent)]  →  [Action decode]
     ~10ms                    ~100-120ms                    ~10ms
                          (主要 bottleneck)
```

- Video prediction 是核心 bottleneck
- 7Hz 控制频率对于 tabletop manipulation 勉强够用
- 对灵巧手等高频控制场景仍然不足
- 但 zero-shot 能力是 revolutionary -- 不需要 robot data

**与 VA 的对比：**
- DreamZero 的 inference 比 1-step flow VA (~1ms) 慢 100x+
- 但 DreamZero 不需要 robot-specific training data
- 这是一个 **数据效率 vs 推理效率** 的 fundamental trade-off

#### Cosmos Policy (Kim et al., 2026)

**Cosmos Policy** 是 NVIDIA 在 DreamZero 路线上的进一步探索。

- 直接 fine-tune 预训练的 video model (Cosmos) 用于 visuomotor control
- **关键简化：** 不需要额外的 architectural components for action generation
- Video model 直接输出 future image，一个轻量的 action head 从 video latent 提取 action
- **强调：** Leveraging spatiotemporal priors from video pretraining

#### Motus (Bi et al., 2025)

**Motus** -- A Unified Latent Action World Model

- 将 world model 和 action prediction 统一在一个 latent space 中
- Latent action 从无标注视频中自动发现
- 用于 embodied agent 的 unified perception-action loop
- 比 DreamZero 更关注 latent space 的紧凑性

#### STORM (Lin et al., 2025)

**STORM** -- Search-Guided Generative World Models for Robotic Manipulation

- 结合 diffusion-based world model 与 search-guided planning
- 用 video prediction 提供 visual plan，再用 search 在 action space 中优化
- 解决了纯 video world model 难以精确控制的问题

#### ChronoDreamer (Zhou & Negrut, 2025)

**ChronoDreamer** -- Action-Conditioned World Model as an Online Simulator

- 将 world model 作为在线仿真器使用
- 支持 MPC (Model Predictive Control) 风格的实时规划
- 在 latent space 中做 rollout + CEM (Cross-Entropy Method) 优化

#### DDP-WM (Yin et al., 2026, arXiv: 2602.01780)

**DDP-WM** -- Disentangled Dynamics Prediction for Efficient World Models

这是从 **efficiency** 角度优化 world model 的重要工作。

- **核心思想：** 将 latent state 演化分解为 primary dynamics (物理交互驱动) 和 secondary dynamics (背景更新)
- **Primary dynamics：** 稀疏的关键动态，用高精度 attention 建模
- **Secondary dynamics：** 低频背景变化，用轻量 cross-attention 更新
- **效果：** 在 Push-T 任务上实现 **9x inference speedup**，MPC success rate 从 90% 提升到 98%
- **意义：** 证明了 world model 的 inference 可以通过动态分解大幅优化

#### Sparse Imagination (Chun et al., 2025, arXiv: 2506.01392)

**ICLR 2026 accepted** -- 这是另一个重要的 world model efficiency 工作。

- **核心：** 减少 latent rollout 中的 token 数量
- 使用 randomized grouped attention 训练稀疏 world model
- Inference 时灵活调整 token 数量以适应计算预算
- **适用范围：** 从简单 trajectory optimization 到复杂 real-world VLA 任务
- 使 world model 可以在 real-time 场景中部署

#### Act2Goal (Zhou et al., 2025)

- 从 world model 到 general goal-conditioned policy
- 使用 video prediction 作为 sub-goal planning
- 将 long-horizon 操作分解为 sub-goal sequence，每个 sub-goal 由 world model 生成

#### Dream2Flow (Dharmarajan et al., 2025)

- 桥接 video generation 和 open-world manipulation
- 从预训练 video model 中提取 3D object flow
- 作为 manipulation 的 visual plan

### 2.5 Level 1-3 的 Inference Pipeline 对比

```
Level 1 (Latent Dynamics -- Dreamer):
  Encode(5ms) → Latent Rollout(0.5ms x H) → Actor(1ms) = ~10-15ms
  适合: 30-100Hz 控制

Level 2 (Video World Model -- UniSim/Genie):
  Encode(10ms) → Video Generation(1-10s) → Action Extract(10ms) = ~1-10s
  适合: Planning (离线/准实时)

Level 3 (World Action Model -- DreamZero):
  Encode(10ms) → Latent Video Pred(100ms) → Action Decode(10ms) = ~130ms
  适合: 5-10Hz 控制

Level 3+ (Efficient WAM -- DDP-WM/Sparse Imagination):
  Encode(5ms) → Sparse Latent Rollout(10-30ms) → Action(2ms) = ~20-40ms
  适合: 20-50Hz 控制 (approaching real-time)
```

### 2.6 Dreamer v3 系列的独特价值

值得单独讨论 Dreamer v3 (Hafner et al., 2023, arXiv: 2301.04104) 的意义：

**为什么 Dreamer v3 重要：**
1. **通用性：** 同一套 hyperparameters 在 150+ tasks 上有效 (Atari, DMC, Minecraft, DMLab)
2. **数据效率：** 比 model-free RL 高 10-100x 的 sample efficiency
3. **Inference 极快：** Latent rollout 是 sub-ms 级的
4. **但：** 限于 low-dimensional observation/action (从 pixel 学习，但 action space 通常是 <20 维)

**对 VLA systems 的启示：**
- Dreamer 的 RSSM 思想可以用于 VLA 的 world knowledge component
- 但 Dreamer 不理解 language，不能处理 open-ended tasks
- **可能的融合方向：** VLM for understanding + Dreamer-style latent planning for action

---

## 3. World Model for VLA -- 融合路线

### 3.1 Video Model as VLA Backbone

这条路线的核心思想：预训练的 video generation model 天然包含 world dynamics knowledge，可以直接用于 robot control。

#### mimic-video (Pai et al., 2025, arXiv: 2512.15692)

- **Video-Action Model (VAM)：** 区别于 VLA，明确提出 VAM 概念
- 用预训练的 Internet-scale video model 的 latent representation
- Flow matching-based action decoder 作为 Inverse Dynamics Model (IDM)
- **效果：** Sample efficiency 提升 10x，convergence 提升 2x (vs VLA)
- **Inference：** Video model forward pass (~30ms) + action decoder (~5ms) = ~35ms

#### Vidarc (Feng et al., 2025)

- Embodied video diffusion model for closed-loop control
- 将 video diffusion model 直接优化为 embodiment-aware
- 解决了 Internet video model 对 robot 不 aware 的问题

#### CoVAR (Yang et al., 2025)

- **Co-generation of Video and Action：** 同时生成未来视频和动作
- Multi-modal diffusion 框架
- Video 和 action 在同一个 diffusion process 中生成
- 这避免了 "先生成 video，再从 video 提取 action" 的两阶段 pipeline

#### Large Video Planner (Chen et al., 2025)

- 用 large video model 做 generalizable robot control
- Video model 生成 visual plan (goal image sequence)
- Inverse dynamics model 将 visual plan → action sequence
- 强调 generalization：相同的 video planner 用于多种 robot 和任务

### 3.2 World Model 辅助的 Model-Based RL for Robotics

#### 经典路线：Dreamer → DayDreamer → Real-World MBRL

```
                      Sim                   Real
Dreamer v1 (2020) → PlaNet (2019) → DayDreamer (2022)
       ↓                                    ↓
Dreamer v2 (2021) ──────────────→ Dreamer Applied (various)
       ↓
Dreamer v3 (2023) ──────────────→ More robust real deployment
```

#### What Drives Success in Physical Planning with JEPA (Terver et al., 2025)

- 来自 Yann LeCun 组
- 研究 Joint-Embedding Predictive Architecture (JEPA) 在物理规划中的成功要素
- JEPA 是 LeCun 提出的 world model 框架的核心组件
- 发现 latent prediction (而非 pixel prediction) 是成功的关键

#### DreamTacVLA (Ye et al., 2025)

- "Learning to Feel the Future"
- 将 tactile prediction 作为 world model 的一部分
- VLA + tactile world model 用于 contact-rich manipulation
- 这是 world model 与 VLA 融合的一个有趣方向 -- 不是视觉 world model，而是触觉 world model

### 3.3 World Model + VLA 的架构模式

目前出现了三种融合模式：

**模式 A: World Model 作为 Data Augmentation**
```
[Video World Model] → 生成合成 robot 视频 → 训练 VLA
Offline 使用，不影响 inference 延迟
```
- UniSim, RoboVIP, AOMGen 采用这种模式
- 优势：VLA inference 完全不受影响
- 劣势：world model 的 dynamics knowledge 只通过数据间接传递

**模式 B: World Model 作为 Visual Planner**
```
[当前观测] → [VLM 理解任务] → [World Model 生成 sub-goal images] → [VA 执行每个 sub-goal]
```
- Large Video Planner, Dream2Flow, Act2Goal 采用这种模式
- 优势：分离了高层规划和低层控制
- 劣势：World model 的 visual planning 延迟可能高达数秒
- **Systems 问题：** Visual planner 可以 async 运行，pre-compute sub-goals

**模式 C: World Model 与 Action 统一生成**
```
[当前观测 + 任务] → [Unified WAM] → [Video latent + Action] 同时输出
```
- DreamZero, CoVAR, Cosmos Policy 采用这种模式
- 优势：端到端优化，world knowledge 和 action generation 深度耦合
- 劣势：Inference 最重 (video generation + action decode)
- **Systems 问题：** 无法将 world model 和 action head 独立调度

---

## 4. Inference Efficiency 对比分析

### 4.1 三大范式的延迟-能力 Pareto 前沿

| 范式 | 典型延迟 | 控制频率 | 泛化能力 | 推理需求 | 代表模型 |
|------|---------|---------|---------|---------|---------|
| **VA (1-step flow)** | 1-5 ms | >200 Hz | 低 (窄领域) | 单 GPU (小模型) | Action-to-Action Flow |
| **VA (10-step diffusion)** | 20-40 ms | 25-50 Hz | 低-中 | 单 GPU | Diffusion Policy (DDIM) |
| **VA (ACT)** | 10-30 ms | 30-100 Hz | 低-中 | 单 GPU | ACT, MoE-ACT |
| **VLA (autoregressive)** | 100-500 ms | 2-10 Hz | 高 | 多 GPU (7B+) | OpenVLA, RT-2 |
| **VLA (1-step flow)** | 50-100 ms | 10-20 Hz | 高 | 多 GPU | FASTER, Mean-Flow VLA |
| **WAM (DreamZero)** | ~130 ms | ~7 Hz | 极高 (zero-shot) | 多 GPU | DreamZero |
| **WAM (efficient)** | 20-40 ms | 25-50 Hz | 高 | 单/多 GPU | DDP-WM, Sparse Imagination |
| **Latent WM + MPC** | 10-15 ms | 60-100 Hz | 中 (需 online RL) | 单 GPU | Dreamer v3, DayDreamer |

### 4.2 Inference Breakdown -- Where is the Bottleneck?

**VA 模型：**
```
Vision Encoding: 5-15ms (30-70% of total for 1-step flow)
Action Generation: 1-5ms (1-step flow) | 20-40ms (10-step diffusion)
Post-processing: 1-2ms

Bottleneck: Vision encoding (当 action generation 优化到 1-step 后)
```

**VLA 模型 (autoregressive, 7B):**
```
Vision Encoding: 10-50ms
LLM Prefill: 20-100ms (取决于 prompt 长度和 visual token 数)
Autoregressive Decode: 50-300ms (7个 action tokens x 7-40ms/token)
Action Detokenize: 1ms

Bottleneck: Autoregressive decode (由 memory-bandwidth 限制)
```

**World Action Model (DreamZero 类):**
```
Observation Encoding: 10ms
Latent Video Prediction: 80-120ms (多步 diffusion/flow)
Action Decoding: 10ms

Bottleneck: Video prediction (compute-bound, 多步迭代)
```

### 4.3 适用性矩阵

| 应用场景 | 控制需求 | VA | VLA | WAM | 最佳选择 |
|---------|---------|----|----|-----|---------|
| 工业产线 (重复) | >100Hz, 固定 | +++ | - | - | VA (1-step flow) |
| 灵巧手操作 | >50Hz, 精细 | +++ | - | + | VA (3D flow) |
| 家庭服务 (语言指令) | 5-20Hz, 开放 | - | +++ | ++ | VLA |
| 自动驾驶决策 | 10-20Hz, 推理 | + | ++ | +++ | WAM 或 VLA |
| 人形机器人全身 | >50Hz, 复杂 | ++ (locomotion) | + (manipulation) | + | Hybrid: VA(低层) + VLA(高层) |
| Zero-shot 新任务 | 容忍延迟 | - | ++ | +++ | WAM (DreamZero) |
| 多机器人协同 | 5-10Hz, 异步 | + | +++ | ++ | VLA (共享 serving) |
| Edge 部署 | <100ms, 低功耗 | +++ | + | - | VA (量化) |

### 4.4 VA 的 Inference 优化已经"饱和"了吗？

一个重要的观察：VA 的 inference 延迟 (1-5ms for 1-step flow) 已经快于大多数机器人的控制周期 (10ms @ 100Hz)。也就是说，**VA 的 inference optimization 空间已经很小了**。

进一步的优化方向：
1. **Vision encoding 加速：** 当 action generation 只需 1ms 时，vision encoding (~10ms) 成为新的 bottleneck
   - 解决方案：lightweight ViT, temporal feature reuse, event camera
2. **End-to-end hardware optimization：** TensorRT, ONNX 编译
3. **Batch inference for multi-robot：** 多个 robot 共享一个 GPU 做 batch inference

相比之下，**VLA 和 WAM 的 inference 优化空间巨大** -- 这正是张昊实验室的机会所在。

---

## 5. Systems 视角分析

### 5.1 VA / World Model 的 Serving 需求与 VLA 有何不同？

| 维度 | VA Serving | VLA Serving | WAM Serving |
|------|-----------|-------------|------------|
| 模型大小 | 10M-500M | 7B-70B | 100M-10B |
| 内存模式 | Compute-bound (diffusion) | Memory-bandwidth-bound (AR decode) | 混合 |
| KV-cache | 不需要 (非 AR) | 核心瓶颈 | 不需要/轻量 |
| Batching | 简单 (固定输入大小) | 复杂 (variable length) | 中等 |
| Latency SLO | 1-10ms (极严格) | 100-500ms (宽松) | 20-200ms (中等) |
| 吞吐需求 | 高 (多 robot 并发) | 中 | 低-中 |
| GPU 利用率 | 低 (模型小, GPU 空闲) | 高 (大模型) | 中-高 |
| 调度策略 | Round-robin 即可 | Need continuous batching | Need rollout scheduling |

**核心洞察：**
- VA 模型小、inference 快，但 GPU 利用率低 -- 一个 GPU 可以服务数十个 robot
- VLA 的 serving 与 LLM serving 高度相似，vLLM 的技术可以直接迁移
- WAM 的 serving 是全新问题：如何调度 latent rollout？rollout 中途可以中断吗？

### 5.2 World Model Rollout 的 Speculative Execution

这是一个非常有趣的 systems 研究问题。

**类比：** Speculative execution 在 CPU 和 LLM 中的成功
- CPU: 分支预测 + 推测执行 → 如果预测错误就 rollback
- LLM: Speculative decoding → draft model 推测 + large model verify

**World Model 的 speculative rollout：**
```
时间步 t:
  实际执行 action a_t
  同时，world model 推测性地 rollout:
    s_{t+1}, s_{t+2}, ..., s_{t+k} (基于当前 policy)
  当 t+1 的真实观测到来时:
    比较 s_{t+1}(predicted) vs s_{t+1}(actual)
    如果 close enough: 直接使用 s_{t+2}...s_{t+k} 的 pre-computed actions
    如果 divergent: discard 并重新 rollout
```

**可行性分析：**
- **Robot dynamics 比 language generation 更可预测：** 物理世界有连续性和惯性，短期 prediction 通常很准
- **Acceptance rate 可能很高：** 在稳态操作中 (如匀速移动)，world model 的预测几乎完美
- **Rejection cost 低：** 重新 rollout 只需 latent computation，不涉及 robot re-planning
- **实现挑战：** 需要 world model 的 state representation 与真实观测可比较

**系统设计：**
```
Thread 1 (Real-time): Observe → Compare → Execute (从 pre-computed buffer)
Thread 2 (Async): World model rollout → Fill pre-computed buffer
                  如果 Thread 1 报告 divergence → Reset rollout
```

### 5.3 Latent World Model vs Pixel-Level World Model 的系统开销

| 维度 | Latent World Model | Pixel-Level World Model |
|------|-------------------|------------------------|
| 代表 | Dreamer v3, DDP-WM | GAIA-1, UniSim, Sora |
| 单步 rollout 计算 | GRU + MLP, ~0.5ms | Full diffusion, ~50-200ms |
| 内存占用 | Latent: ~1KB/step | Pixel: ~200KB-1MB/step |
| H=15 rollout | ~8ms | ~750ms - 3s |
| GPU 需求 | 单 GPU 足够 | 多 GPU (video generation) |
| 质量 | 适合短期 | 长期 rollout 更真实 |
| Compositionality | 差 (latent 不可解释) | 好 (pixel 可以被检查) |

**系统设计影响：**

1. **Latent WM：** 可以在 robot 本地 edge GPU 上运行，延迟极低
   - Serving 需求类似 VA -- 小模型、低延迟、高频率
   - 不需要复杂的 serving infrastructure
   - 适合 real-time MPC

2. **Pixel WM：** 需要 cloud GPU，延迟高
   - Serving 需求类似 video generation -- 高算力、高内存
   - FastVideo 的技术直接适用
   - 适合 offline planning 或 asynchronous visual planning

### 5.4 World Model 的 Batching 与 Scheduling

**独特挑战：** World model inference 不是 "一问一答" 的 request-response 模式，而是 "持续 rollout" 模式。

```
VLA Serving Pattern (request-response):
  Robot → [Request: observation + instruction] → Server → [Response: action]
  每个请求独立

WAM Serving Pattern (continuous rollout):
  Robot → [Stream: observation_t] → Server → [Rollout: s_{t+1}...s_{t+H}] → [Action: a_t]
                                              ↑
  Robot → [Stream: observation_{t+1}] → Server → [Update rollout starting from t+1]
  请求之间有状态依赖
```

这意味着：
- **Continuous batching 需要适配：** 每个 robot 的 rollout 有内部状态 (latent state)，不能随意交换
- **State management：** Server 需要为每个 robot 维护 latent state buffer
- **Preemption：** 当新观测到来且 prediction 不准时，需要中断当前 rollout
- **这与 vLLM 的 continuous batching 有本质区别：** VLA/VLM 的 KV-cache 是 append-only 的，而 world model 的 state 可能需要 rollback

---

## 6. 与张昊实验室技术栈的迁移分析

### 6.1 FastVideo → Video World Model 加速

FastVideo 的核心技术 (STA, VSA, step-parallel) 与 video world model 有天然的对应关系：

| FastVideo 技术 | Video World Model 应用 | 预期效果 | 技术挑战 |
|---------------|----------------------|---------|---------|
| STA (Sliding Tile Attention) | Latent video prediction 的 spatial attention | 2-4x 加速 | World model 的 latent 分辨率可能较低，STA 收益有限 |
| VSA (Video Sparse Attention) | Temporal attention across predicted frames | 3-5x 加速 | 预测帧之间的 temporal pattern 与真实视频不同 |
| Step-parallel denoising | Parallel rollout steps | 2-3x 加速 | Rollout 有因果依赖 (s_t → s_{t+1})，并行化受限 |
| 模型蒸馏 | Multi-step → single-step world model | 10x+ 加速 | 单步 world model 的 prediction 质量可能下降 |
| Quantization | 低精度 latent computation | 2-4x 加速 | Latent 精度对 long-horizon 累积误差敏感 |

**最高价值迁移点：模型蒸馏**

FastVideo 将多步 video diffusion 蒸馏为少步的技术，可以直接用于将 DreamZero 类的多步 video prediction 蒸馏为 1-2 步：

```
Teacher: DreamZero (10-step video prediction, ~120ms)
     ↓ Consistency/MeanFlow distillation (FastVideo 技术)
Student: DreamZero-Fast (1-step prediction, ~15ms)
```

如果成功，这将使 WAM 的延迟从 ~130ms 降到 ~25ms，控制频率从 7Hz 提升到 40Hz，大幅扩展其 real-time 适用范围。

### 6.2 vLLM → WAM Serving

vLLM 的部分技术可以迁移到 WAM serving，但需要重大适配：

| vLLM 技术 | WAM 应用 | 适配难度 |
|-----------|---------|---------|
| Continuous batching | Multi-robot rollout batching | 中 (需要处理 state dependency) |
| PagedAttention | Latent state 的 paged memory management | 低 (直接适用) |
| Prefix caching | 相同环境的 latent state 缓存 | 中 (什么算 "相同"?) |
| PD Disaggregation | Encode-Rollout-Action 三阶段分离 | 高 (直接可迁移) |
| Speculative decoding | Speculative rollout (见 5.2) | 高 (新研究方向) |

### 6.3 DistServe → WAM 的 Encode-Rollout-Action (ERA) Disaggregation

类比 VLM 的 EPD (Encode-Prefill-Decode) 分离，WAM 天然地分为三个阶段：

```
ERA Disaggregation for WAM:

Encode:  Vision encoding (compute-bound)
         → 专用 GPU cluster (high compute, moderate memory)

Rollout: Latent world model rollout (compute + memory)
         → 专用 GPU cluster (需要 maintain state)

Action:  Action decoding (lightweight)
         → 可以在 edge 或 shared pool 上运行
```

这与 EPD 的思路完全一致，但有一个独特之处：**Rollout 阶段是有状态的**，需要为每个 robot 维护 persistent latent state。这要求 serving system 支持 stateful session management。

---

## 7. Research Gaps 与机会

### 7.1 已识别的高价值 Gaps

#### Gap 1: World Model Inference Efficiency (极高价值)

**现状：** Video world model (DreamZero) 的 inference 太慢 (7Hz)，latent world model (Dreamer) 太简单 (不支持 rich visual planning)。中间的 "efficient visual world model" 缺乏系统性研究。

**机会：**
- FastVideo 的蒸馏技术 → 1-step video world model
- Sparse Imagination (ICLR 2026) 开辟了方向但仅是起点
- DDP-WM 的动态分解思想可以进一步推广

**与实验室的匹配：** 极高 (FastVideo 技术直接迁移)

#### Gap 2: WAM Serving System (极高价值)

**现状：** 不存在专门的 World Action Model serving system。DreamZero 等工作只关注 single-instance inference，没有考虑 multi-robot serving、state management、rollout scheduling。

**机会：**
- 构建 "WAM 的 vLLM" -- ERA disaggregation + stateful session + speculative rollout
- 这是一个完全空白的领域

**与实验室的匹配：** 极高 (vLLM + DistServe 经验)

#### Gap 3: VA 的 Vision Encoding Bottleneck (高价值)

**现状：** VA 的 action generation 已经优化到 sub-ms，但 vision encoding (~10ms) 成为新的 bottleneck。

**机会：**
- Temporal feature reuse (相邻帧的 visual features 高度相似)
- Lightweight vision encoder 蒸馏 (DINOv2 → lightweight student)
- Event camera 利用 (sparse update 而非 full frame encoding)

**与实验室的匹配：** 中 (需要 vision encoder 优化经验)

#### Gap 4: Unified VA/VLA/WAM Serving (高价值)

**现状：** 三种范式各自独立，没有统一的 serving framework。Dual-system 架构 (GR00T N1) 需要同时 serve VLA 和 VA，但没有系统支持。

**机会：**
- 统一的 heterogeneous model serving framework
- 类似 microservice architecture：VLA reasoning 服务 + VA action 服务 + WAM planning 服务
- 智能路由：根据任务复杂度动态选择 VA/VLA/WAM

**与实验室的匹配：** 高 (systems design 强项)

#### Gap 5: Speculative Rollout for World Models (中-高价值)

**现状：** Speculative execution 在 LLM (speculative decoding) 和 CPU (branch prediction) 中已被证明有效，但在 world model 中完全未探索。

**机会：**
- World model 的 speculative rollout 机制设计
- Acceptance criterion for physical states (比 token-level verification 更复杂)
- 与 Falcon (2025, arXiv: 2503.00339) 的 "partial denoising + action reuse" 有相似精神

**与实验室的匹配：** 高 (Lookahead/SD 经验)

### 7.2 Research Entry Point 推荐 (按优先级)

1. **FastVideo 蒸馏技术 → 1-step World Action Model** -- 最快速的验证路径，直接 leverage FastVideo 代码和经验。目标：将 DreamZero 的 7Hz 提升到 40Hz+。

2. **ERA Disaggregated WAM Serving** -- 中期项目，构建 world model 的专门 serving 系统。与 VLA serving 的研究互补。

3. **Speculative Rollout** -- 长期项目，全新的 systems 概念。需要与 RL/world model 社区合作。

4. **Unified VA/VLA/WAM Serving Router** -- 综合性工作，需要先建立各自 baseline。

---

## 附录 A: 关键论文索引

### VA 核心论文
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| Diffusion Policy | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | 2023 | 2303.04137 |
| DP3 | 3D Diffusion Policy | 2024 | 2403.03954 |
| ACT | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | 2023 | 2304.13705 |
| BeT | Behavior Transformers | 2022 | 2206.11251 |
| FlowPolicy | Consistency Flow Matching for Manipulation | 2024 | 2412.04987 |
| DM1 | MeanFlow One-Step Diffusion for Manipulation | 2025 | 2510.07865 |
| ManiFlow | Consistency Flow Training for Manipulation | 2025 | 2509.01819 |
| Falcon | Partial Denoising + Action Reuse | 2025 | 2503.00339 |
| Action-to-Action Flow | 0.56ms Action Generation | 2026 | 2602.07322 |
| Sparse ActionGen | Real-time Pruning for Diffusion Policy | 2026 | 2601.12894 |
| E3Flow | SE(3)-Equivariant Flow Policy | 2026 | - |
| One-Step Flow Policy | Self-Distillation for Fast Visuomotor | 2026 | - |
| MGP | Masked Generative Policy | 2025 | - |
| PocketDP3 | Pocket-Scale 3D Visuomotor Policy | 2026 | - |

### World Model 核心论文
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| Dreamer v3 | Mastering Diverse Domains through World Models | 2023 | 2301.04104 |
| DayDreamer | World Models for Physical Robot Learning | 2022 | 2206.14176 |
| DIAMOND | Diffusion for World Modeling | 2024 | 2405.12399 |
| Genie | Generative Interactive Environments | 2024 | 2402.15391 |
| UniSim | Universal Simulator | 2023 | 2310.06680 |
| DreamZero | World Action Models are Zero-shot Policies | 2026 | 2602.15922 |
| Cosmos Policy | Fine-Tuning Video Models for Visuomotor Control | 2026 | - |
| DDP-WM | Disentangled Dynamics Prediction | 2026 | 2602.01780 |
| Sparse Imagination | Efficient Visual World Model Planning (ICLR 2026) | 2025 | 2506.01392 |
| STORM | Search-Guided Generative World Models | 2025 | - |
| ChronoDreamer | Action-Conditioned World Model | 2025 | - |
| Motus | Unified Latent Action World Model | 2025 | - |
| Act2Goal | From World Model to Goal-Conditioned Policy | 2025 | - |

### Video → Action 融合论文
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| mimic-video | Video-Action Models Beyond VLAs | 2025 | 2512.15692 |
| Vidarc | Embodied Video Diffusion for Closed-Loop Control | 2025 | - |
| CoVAR | Co-generation of Video and Action | 2025 | - |
| Large Video Planner | Generalizable Robot Control via Video | 2025 | - |
| Dream2Flow | Video Generation to 3D Object Flow | 2025 | - |
| Cosmos-H-Surgical | Surgical Robot via World Modeling | 2025 | - |

### Efficiency 核心论文
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| FASTER | Single-Step Flow VLA | 2026 | 2603.19199 |
| Mean-Flow VLA | One-Step Mean-Flow VLA | 2026 | 2603.01469 |
| Action-to-Action Flow | 0.56ms Action Generation | 2026 | 2602.07322 |
| DDP-WM | 9x Inference Speedup for World Model | 2026 | 2602.01780 |
| Sparse Imagination | Efficient Visual WM Planning | 2025 | 2506.01392 |

---

## 附录 B: 关键时间线

```
2019-2020
  |-- PlaNet, Dreamer v1 (latent world model for RL)

2021-2022
  |-- Dreamer v2 (discrete latent)
  |-- DayDreamer (real robot Dreamer)
  |-- BeT (Behavior Transformer)

2023
  |-- Diffusion Policy (DDPM, 100-step, 开创性)
  |-- ACT (CVAE + Transformer, single-pass)
  |-- Dreamer v3 (universal hyperparameters)
  |-- GAIA-1 (video world model for driving)
  |-- UniSim (universal video simulator)

2024
  |-- DP3 (3D Diffusion Policy)
  |-- Genie (interactive video world model)
  |-- FlowPolicy (1-2 step consistency flow)
  |-- DIAMOND (diffusion as world model)

2025 H1
  |-- DM1, ManiFlow (1-step flow for manipulation)
  |-- Falcon (partial denoising + action reuse)
  |-- Sparse Imagination (ICLR 2026, efficient WM planning)
  |-- mimic-video (Video-Action Model)

2025 H2
  |-- STORM, ChronoDreamer (WM for robotic planning)
  |-- Motus (unified latent action WM)
  |-- Cosmos (NVIDIA world foundation model)

2026 Q1 (当前)
  |-- DreamZero (World Action Model, 7Hz real-time)
  |-- Cosmos Policy (video model → visuomotor)
  |-- DDP-WM (9x efficient world model)
  |-- Action-to-Action Flow (0.56ms VA)
  |-- One-Step Flow Policy (self-distillation)
  |-- E3Flow (SE(3)-equivariant flow)
  |-- PocketDP3 (compact 3D policy)
```

---

## 附录 C: 核心论断总结

1. **VA 的 inference 优化已接近物理极限：** 1-step flow VA 的 action generation (~1ms) 远快于 vision encoding (~10ms) 和 sensor readout (~10-33ms)。进一步的 VA 优化应关注 vision encoding 而非 action generation。

2. **World Action Model 是 2026 年最有前景的新范式：** DreamZero 证明了 zero-shot 能力，但 inference 延迟 (7Hz) 限制了其实际部署。**FastVideo → WAM acceleration** 是一个自然且高价值的研究方向。

3. **三大范式 (VA/VLA/WAM) 正在趋向融合：** Dual-system 架构 (VLA reasoning + VA control)、Video-Action Models (mimic-video)、以及 world model augmented VLA 都在尝试融合各自的优势。这种融合需要新的 serving system 设计。

4. **Systems 研究的蓝海在 WAM 和 Dual-System Serving：** VA serving 已经"解决" (模型小、inference 快)，VLA serving 正在快速发展 (EPD disaggregation)，但 WAM serving 和 heterogeneous VA+VLA+WAM serving 完全空白。

5. **Speculative rollout 是一个值得探索的新 systems 概念：** 将 speculative decoding 的思想推广到 world model 的 rollout 过程中，可能是连接 systems 和 world model 研究的独特切入点。

---

*本 survey 基于 2026 年 4 月的文献检索编写。所有标注 arXiv ID 的论文可通过 arXiv 直接验证。部分未标注 arXiv ID 的论文信息来自搜索结果的标题和摘要，具体内容以原始论文为准。*
