# World Action Model / Video World Model / VA Policy 最新论文调研

> 调研时间: 2026-04-11
> 覆盖范围: 2025-2026 年 arXiv 论文
> 关注领域: World Action Model, Video World Model, VA (Vision-Action) Policy, World Model + VLA 融合, 工业界进展

---

## 1. World Action Model (WAM) 论文

### 1.1 DreamZero: World Action Models are Zero-shot Policies
- **arXiv ID**: 2602.15922
- **时间**: 2026-02
- **核心贡献**: 提出 World Action Model (WAM) 概念，使用 video diffusion 学习物理动力学，同时预测未来世界状态和动作。14B 参数模型实现 7Hz 实时 closed-loop control。相比 VLA 模型泛化能力提升 2x 以上，支持 cross-embodiment transfer，仅需 10-20 分钟数据即可适配新任务，30 分钟数据适配新 embodiment。
- **Real-time 关联**: 7Hz closed-loop control 已接近实用门槛，但 14B 参数规模对部署带来挑战。

### 1.2 World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry
- **arXiv ID**: 2604.01985
- **时间**: 2026-04
- **核心贡献**: 将 action-conditioned prediction 分解为 state plausibility 和 action reachability 两部分，通过 cycle consistency verification 实现 world model 的自我改进。
- **Real-time 关联**: 提出的验证机制可用于在线检测 world model 预测质量，对实时系统的可靠性有重要意义。

### 1.3 Do World Action Models Generalize Better than VLAs? A Robustness Study
- **arXiv ID**: 2603.22078
- **时间**: 2026-03
- **核心贡献**: 对比研究表明 WAM 通过显式预测未来动力学，在扰动场景下比 VLA 具有更强的鲁棒性。
- **Real-time 关联**: WAM 的鲁棒性优势在真实部署中至关重要，但需权衡 world model 推理的额外计算开销。

### 1.4 JailWAM: Jailbreaking World Action Models in Robot Control
- **arXiv ID**: 2604.05498
- **时间**: 2026-04
- **核心贡献**: 首个针对 WAM 的 jailbreak 攻击框架，提出三级安全分类框架，攻击成功率达 84.2%。
- **Real-time 关联**: 安全性是实时部署的关键考量，揭示了 WAM 在对抗环境下的脆弱性。

### 1.5 VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models
- **arXiv ID**: 2603.19370
- **时间**: 2026-03
- **核心贡献**: 提出 post-training 框架，通过 policy optimization 直接改善 video action model 的视觉动力学质量，使用 expert latent space dynamics 定义 reward。
- **Real-time 关联**: 改善视觉预测质量可减少 closed-loop control 中的错误累积。

### 1.6 GigaWorld-Policy: An Efficient Action-Centered World-Action Model
- **arXiv ID**: 2603.17240
- **时间**: 2026-03
- **核心贡献**: 以动作为中心的 WAM，预测条件于当前观测的未来动作序列，视频生成为可选模块。无需生成视频即可直接输出动作，加速部署。
- **Real-time 关联**: **直接关联** -- 将视频生成设为可选极大降低了推理开销，适合实时场景。

### 1.7 Veo-Act: How Far Can Frontier Video Models Advance Generalizable Robot Manipulation?
- **arXiv ID**: 2604.04502
- **时间**: 2026-04
- **核心贡献**: 研究前沿视频生成模型作为 high-level motion planner，结合 VLA 作为 low-level executor 的层级框架。
- **Real-time 关联**: 层级结构中 high-level planner 可低频运行，low-level executor 高频运行，适合实时控制。

### 1.8 DriveVA: Video Action Models are Zero-Shot Drivers
- **arXiv ID**: 2604.04198
- **时间**: 2026-04
- **核心贡献**: 在 shared latent generative process 中联合解码未来视觉预测和动作序列，用于自动驾驶，NAVSIM 上达到 90.9 PDM score。
- **Real-time 关联**: 自动驾驶场景对实时性要求极高，shared latent space 设计有助于降低总推理延迟。

### 1.9 Action Images: End-to-End Policy Learning via Multiview Video Generation
- **arXiv ID**: 2604.06168
- **时间**: 2026-04
- **核心贡献**: 将 robot policy learning 构建为 multiview video generation，将 7-DoF 动作转化为可解释的 "action images"，以 2D pixel 为基础。
- **Real-time 关联**: 视觉化动作表示便于调试和验证，但多视角生成的计算量需要优化。

---

## 2. Video World Model for Robotics

### 2.1 ViVa: A Video-Generative Value Model for Robot RL
- **arXiv ID**: 2604.08168
- **时间**: 2026-04
- **核心贡献**: 复用预训练视频生成器进行 value estimation，将价值评估根植于预期的 embodiment dynamics。
- **Real-time 关联**: 视频生成用于 value 估计而非 planning，计算开销集中在训练阶段。

### 2.2 DreamPlan: Efficient RL Fine-Tuning of Vision-Language Planners via Video World Models
- **arXiv ID**: 2603.16860
- **时间**: 2026-03
- **核心贡献**: 完全在 video world model "想象" 中进行 vision-language planner 的 reinforcement fine-tuning，注入物理知识而无需大规模真实数据采集。
- **Real-time 关联**: 训练效率提升，但推理阶段 planner 已脱离 world model，不增加推理负担。

### 2.3 PlayWorld: Learning Robot World Models from Autonomous Play
- **arXiv ID**: 2603.09030
- **时间**: 2026-03
- **核心贡献**: 完全从无监督机器人自主探索数据训练 video world simulator，捕获人类采集数据中缺失的复杂长尾物理交互。
- **Real-time 关联**: 提供更丰富的 world model 训练数据，间接改善下游 policy 的鲁棒性。

### 2.4 Interactive World Simulator for Robot Policy Training and Evaluation
- **arXiv ID**: 2603.08546
- **时间**: 2026-03
- **核心贡献**: 使用 consistency model 实现快速、稳定的物理交互模拟，world model 生成数据训练的 policy 可匹配真实世界性能。
- **Real-time 关联**: Consistency model 本身推理极快，适合实时 world model 推理场景。

### 2.5 EgoSim: Egocentric World Simulator for Embodied Interaction Generation
- **arXiv ID**: 2604.01001
- **时间**: 2026-04
- **核心贡献**: 闭环第一人称模拟器，生成空间一致性交互视频并持续更新底层 3D 场景状态。
- **Real-time 关联**: 闭环模拟器设计对实时 embodied 系统的训练和测试至关重要。

### 2.6 CRAFT: Video Diffusion for Bimanual Robot Data Generation
- **arXiv ID**: 2604.03552
- **时间**: 2026-04
- **核心贡献**: 基于 video diffusion 的双臂操作示范数据生成框架，可合成时间一致的操作视频及动作标签。
- **Real-time 关联**: 数据增强工具，不直接影响推理速度，但改善 policy 训练质量。

### 2.7 Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model
- **arXiv ID**: 2604.03181
- **时间**: 2026-04
- **核心贡献**: 联合建模 3D 时空状态，预测 heatmap 和 RGB 视频，对齐 pretraining 和 fine-tuning 的表示格式。
- **Real-time 关联**: 多视角视频扩散的推理开销较大，需要进一步压缩才能实时部署。

### 2.8 DiT4DiT: Jointly Modeling Video Dynamics and Actions
- **arXiv ID**: 2603.10422
- **时间**: 2026-03
- **核心贡献**: 耦合 video 和 action diffusion transformer，从视频生成过程的中间去噪特征中提取时序 grounded conditions 用于动作预测。
- **Real-time 关联**: 共享 diffusion 计算可能降低总开销，但 DiT 本身推理成本仍较高。

### 2.9 V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors
- **arXiv ID**: 2603.18811
- **时间**: 2026-03
- **核心贡献**: 利用视频生成模型作为运动先验，通过 visual-kinematic alignment 映射视觉预测到可执行机器人轨迹。
- **Real-time 关联**: 自动化轨迹生成，主要用于离线场景。

### 2.10 ImagiNav: Scalable Embodied Navigation via Generative Visual Prediction and Inverse Dynamics
- **arXiv ID**: 2603.13833
- **时间**: 2026-03
- **核心贡献**: 解耦视觉规划与执行，使用 fine-tuned 生成视频模型想象未来轨迹，inverse dynamics model 提取动作。
- **Real-time 关联**: 解耦设计允许异步规划和执行，有利于实时系统。

### 2.11 mimic-video: Video-Action Models
- **arXiv ID**: 2512.15692
- **时间**: 2025-12
- **核心贡献**: 基于 flow matching 的 action decoder，实现 10x sample efficiency 提升和 2x 更快收敛。
- **Real-time 关联**: Flow matching 比传统 diffusion 推理步数更少，有利于实时推理。

### 2.12 ViPRA: Video Prediction for Robot Actions
- **arXiv ID**: 2511.07732
- **时间**: 2025-11
- **核心贡献**: Chunked flow matching decoder 实现平滑、高频连续控制，达到 22 Hz。
- **Real-time 关联**: **直接关联** -- 22 Hz 控制频率对多数操作任务已接近实用水平。

### 2.13 PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation
- **arXiv ID**: 2601.03782
- **时间**: 2026-01
- **核心贡献**: 预训练 3D world model，以 point flows 在共享 3D 空间中表示 state 和 action，支持 embodiment-agnostic 操作。
- **Real-time 关联**: 3D point flow 表示相比视频生成计算更轻量。

---

## 3. VA (Vision-Action) Policy 效率与加速

### 3.1 One-Step / Few-Step 推理

#### 3.1.1 One Step Is Enough: Dispersive MeanFlow Policy Optimization
- **arXiv ID**: 2601.20701
- **时间**: 2026-01
- **核心贡献**: 真正的单步生成 (无需 distillation)，dispersive regularization 防止表示坍缩，RL fine-tuning。在真实 Franka 机器人上达到**数百 Hz**，5-20x 加速。
- **Real-time 关联**: **核心论文** -- 数百 Hz 的推理速度远超实时需求。

#### 3.1.2 Mean-Flow based One-Step Vision-Language-Action
- **arXiv ID**: 2603.01469
- **时间**: 2026-03
- **核心贡献**: 单步 action 生成，比竞争方法快 8.7x 和 83.9x，通过 flow-matching 无需 consistency constraints。
- **Real-time 关联**: **核心论文** -- 极致的推理加速，适合部署在资源受限平台。

#### 3.1.3 Action-to-Action Flow Matching
- **arXiv ID**: 2602.07322
- **时间**: 2026-02
- **核心贡献**: 从随机噪声初始化转向 action-informed sampling，**单步推理仅 0.56 ms 延迟**。
- **Real-time 关联**: **核心论文** -- 0.56 ms 延迟是目前报告的最低值之一。

#### 3.1.4 FODMP: Fast One-Step Diffusion of Movement Primitives
- **arXiv ID**: 2603.24806
- **时间**: 2026-03
- **核心贡献**: 在 ProDMPs 参数空间中进行 single-step consistency distillation，比 MPD 快 10x，比 action-chunking 快 7x。
- **Real-time 关联**: Movement primitive 空间的单步生成特别适合实时反应式任务。

#### 3.1.5 CoLA-Flow Policy: Temporally Coherent Imitation Learning
- **arXiv ID**: 2601.23087
- **时间**: 2026-01
- **核心贡献**: 在 continuous latent action space 中进行 flow matching，近单步推理，同时保持轨迹平滑性。
- **Real-time 关联**: Latent space 操作降低维度，加速推理。

#### 3.1.6 OMP: One-step MeanFlow Policy
- **arXiv ID**: 2512.19347
- **时间**: 2025-12
- **核心贡献**: 引入 directional alignment 机制解决 MeanFlow 中的 spectral bias，实现高精度单步推理。
- **Real-time 关联**: 解决了单步生成的精度瓶颈问题。

#### 3.1.7 ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows
- **arXiv ID**: 2509.22695
- **时间**: 2025-09
- **核心贡献**: SE(3)-equivariant rectified flows，单步推理误差降低 48.5%。
- **Real-time 关联**: Equivariant 设计减少所需数据量和推理步数。

### 3.2 Diffusion Policy 加速

#### 3.2.1 E3Flow: SO(3) Equivariant Learning
- **arXiv ID**: 2603.23227
- **时间**: 2026-03
- **核心贡献**: 通过 SO(3) equivariant learning 结合 spherical harmonics 实现 7x 推理加速，同时提升成功率。
- **Real-time 关联**: 对称性利用是降低计算量的有效途径。

#### 3.2.2 CF-SDP: Classifier-Free Shortcut Diffusion Policy
- **arXiv ID**: 2504.09927
- **时间**: 2025-04
- **核心贡献**: Classifier-free guidance 与 shortcut acceleration 结合，近 5x 推理加速。
- **Real-time 关联**: 即插即用的加速方案，无需重训模型。

#### 3.2.3 D3P: Dynamic Denoising Diffusion Policy
- **arXiv ID**: 2508.06804
- **时间**: 2025-08
- **核心贡献**: 通过 RL 自适应分配每个 action 的去噪步数，实现 2.2x 推理加速。
- **Real-time 关联**: 自适应步数分配对动态环境特别有价值。

#### 3.2.4 Sparse ActionGen
- **arXiv ID**: 2601.12894
- **时间**: 2026-01
- **核心贡献**: Rollout-adaptive pruning 和 cached activation reuse，最高 4x 生成加速。
- **Real-time 关联**: 缓存复用策略对连续推理场景效果显著。

#### 3.2.5 Dynamic Test-Time Compute Scaling
- **arXiv ID**: 2511.20906
- **时间**: 2025-11
- **核心贡献**: 根据观测到的任务难度自适应调整 integration horizon，2.6-4.4x 计算减少。
- **Real-time 关联**: 简单动作快速生成，复杂动作分配更多计算，实时系统可按需调节。

#### 3.2.6 Falcon: Fast Visuomotor Policies
- **arXiv ID**: 2503.00339
- **时间**: 2025-02
- **核心贡献**: Training-free 加速插件，通过 partial denoising reuse 实现 2-7x 加速。
- **Real-time 关联**: 无需重训即可加速现有 diffusion policy，部署友好。

#### 3.2.7 Self-Imitated Diffusion Policy
- **arXiv ID**: 2601.22965
- **时间**: 2026-01
- **核心贡献**: Self-imitation 机制，推理时间从 273ms 降至 110ms (2.5x 加速)。
- **Real-time 关联**: 110ms 推理延迟约对应 9 Hz 控制频率。

#### 3.2.8 SeFA-Policy: Selective Flow Alignment
- **arXiv ID**: 2511.08583
- **时间**: 2025-11
- **核心贡献**: Selective flow alignment 实现超过 98% latency reduction，同时保持 observation consistency。
- **Real-time 关联**: **核心论文** -- 98% latency reduction 代表量级级别的加速。

### 3.3 轻量化模型

#### 3.3.1 PocketDP3: 极致参数压缩
- **arXiv ID**: 2601.22018
- **时间**: 2026-01
- **核心贡献**: 参数量降至先前方法的不到 1%，支持两步推理且不牺牲性能。
- **Real-time 关联**: **核心论文** -- 极致压缩适合边缘设备部署。

#### 3.3.2 KAN We Flow? RWKV-KAN for 3D Flow Matching
- **arXiv ID**: 2602.01115
- **时间**: 2026-02
- **核心贡献**: RWKV-KAN blocks 替代 UNet，参数减少 86.8%，保持 SOTA 成功率。
- **Real-time 关联**: 轻量级 backbone 对实时推理至关重要。

#### 3.3.3 TinyVLA: Fast, Data-Efficient VLA Models
- **arXiv ID**: 2409.12514
- **时间**: 2025-05 (最新版)
- **核心贡献**: 紧凑型 VLA 模型，更快推理速度和更高数据效率，无需大规模预训练。
- **Real-time 关联**: 专注推理速度优化的轻量 VLA。

#### 3.3.4 PointNet4D: 4D Point Cloud Processing
- **arXiv ID**: 2512.01383
- **时间**: 2025-12
- **核心贡献**: 结合 Mamba 和 Transformer 的轻量 4D backbone，用于实时点云处理。
- **Real-time 关联**: Mamba 的线性复杂度适合实时处理长序列。

### 3.4 Asynchronous / 高频控制

#### 3.4.1 DuoCore-FS: Asynchronous Fast-Slow VLA
- **arXiv ID**: 2512.20188
- **时间**: 2025-12
- **核心贡献**: 真正异步的 Fast-Slow VLA 框架，实现 30 Hz whole-body action-chunk generation，约为先前 VLA 模型的 3 倍速度。
- **Real-time 关联**: **核心论文** -- 30 Hz 异步架构是目前 VLA 实时控制的标杆方案。

#### 3.4.2 MinD: Dual-System World Model
- **arXiv ID**: 2506.18897
- **时间**: 2025-06
- **核心贡献**: 低频视觉生成器 + 高频 diffusion policy 双系统，达到 11.3 FPS，带故障预测功能。
- **Real-time 关联**: 双频率设计是平衡精度与速度的实用方案。

---

## 4. ACT (Action Chunking Transformer) 进展 (2025-2026)

### 4.1 FTACT: Force Torque aware ACT
- **arXiv ID**: 2509.23112
- **时间**: 2025-09
- **核心贡献**: 为 ACT 增加力/力矩感知，支持 images、joint states、forces 的 end-to-end learning。
- **Real-time 关联**: 力反馈是接触丰富任务的实时控制要素。

### 4.2 FEWT: Frequency-Enhanced Wavelet-based Transformers
- **arXiv ID**: 2509.11109
- **时间**: 2025-09
- **核心贡献**: 频域增强 ACT，通过 wavelet decomposition 提升成功率达 30% (仿真) 和 6-12% (真实)。
- **Real-time 关联**: Wavelet 变换本身计算高效，不显著增加推理延迟。

### 4.3 START: Sub-task Aware Robotic Transformer
- **arXiv ID**: 2511.03181
- **时间**: 2025-11
- **核心贡献**: 扩展 ACT，加入 sub-task IDs 提供显式时序 grounding，用于长程可变形物体操作，成功率 97%。
- **Real-time 关联**: Sub-task 分解可实现更精细的实时任务切换。

### 4.4 RCM-ACT: Remote Center of Motion ACT for Surgical Robotics
- **arXiv ID**: 2508.19191
- **时间**: 2025-08
- **核心贡献**: 结合 action chunking transformer 与实时运动学重对齐，用于毫米级眼科手术。
- **Real-time 关联**: 手术机器人场景对实时性和精度有极端要求。

### 4.5 Haptic-Informed ACT
- **arXiv ID**: 2506.18212
- **时间**: 2025-06
- **核心贡献**: 触觉反馈增强 ACT，实现实时抓取失败检测和自适应修正。
- **Real-time 关联**: 实时触觉闭环控制。

### 4.6 MTIL: Mamba Temporal Imitation Learning
- **arXiv ID**: 2505.12410
- **时间**: 2025-05
- **核心贡献**: 利用线性时间 state space model 编码完整轨迹历史，超越 ACT 和 Diffusion Policy。
- **Real-time 关联**: Mamba 的线性复杂度对长序列实时推理有优势。

---

## 5. World Model + VLA 融合

### 5.1 DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA
- **arXiv ID**: 2603.29844
- **时间**: 2026-03
- **核心贡献**: 在 VLM feature space 中引入 latent intent bottleneck，VLM-based System-2 执行 latent world modeling 桥接高层推理与运动执行。
- **Real-time 关联**: Latent space world modeling 比像素空间高效得多。

### 5.2 VLAW: Iterative Co-Improvement of VLA Policy and World Model
- **arXiv ID**: 2602.12063
- **时间**: 2026-02
- **核心贡献**: 提出 VLA 和 World Model 的迭代共同改进框架，使用 action-conditioned video generation model 生成合成数据改善两者。
- **Real-time 关联**: 迭代改进提升 policy 质量，推理阶段 world model 可选。

### 5.3 WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL
- **arXiv ID**: 2602.13977
- **时间**: 2026-02
- **核心贡献**: 通过 controllable action-conditioned video world model 和 keyframe-initialized rollouts 调节 RL 与不完美 imagined dynamics 的交互。
- **Real-time 关联**: World model 用于训练阶段，不影响推理速度。

### 5.4 GigaBrain-0.5M: VLA + World Model-Based RL
- **arXiv ID**: 2602.12099
- **时间**: 2026-02
- **核心贡献**: 集成 world model-based RL，实现强大的 cross-task adaptation，在复杂真实操作任务上获得显著性能提升。
- **Real-time 关联**: RL fine-tuning 改善 policy 质量，部署时为标准 VLA 推理。

### 5.5 Towards Practical World Model-based RL for VLAs (VLA-MBPO)
- **arXiv ID**: 2603.20607
- **时间**: 2026-03
- **核心贡献**: 提出 VLA-MBPO 框架，利用统一多模态模型实现数据高效的 world modeling，支持 multi-view consistency。
- **Real-time 关联**: 统一多模态模型减少了独立 world model 的额外开销。

### 5.6 Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving
- **arXiv ID**: 2603.27287
- **时间**: 2026-03
- **核心贡献**: 交替执行未来帧预测和轨迹规划，实现 world modeling 和 control 的闭环交互。
- **Real-time 关联**: 交替式设计可流水线化执行，降低延迟。

### 5.7 World2Act: Latent Action Post-Training via Skill-Compositional World Models
- **arXiv ID**: 2603.10422
- **时间**: 2026-03
- **核心贡献**: 将 VLA 与 world model 的 video-dynamics latents 直接对齐，引入 skill-decomposition 实现不同任务时长的一致时序建模。
- **Real-time 关联**: Latent alignment 方式比显式视频生成更高效。

### 5.8 Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control
- **arXiv ID**: 2601.16163
- **时间**: 2026-01
- **核心贡献**: 通过 single stage post-training 将预训练视频生成模型适配为 robot policy，支持 model-based planning with state predictions and value functions。
- **Real-time 关联**: 统一架构减少了独立 world model + policy 的总开销。

### 5.9 WMPO: World Model-based Policy Optimization for VLAs
- **arXiv ID**: 2511.09515
- **时间**: 2025-11
- **核心贡献**: Pixel-based world model 实现 on-policy GRPO，无需真实世界交互，展现出 self-correction 等 emergent behaviors。
- **Real-time 关联**: World model 仅用于训练，推理时标准 VLA 速度。

### 5.10 STORM: Search-Guided Generative World Models
- **arXiv ID**: 2512.18477
- **时间**: 2025-12
- **核心贡献**: 结合 diffusion-based action generation、conditional video prediction 和 MCTS，平均成功率 51.0%。
- **Real-time 关联**: MCTS 搜索增加推理时间，需要权衡搜索深度与速度。

### 5.11 World-Gymnast: Training Robots with RL in a World Model
- **arXiv ID**: 2602.02454
- **时间**: 2026-02
- **核心贡献**: 在 action-conditioned video world model 中执行 RL fine-tuning，真实世界 transfer 优于监督学习和软件仿真。
- **Real-time 关联**: World model 用于训练加速，不影响部署。

### 5.12 StarVLA: A Lego-like Codebase for VLA Model Developing
- **arXiv ID**: 2604.05014
- **时间**: 2026-04
- **核心贡献**: 开源框架，统一支持 VLM 和 world-model backbone，标准化 VLA 开发流程。
- **Real-time 关联**: 标准化框架便于不同 world model + VLA 方案的公平比较和快速迭代。

---

## 6. 工业界进展 (Industry)

### 6.1 NVIDIA Cosmos

#### 6.1.1 Cosmos World Foundation Model Platform for Physical AI
- **arXiv ID**: 2501.03575
- **时间**: 2025-01 (修订至 2025-07)
- **核心贡献**: 综合平台，包含视频数据策展、预训练模型和 tokenizer，支持面向 physical AI 的定制 world model 构建。
- **Real-time 关联**: 提供 foundation model 基座，下游任务可针对速度优化。

#### 6.1.2 Cosmos-Predict2.5: World Simulation with Video Foundation Models
- **arXiv ID**: 2511.00062
- **时间**: 2025-10 (修订至 2026-02)
- **核心贡献**: 统一 Text2World、Image2World、Video2World 生成能力，通过 RL refinement 提升视频质量。
- **Real-time 关联**: 视频世界模拟器的质量提升有助于更可靠的 policy 训练。

#### 6.1.3 Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
- **arXiv ID**: 2503.15558
- **时间**: 2025-03
- **核心贡献**: 面向物理推理的多模态 LLM，基于时空物理层次本体进行推理。
- **Real-time 关联**: 物理常识推理可用于 high-level planning，但 LLM 推理速度是瓶颈。

#### 6.1.4 Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control
- **arXiv ID**: 2503.14492
- **时间**: 2025-03
- **核心贡献**: 条件式世界生成，支持 segmentation、depth 等空间控制信号，面向 Sim2Real robotics。
- **Real-time 关联**: Sim2Real 管线的核心组件。

#### 6.1.5 Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation
- **arXiv ID**: 2506.09042
- **时间**: 2025-06
- **核心贡献**: 专用驾驶场景模型，生成高保真多视角视频场景用于自动驾驶边缘场景训练。
- **Real-time 关联**: 数据生成工具，间接改善自动驾驶系统的实时决策。

#### 6.1.6 Foundational World Models Accurately Detect Bimanual Manipulator Failures
- **arXiv ID**: 2603.06987
- **时间**: 2026-03
- **核心贡献**: 使用 Cosmos Tokenizer 训练概率 world model 检测机器人故障，参数量仅为可比方法的 5%。
- **Real-time 关联**: **直接关联** -- 5% 参数量的 world model 适合实时故障检测。

### 6.2 Google / DeepMind

> 注: Google Genie 2 (2024-12 发布) 作为交互式 3D 世界生成模型，目前尚无正式 arXiv 论文。以下为相关后续工作:

#### 6.2.1 相关技术路线
- Genie 2 采用 autoregressive latent diffusion model 从单张图像生成可交互 3D 世界
- 支持键盘和鼠标动作控制
- 未公开论文但在技术博客中详述了架构 (2024-12)
- 2025-2026 年期间，Google DeepMind 的公开 world model 工作主要体现在 Gemini 多模态推理和 robotics foundation model 的整合中

### 6.3 Physical Intelligence (Pi)

#### 6.3.1 Pi0 系列 Foundation Model
- Physical Intelligence 的 pi0 是当前最具代表性的 generalist robot foundation model 之一
- 基于 flow matching 的 action generation
- 2025-2026 年相关评测和对比:

##### Benchmarking the Generality of VLA Models (MultiNet v1.0)
- **arXiv ID**: 2512.11315
- **时间**: 2025-12
- **核心贡献**: 统一评测 GPT-5、Pi0、Magma 在 6 个能力维度上的泛化性，发现没有模型展现一致的通用性。
- **Real-time 关联**: 揭示当前 generalist model 在未见领域的性能下降问题。

##### Restoring Linguistic Grounding in VLA Models (IGAR)
- **arXiv ID**: 2603.06001
- **时间**: 2026-03
- **核心贡献**: 发现 VLA 模型（含 Pi0, Pi0.5）存在 "linguistic blindness" 问题，提出 inference-time attention recalibration 修复方案。
- **Real-time 关联**: Inference-time 修复方案不增加训练成本。

##### H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
- **arXiv ID**: 2507.23523
- **时间**: 2025-07
- **核心贡献**: 利用人类操作视频增强双臂机器人学习，超越包括 Pi0 和 RDT 在内的现有方法。

#### 6.3.2 HY-Embodied-0.5: Embodied Foundation Models
- **arXiv ID**: 2604.07430
- **时间**: 2026-04
- **核心贡献**: 面向真实世界 embodied agent 的 foundation model，Mixture-of-Transformers 架构，提供 2B 和 32B 参数版本。
- **Real-time 关联**: 2B 参数版本面向实时部署场景。

---

## 7. Model-Based RL for Robotics 效率进展

### 7.1 DREAMer-VXS: Latent World Model for AGV Exploration
- **arXiv ID**: 2512.00005
- **时间**: 2025-12
- **核心贡献**: 通过 latent trajectory planning 实现环境交互次数减少 90%。
- **Real-time 关联**: 数据效率提升 10x，减少真实世界交互需求。

### 7.2 Robo-Dopamine: General Process Reward Modeling
- **arXiv ID**: 2512.23703
- **时间**: 2025-12
- **核心贡献**: Step-aware process reward model，仅需约 1 小时真实机器人交互 (150 rollouts) 即可从近零基线达到 95% 成功率。
- **Real-time 关联**: 极致的 sample efficiency 降低了训练成本。

### 7.3 Efficient Model-Based RL via Online Learning
- **arXiv ID**: 2510.18518
- **时间**: 2025-10
- **核心贡献**: 基于 regret bounds 的在线学习方法，在真实机械臂上数小时内达到可比性能。
- **Real-time 关联**: 在线适应能力对动态环境中的实时系统至关重要。

### 7.4 Residual MPC: Blending RL with GPU-Parallelized MPC
- **arXiv ID**: 2510.12717
- **时间**: 2025-10
- **核心贡献**: GPU 并行 MPC 结合 RL 的残差结构，更高 sample efficiency 和 zero-shot terrain adaptation。
- **Real-time 关联**: GPU 并行化 MPC 适合实时控制回路。

### 7.5 SRPO: Self-Referential Policy Optimization
- **arXiv ID**: 2511.15605
- **时间**: 2025-11
- **核心贡献**: 利用 latent world representations 鲁棒测量行为进度，以最少 RL 步数达到 99.2% 成功率。
- **Real-time 关联**: 极高 sample efficiency 减少训练周期。

### 7.6 BOOM: Bootstrap Off-policy with World Model
- **arXiv ID**: 2511.00423
- **时间**: 2025-11
- **核心贡献**: 紧密集成 planning 和 off-policy learning，policy 和 planner 与联合学习的 world model 协同工作。
- **Real-time 关联**: Planning + learning 的紧密集成可在推理时提供更优决策。

---

## 8. 综合趋势总结

### 8.1 World Action Model (WAM) 已成为独立范式
- DreamZero (2602.15922) 正式确立了 WAM 概念
- WAM 通过联合建模视频和动作，在泛化性和鲁棒性上优于纯 VLA
- 但推理开销（14B 参数 / 7Hz）仍是实时部署的主要瓶颈

### 8.2 单步/少步推理已突破瓶颈
- Flow matching + MeanFlow 路线: 真正单步，数百 Hz (2601.20701)
- Action-to-Action: 0.56 ms 延迟 (2602.07322)
- Consistency distillation: 可将任意 diffusion policy 压缩到 1-2 步
- **趋势**: 从 "能否做到" 转向 "如何在不损精度下做到"

### 8.3 异步架构是 VLA 实时化的关键
- DuoCore-FS 的 Fast-Slow 异步设计 (30 Hz) 代表当前最佳实践
- MinD 的双频率系统 (11.3 FPS) 证明了分频设计的可行性
- **趋势**: 高频动作执行 + 低频世界理解的解耦

### 8.4 World Model 主要用于训练而非推理
- 多数 WM+VLA 融合工作中，world model 在训练阶段提供 synthetic data 或 reward signal
- 推理阶段通常退化为标准 VLA
- 例外: GigaWorld-Policy 将视频生成设为可选，Cosmos Policy 统一了 policy 和 world model

### 8.5 轻量化趋势显著
- PocketDP3: 参数降至 <1%
- KAN We Flow: 参数减少 86.8%
- Cosmos 故障检测器: 仅需 5% 参数
- **趋势**: 边缘部署和嵌入式场景的需求推动极致压缩

### 8.6 NVIDIA Cosmos 建立了工业级 World Model 生态
- 从 tokenizer 到 foundation model 到下游任务的完整 pipeline
- 覆盖自动驾驶、机器人操作、物理推理
- 开源策略加速了社区采用
