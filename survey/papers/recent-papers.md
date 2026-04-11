# VLM/VLA Inference Efficiency 最新进展 (2025-2026)

> 调研日期: 2026-04-11
> 目标: 为 UCSD PhD (导师: 张昊, vLLM 作者) 方向 — VLM/VLA real-time systems — 梳理最新文献与项目

---

## 目录

1. [VLM Serving/Inference 系统论文](#1-vlm-servinginference-系统论文)
2. [Vision Token Compression & KV Cache 优化](#2-vision-token-compression--kv-cache-优化)
3. [Speculative Decoding for VLM](#3-speculative-decoding-for-vlm)
4. [VLA Inference/Deployment 论文](#4-vla-inferencedeployment-论文)
5. [VLA 效率优化 (Token Pruning, Quantization, Layer Skipping)](#5-vla-效率优化)
6. [VLA Benchmark & Survey](#6-vla-benchmark--survey)
7. [关键开源项目与框架](#7-关键开源项目与框架)
8. [Industry Trends](#8-industry-trends)
9. [Edge Deployment](#9-edge-deployment)
10. [研究方向建议](#10-研究方向建议)

---

## 1. VLM Serving/Inference 系统论文

### 1.1 Nova: Real-Time Agentic VLM Serving with Adaptive Cross-Stage Parallelization
- **arXiv:** 2509.21301
- **作者:** Yuhang Xu, Shengzhong Liu, Dong Zhang, Bingheng Yan, Fan Wu, Guihai Chen
- **时间:** 2025.09
- **核心贡献:** 面向 agentic VLM 的单 GPU 调度框架，将 vision encoding、LLM prefill、decode 三阶段流水线化，采用弹性 GPU 空间划分 (elastic GPU spatial partitioning)，最大延迟降低 23.3%，同时保持有竞争力的吞吐量
- **与 real-time VLM/VLA 的关联:** 直接解决 VLM serving 的多阶段调度问题，是构建 real-time VLM serving system 的核心参考

### 1.2 RPS-Serve: Modality-aware Scheduling for MLLM Inference
- **arXiv:** 2603.26498
- **作者:** Konstantinos Papaioannou, Thaleia Dimitra Doudali
- **时间:** 2026.03
- **核心贡献:** 提出 "Rocks, Pebbles and Sand" 类比来处理不同模态的资源需求差异 — 视频如石头、图像如鹅卵石、文本如沙子。通过模态感知的请求优先级调度减少 time-to-first-token (TTFT)
- **与 real-time VLM/VLA 的关联:** 多模态混合请求场景下的调度策略，对 VLM serving 的 SLO 保证至关重要

### 1.3 Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling
- **arXiv:** 2512.17574
- **作者:** Lingxiao Zhao, Haoran Zhou, Yuezhi Che, Dazhao Cheng
- **时间:** 2025.12
- **核心贡献:** 提出 FlashCodec + UnifiedServe 框架，优化视频解码和 vision-to-text 流水线各阶段，实现 3.0x 请求容量提升和 4.4x 吞吐量提升，或 1.5x 更严格 SLO 保证
- **与 real-time VLM/VLA 的关联:** Disaggregated serving 是下一代 VLM serving 架构的关键方向

### 1.4 HeteroServe: Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity
- **arXiv:** 2603.12707
- **作者:** Donglin Yu
- **时间:** 2026.03
- **核心贡献:** 在模态边界处进行异构 GPU 划分，将 KV cache 传输复杂度从 O(L*s_ctx) 降至 O(N_v*d) bytes，相比同构部署降低 37% 成本
- **与 real-time VLM/VLA 的关联:** 异构 GPU 集群上的高效 VLM serving，降低部署成本

### 1.5 Modality Inflation: Energy Characterization and Optimization for MLLM Inference
- **arXiv:** 2512.22695
- **作者:** Mona Moghadampanah, Adib Rezaei Shahmirzadi, Farhana Amin, Dimitrios S. Nikolopoulos
- **时间:** 2025.12
- **核心贡献:** 全面分析 MLLM inference 各阶段 (vision encoding, prefill, decoding) 的能耗特征，多模态 overhead 从 17% 到 94% 不等，提出 stage-wise DVFS 优化方案
- **与 real-time VLM/VLA 的关联:** 为 VLM serving 的能效优化提供基础分析框架

### 1.6 Empirical Recipes for Efficient and Compact VLMs
- **arXiv:** 2603.16987
- **作者:** Jiabo Huang, Zhizhong Li, Sina Sajadmanesh, Weiming Zhuang, Lingjuan Lyu
- **时间:** 2026.03
- **核心贡献:** 端到端效率分析，在 InternVL3-2B 上降低 TTFT 53%，在 SmolVLM-256M 上降低 TTFT 93%，提供跨架构和框架的优化 recipes
- **与 real-time VLM/VLA 的关联:** 提供了实用的 VLM 加速技巧，直接可用于 real-time 部署

---

## 2. Vision Token Compression & KV Cache 优化

### 2.1 Vision Token Compression

#### FlashVLM: Text-Guided Visual Token Selection
- **arXiv:** 2512.20561 | **时间:** 2025.12
- **核心贡献:** 基于图像 token 与文本 embedding 的跨模态相似度进行 token 选择，pruning 77.8% visual tokens 同时超越 baseline 性能 ("beyond lossless compression")
- **关联:** Training-free，可直接集成到 serving pipeline

#### Adaptive-VoCo: Complexity-Aware Visual Token Compression
- **时间:** 2025.12
- **核心贡献:** 基于熵和 attention variance 动态调整压缩率

#### LUVC: Towards Lossless Ultimate Vision Token Compression
- **arXiv:** 2512.09010 | **时间:** 2025.12
- **核心贡献:** 逐层渐进压缩 visual tokens 直到最终层完全消除，实现 2x speedup 且精度无损

#### Focus: Streaming Concentration Architecture
- **arXiv:** 2512.14661 | **时间:** 2025.12
- **核心贡献:** 语义、时空、向量三层级压缩范式，2.4x speedup + 3.3x 能耗降低

#### TIE: Text-Guided Semantic Image Encoder
- **时间:** 2025.11
- **核心贡献:** Query-conditioned image encoding，仅需一半 image tiles 即可达到相同性能

#### GlimpsePrune: Dynamic Visual Token Pruning
- **arXiv:** 2508.01548 | **时间:** 2025.08
- **核心贡献:** 受人类认知启发，pruning 92.6% visual tokens 同时保持 baseline 性能

#### Balanced Token Pruning (NeurIPS 2025)
- **arXiv:** 2505.22038 | **时间:** 2025.05
- **核心贡献:** 多阶段 pruning 关注下游层影响，78% 压缩率保留 96.7% 平均性能

#### Skip-Vision (ICCV 2025)
- **arXiv:** 2503.21817 | **时间:** 2025.03
- **核心贡献:** Skip-FFN 策略跳过冗余 token 的 FFN 计算，训练降低 35%，inference FLOPs 降低 75%

#### ADSC: Attention-Driven Self-Compression
- **arXiv:** 2602.12618 | **时间:** 2026.02
- **核心贡献:** LLM 自身作为压缩指导，FLOPs 降低 53.7%，KV-cache 内存降低 56.7%，保持 98.2% 性能

#### VFlowOpt: Visual Information Flow-Guided Optimization
- **arXiv:** 2508.05211 | **时间:** 2025.09
- **核心贡献:** 渐进式 pruning with recycling，89% KV-Cache 内存降低，3.8x faster inference，pruning 90% visual tokens

#### Input-Adaptive Visual Preprocessing
- **时间:** 2025.12
- **核心贡献:** 根据图像内容动态调整输入分辨率，inference time 降低 50%+，visual token count 降低 55%

### 2.2 KV Cache 优化

#### CalibQuant: 1-Bit KV Cache Quantization for Multimodal LLMs
- **arXiv:** 2502.14882 | **时间:** 2025.03
- **核心贡献:** 极端 1-bit KV cache 量化 + post-scaling + calibration，通过 Triton 优化在 InternVL 上实现 10x 吞吐量提升
- **关联:** 极低 bit 量化对 edge 部署意义重大

#### PureKV: Plug-and-Play KV Cache with Spatial-Temporal Sparse Attention
- **arXiv:** 2510.25600 | **时间:** 2025.10
- **核心贡献:** 5.0x KV cache 压缩 + 3.16x prefill 加速

#### Q Cache: Visual Attention in Less than Half of Decode Layers
- **arXiv:** 2602.01901 | **时间:** 2026.02
- **核心贡献:** Lazy Attention + cross-layer attention sharing，35%+ KV cache 降低，1.5x 吞吐量提升

#### HAE: Hierarchical Adaptive Eviction for KV Cache in MLLMs
- **arXiv:** 2602.02197 | **时间:** 2026.02
- **核心贡献:** Dual-attention pruning + dynamic decoding eviction，KV cache 内存降低 41%

#### ConceptMoE: Adaptive Token-to-Concept Compression
- **arXiv:** 2601.21420 | **时间:** 2026.01
- **核心贡献:** 语义相似 token 动态合并为 concept，attention 计算降低 R^2，175% prefill + 117% decoding speedup

#### PrefixKV: Adaptive Prefix KV Cache
- **arXiv:** 2412.03409 | **时间:** 2025.10 (revised)
- **核心贡献:** Layer-wise 自适应 KV 保留，binary search 确定最优 prefix 配置

#### GUI-KV: Spatio-Temporal KV Cache for GUI Agents
- **arXiv:** 2510.00536 | **时间:** 2025.10
- **核心贡献:** GUI 特定的空间显著性 + 时间冗余 KV cache 压缩，decoding FLOPs 降低 38.9%

### 2.3 视频 Token 压缩

#### EVS: Efficient Video Sampling
- **arXiv:** 2510.14624 | **时间:** 2025.10
- **核心贡献:** 识别并移除时间上静态的 patch，LLM TTFT 降低 4x

#### METok: Multi-Stage Event-based Token Compression (EMNLP 2025)
- **arXiv:** 2506.02850 | **时间:** 2025.06
- **核心贡献:** 三阶段 training-free 框架，FLOPs 降低 80.6%，KV cache 降低 93.5%

#### Video-XL-Pro: Reconstructive Token Compression
- **arXiv:** 2503.18478 | **时间:** 2025.03
- **核心贡献:** ReCoT 可学习模块，单 A100 处理 8K+ frames

#### CacheFlow: Compressive Streaming Memory
- **arXiv:** 2511.13644 | **时间:** 2025.11
- **核心贡献:** Dynamic Token Dropping + compressive memory，减少 87% tokens 用于 long-form video QA

---

## 3. Speculative Decoding for VLM

### 3.1 MASSV: Multimodal Adaptation for Speculative Decoding
- **arXiv:** 2505.10526
- **作者:** Mugilan Ganesan, Shane Segal, Ankur Aggarwal, Nish Sinnadurai, Sean Lie, Vithursan Thangarasa
- **时间:** 2025.05
- **核心贡献:** 将小型 LM 转换为 multimodal drafter，visually-grounded 任务上端到端推理加速 1.46x
- **关联:** 第一批专门针对 VLM 的 speculative decoding 工作

### 3.2 Fast-dVLM: Block-Diffusion VLM via Direct Conversion
- **arXiv:** 2604.06832
- **作者:** Chengyue Wu, Shiyi Lan, Yonggan Fu 等 (Song Han 组)
- **时间:** 2026.04
- **核心贡献:** 将预训练 AR VLM 直接转换为 diffusion-based 模型，实现并行 token 生成。集成 SGLang + FP8 量化后达到 6x+ 端到端 inference speedup，11 个 multimodal benchmark 上与 AR 版本性能匹配
- **关联:** 非常重要 — 展示了非 AR 解码在 VLM 上的可行性，对 real-time 系统有重大意义

### 3.3 LVSpec: Loosely Speculative Decoding for Video LLMs
- **arXiv:** 2604.05650
- **作者:** Yicheng Ji, Jun Zhang, Jinpeng Chen 等
- **时间:** 2026.04
- **核心贡献:** Training-free 松弛式 speculative decoding，Qwen2.5-VL-32B 上 2.70x speedup，LLaVA-OneVision-72B 上 2.94x speedup，保持 >99.8% 目标性能
- **关联:** 大模型上的显著加速比，实用价值很高

### 3.4 ParallelVLM: Visual Alignment Aware Speculative Decoding
- **arXiv:** 2603.19610
- **时间:** 2026.03
- **核心贡献:** Training-free，解决 draft/target model 互相等待问题，LLaVA-OneVision-72B 上 3.36x speedup
- **关联:** 对大规模 VLM serving 的吞吐量提升有直接帮助

### 3.5 MMSpec: Benchmarking Speculative Decoding for VLMs
- **arXiv:** 2603.14989
- **时间:** 2026.03
- **核心贡献:** 首个全面评估 VLM speculative decoding 的 benchmark，覆盖 10 种算法 + 6 类任务，提出 ViSkip (dynamic vision token adaptation)
- **关联:** 为 VLM speculative decoding 研究提供标准化评估

---

## 4. VLA Inference/Deployment 论文

### 4.1 VLAgents: A Policy Server for Efficient VLA Inference
- **arXiv:** (2026.01)
- **作者:** Tobias Julg, Khaled Gamal, Nisarga Nilavadi 等
- **时间:** 2026.01
- **核心贡献:** 模块化 VLA policy server，集成 7 种策略 (含 OpenVLA 和 Pi Zero)，改善 inference latency。提供抽象层使得不同 VLA 模型可以统一 serving
- **关联:** 最直接的 VLA serving system 工作，相当于 VLA 领域的 "vLLM"

### 4.2 ActionFlow: Pipelined Action Acceleration on Edge
- **作者:** Yuntao Dai, Hang Gu, Teng Wang 等
- **时间:** 2025.12
- **核心贡献:** 跨请求流水线策略，OpenVLA-7B 上实现 2.55x FPS 提升，无需重新训练。在资源受限设备上实现实时机器人控制
- **关联:** 直接解决 VLA edge deployment 的 latency 问题

### 4.3 StreamVLA: Completion-State Gating
- **arXiv:** 2602.01100
- **时间:** 2026.02
- **核心贡献:** Dual-system 架构 + "Lock-and-Gated" 机制处理 long-horizon manipulation，LIBERO 上 98.5% success rate + 48% latency 降低
- **关联:** 将 fast/slow 双系统理论应用于 VLA，实现低延迟 + 高成功率

### 4.4 dVLA: Diffusion VLA with Multimodal Chain-of-Thought
- **arXiv:** 2509.25681
- **时间:** 2025.09
- **核心贡献:** 引入 prefix attention mask + KV caching 加速，实现约 6x inference speedup
- **关联:** Diffusion-based VLA 的 KV cache 优化

### 4.5 MoTVLA: Mixture-of-Transformers VLA
- **arXiv:** 2510.18337
- **时间:** 2025.10
- **核心贡献:** Mixture-of-Transformers 架构实现 fast-slow reasoning 集成，平衡语言可控性与 inference 效率

### 4.6 VLA-AN: Efficient VLA for Aerial Navigation
- **时间:** 2025.12
- **核心贡献:** 资源受限 UAV 上 8.3x inference throughput 提升

### 4.7 A Dual Process VLA
- **arXiv:** 2410.15549
- **时间:** 2024.10
- **核心贡献:** 分离大模型 (reasoning) 和小模型 (real-time control) 的层级式架构，受 dual-process theory 启发
- **关联:** 大小模型协同的 VLA 架构，可能是 real-time 系统的重要范式

---

## 5. VLA 效率优化

### 5.1 Token Pruning / Compression

#### BFA++: Hierarchical Best-Feature-Aware Token Prune for VLA
- **arXiv:** 2602.20566 | **时间:** 2026.02
- **核心贡献:** 层级化 token pruning (intra-view + inter-view)，约 10% success rate 提升 + 1.8x speedup

#### DepthCache: Depth-Guided Token Merging for VLA Inference
- **时间:** 2026.03
- **核心贡献:** 深度引导的 token compression，pi_0.5/OpenVLA/GR00T 上 1.28x inference speedup

#### Compressor-VLA: Instruction-Guided Visual Token Compression
- **arXiv:** 2511.18950 | **时间:** 2025.11
- **核心贡献:** 任务导向的 token 压缩，FLOPs 降低 59%，visual token 降低 3x

#### ETA-VLA: Efficient Token Adaptation via Temporal Fusion
- **arXiv:** 2603.25766 | **时间:** 2026.03
- **核心贡献:** 面向自动驾驶 VLA，pruning 85% visual tokens，FLOPs 降低 61%，保留 94% accuracy

### 5.2 Quantization

#### HBVLA: 1-Bit Post-Training Quantization for VLA
- **arXiv:** (2026.02)
- **核心贡献:** OpenVLA-OFT 的 1-bit 量化保留 92.2% full-precision 性能，可部署于资源受限平台

#### QVLA: Channel-wise Quantization for VLA
- **arXiv:** (2026.02)
- **核心贡献:** Channel-wise 量化实现 1.49x speedup，仅需 29.2% 原始 VRAM，98.9% 性能保留

#### SQAP-VLA: Synergistic Quantization-Aware Pruning
- **arXiv:** 2509.09090 | **时间:** 2025.09
- **核心贡献:** 首个结构化量化+token pruning 联合框架，1.93x speedup + 4.5% success rate 提升

### 5.3 Layer Skipping / Architecture Efficiency

#### DySL-VLA: Dynamic-Static Layer-Skipping
- **arXiv:** 2602.22896 | **时间:** 2026.02
- **核心贡献:** 识别任务中关键步骤 vs 非关键步骤，3.75x speedup，trainable parameters 降低 85.7x

#### CogVLA: Cognition-Aligned VLA (NeurIPS 2025)
- **arXiv:** 2508.21046 | **时间:** 2025.08
- **核心贡献:** 三阶段渐进架构 + instruction-driven routing，训练成本降低 2.5x，inference latency 降低 2.8x

#### SemanticVLA: Semantic-Aligned Sparsification
- **时间:** 2025.11
- **核心贡献:** 超越 OpenVLA 21.1% success rate，同时训练成本和 inference latency 均降低 3.0x 和 2.7x

---

## 6. VLA Benchmark & Survey

### 6.1 A Survey on Efficient Vision-Language-Action Models
- **arXiv:** 2510.24795
- **作者:** Zhaoshu Yu, Bo Wang 等
- **时间:** 2025.10
- **核心贡献:** 全面综述 VLA 效率提升方法，覆盖模型设计、训练、数据收集三个维度

### 6.2 Efficient VLA Models for Embodied Manipulation: A Systematic Survey
- **arXiv:** 2510.17111
- **作者:** Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng
- **时间:** 2025.10
- **核心贡献:** 从模型架构、感知特征、动作生成、训练/推理策略四个维度分类效率方案

### 6.3 Benchmarking the Generality of VLA Models
- **arXiv:** 2512.11315
- **时间:** 2025.12
- **核心贡献:** 评估 GPT-5、Pi0、Magma 等模型在 6 个能力维度上的泛化性，发现没有模型展现一致泛化能力

---

## 7. 关键开源项目与框架

### 7.1 vLLM — Multimodal 支持进展

- **当前状态:** 积极开发 multimodal 支持
- **已支持 VLM 模型:**
  - Phi-4 Vision (microsoft/Phi-4-reasoning-vision-15B)
  - Qwen2.5-VL 系列
  - InternVL 系列
  - LLaVA-OneVision
  - NVIDIA llama-nemotron-embed-vl (embedding)
  - Cheers Multimodal Model
  - EXAONE-4.5 (开发中)
- **关键开发方向:**
  - ViT Full CUDA Graph RFC (视觉 encoder GPU 优化)
  - 多模态 processor caching
  - 跨 parallel processing ranks 的图像数据分发修复
  - GGUF 格式兼容性改进
- **挑战:** 新硬件兼容性问题 (如 Jetson Thor SM100 GPU compilation 崩溃)
- **与研究方向关联:** vLLM 是张昊的核心项目，VLM 多阶段 serving 优化是自然延伸

### 7.2 SGLang — Multimodal 支持进展

- **当前状态:** 高性能 serving 框架，支持多模态推理
- **已支持:**
  - LLaVA-OneVision (多图/视频处理)
  - Diffusion models (WAN, Qwen-Image)
  - Embedding models / Reward models
- **关键里程碑:**
  - SGLang Diffusion (2026.01 发布) — 加速视频和图像生成
  - Fast-dVLM 集成 SGLang 实现 6x+ VLM inference speedup
- **论文引用:**
  - olmOCR (2025.02) — 使用 vLLM/SGLang backend 的 PDF OCR pipeline
  - FlagEvalMM (2025.06) — 使用 vLLM/SGLang 加速 multimodal benchmark
  - Fast-dVLM (2026.04) — SGLang 集成 + FP8 量化

### 7.3 VLAgents — VLA Policy Server
- **状态:** 2026.01 发布
- **意义:** 首个模块化 VLA policy serving 框架，集成 7 种策略 (OpenVLA, Pi Zero 等)
- **关联:** 可视为 VLA 领域的 "vLLM"，是重要的 research infrastructure

### 7.4 关键 VLA 模型生态

| 模型 | 团队 | 参数量 | 特点 |
|------|------|--------|------|
| OpenVLA / OpenVLA-OFT | Stanford/Berkeley | 7B | 开源 VLA 基准模型 |
| Pi Zero (π₀) | Physical Intelligence | - | Flow matching based |
| Pi 0.5 (π₀.5) | Physical Intelligence | - | 增强版 π₀ |
| GR00T N1 | NVIDIA | - | Dual-system VLA (VL + Diffusion Transformer) |
| TinyVLA | 多所高校 | Small | 速度和数据效率优于 OpenVLA |
| SmolVLA | HuggingFace | Small | 轻量级 VLA |

---

## 8. Industry Trends

### 8.1 NVIDIA

#### GR00T N1: Open Foundation Model for Generalist Humanoid Robots
- **arXiv:** 2503.14734 | **时间:** 2025.03
- **作者:** Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da 等 (NVIDIA 团队, 含 Jim Fan, Dieter Fox, Jan Kautz)
- **核心贡献:** Dual-system 设计 — VLM 模块理解环境 + Diffusion Transformer 实时运动控制。在异质数据混合 (真实轨迹 + 人类视频 + 合成数据) 上训练，部署于 Fourier GR-1 人形机器人，语言条件双臂操作
- **关联:** NVIDIA 的 VLA 旗舰项目，dual-system 设计与 VLM serving 强相关

#### HY-Embodied-0.5 (Tencent Robotics X 联合)
- **arXiv:** 2604.07430 | **时间:** 2026.04
- **核心贡献:** 2B activated parameters 的 edge 部署版本 + 32B 大模型版本，MoT 架构 + on-policy distillation

### 8.2 Physical Intelligence

#### π₀ (Pi Zero)
- **时间:** 2024.10 (原始发布)
- **核心贡献:** Flow matching based robot foundation model，成为 VLA 领域的重要基线模型
- **后续:** 多篇论文 (ReMem-VLA, LiLo-VLA, CompliantVLA 等) 在 π₀/π₀.5 上进行改进

#### π₀.5 (Pi 0.5)
- **时间:** 2025.03
- **核心贡献:** π₀ 增强版，被广泛用作 VLA benchmark 基线

### 8.3 Song Han 组 (MIT/NVIDIA)
- **Fast-dVLM** (2026.04): 6x VLM inference speedup via block diffusion
- 持续在 efficient inference 方向有重要产出

### 8.4 LFM2 (Liquid Foundation Model 2)
- **时间:** 2025.11
- **核心贡献:** Hybrid backbone 实现 CPU 上 2x faster prefill + decode，multimodal 变体支持 accuracy-latency 可调

---

## 9. Edge Deployment

### 9.1 VLM Edge Deployment

#### HyperVL: Efficient Multimodal LLM for Edge Devices
- **时间:** 2025.12
- **核心贡献:** Visual Resolution Compressor + Dual Consistency Learning，降低移动设备上的 latency 和功耗

#### Extreme Model Compression for Edge VLMs
- **arXiv:** 2511.18504 | **时间:** 2025.11
- **核心贡献:** Sparse Temporal Token Fusion + Adaptive Neural Compression，相比 LLaVA-1.5 7B 降低 62x on-device FLOPs

#### LFM2
- **时间:** 2025.11
- **核心贡献:** CPU 上 2x faster prefill + decode

### 9.2 VLA Edge Deployment

#### Lite VLA: Efficient VLA on CPU-Bound Edge Robots
- **arXiv:** 2511.05642 | **时间:** 2025.11
- **核心贡献:** 在移动机器人嵌入式硬件上部署 compact VLM，无需云端依赖的实时场景理解和推理

#### ActionFlow (Edge Pipelining)
- **时间:** 2025.12
- **核心贡献:** OpenVLA-7B 在 edge 上 2.55x FPS 提升

### 9.3 Edge 硬件生态

- **NVIDIA Jetson Orin Nano:** 主流 edge AI 评估平台，sub-10W 功耗，多篇论文以此为 benchmark (VDPP: >43.5 FPS, LEAN-3D: sub-second latency)
- **Jetson Thor:** 下一代平台，SM100 GPU，vLLM 正在解决兼容性问题
- **关键指标:** sub-10W 功耗 + sub-100ms latency 是 edge 部署的通用目标

### 9.4 Edge-Cloud 协同

#### MSAO: Adaptive Modality Sparsity-Aware Offloading
- **arXiv:** 2604.02945 | **时间:** 2026.04
- **核心贡献:** 多模态 LLM inference 动态负载调度，端到端 latency 降低 30%，资源 overhead 降低 30-65%

---

## 10. 研究方向建议

基于以上文献梳理，以下是与 "VLM/VLA real-time systems" 方向高度相关的潜在研究机会:

### 10.1 高优先级方向

1. **VLM Multi-Stage Disaggregated Serving**
   - 现状: Nova, FlashCodec/UnifiedServe 开始探索，但仍处于早期
   - 机会: 将 vLLM 的 paged attention + continuous batching 扩展到 vision encoding + prefill + decode 的完整 pipeline
   - 优势: 与张昊教授的 vLLM 背景完美契合

2. **VLA-Aware Serving System**
   - 现状: VLAgents 是首个尝试，但仍很初级
   - 机会: 设计一个 VLA-native 的 serving system，考虑 action token 的实时性约束、multi-robot batching、temporal locality
   - 优势: 蓝海领域，系统研究者极少

3. **Vision Token 动态压缩 for Real-Time Serving**
   - 现状: 大量 offline compression 工作，但 serving-time 动态压缩很少
   - 机会: 在 serving 时根据 SLO 要求动态调整 vision token 压缩率
   - 优势: 连接 token compression (算法) 和 serving system (系统)

### 10.2 中期方向

4. **Non-Autoregressive Decoding for Real-Time VLM/VLA**
   - Fast-dVLM 展示了 6x speedup 的潜力，但仅限于 VLM
   - 将 block diffusion / parallel decoding 应用于 VLA action generation

5. **Speculative Decoding for VLA**
   - VLM speculative decoding 刚起步 (MASSV, LVSpec, ParallelVLM)
   - VLA 的 action token 有很强的时间连续性，天然适合 speculative decoding

6. **Heterogeneous Hardware VLM/VLA Serving**
   - HeteroServe 开始探索，但仅考虑 GPU 异构
   - 机会: GPU + NPU + edge device 的混合部署，特别适合 robot 场景

### 10.3 交叉方向

7. **Dual-System VLA Serving**
   - GR00T N1、StreamVLA、Dual Process VLA 都采用大小模型分离
   - 系统问题: 如何高效 serving 大小模型协同的 VLA pipeline

8. **Multi-Robot VLA Serving**
   - 多个机器人共享一个 VLA serving backend
   - 需要考虑 prefix sharing、action token batching、latency SLO

---

## 参考文献索引 (按 arXiv ID)

| arXiv ID | 简称 | 方向 |
|----------|------|------|
| 2509.21301 | Nova | VLM Serving |
| 2603.26498 | RPS-Serve | MLLM Scheduling |
| 2512.17574 | FlashCodec/UnifiedServe | Disaggregated MLLM |
| 2603.12707 | HeteroServe | Heterogeneous VLM Serving |
| 2512.22695 | Modality Inflation | Energy Optimization |
| 2603.16987 | Empirical Recipes | VLM Efficiency |
| 2512.20561 | FlashVLM | Token Selection |
| 2512.09010 | LUVC | Token Compression |
| 2512.14661 | Focus | Streaming Compression |
| 2602.12618 | ADSC | Self-Compression |
| 2508.05211 | VFlowOpt | Token Pruning |
| 2502.14882 | CalibQuant | 1-bit KV Cache |
| 2510.25600 | PureKV | KV Cache Compression |
| 2602.01901 | Q Cache | KV Cache Sharing |
| 2602.02197 | HAE | KV Cache Eviction |
| 2601.21420 | ConceptMoE | Token-to-Concept |
| 2505.10526 | MASSV | VLM Speculative Decoding |
| 2604.06832 | Fast-dVLM | Block Diffusion VLM |
| 2604.05650 | LVSpec | Video LLM Spec Decoding |
| 2603.19610 | ParallelVLM | VLM Spec Decoding |
| 2603.14989 | MMSpec | Spec Decoding Benchmark |
| 2602.01100 | StreamVLA | VLA Streaming |
| 2509.25681 | dVLA | Diffusion VLA |
| 2510.18337 | MoTVLA | Mixture VLA |
| 2602.20566 | BFA++ | VLA Token Prune |
| 2511.18950 | Compressor-VLA | VLA Token Compression |
| 2603.25766 | ETA-VLA | VLA Token Adaptation |
| 2602.22896 | DySL-VLA | VLA Layer Skipping |
| 2508.21046 | CogVLA | Cognition-Aligned VLA |
| 2509.09090 | SQAP-VLA | VLA Quant+Prune |
| 2510.24795 | Survey: Efficient VLA | VLA Survey |
| 2510.17111 | Survey: VLA Manipulation | VLA Survey |
| 2503.14734 | GR00T N1 | NVIDIA VLA |
| 2511.05642 | Lite VLA | Edge VLA |
| 2604.02945 | MSAO | Edge-Cloud MLLM |
| 2505.22038 | Balanced Token Pruning | NeurIPS 2025 |
| 2503.21817 | Skip-Vision | ICCV 2025 |
| 2506.02850 | METok | EMNLP 2025 |
| 2505.19812 | EMLoC | ICML 2025 |
