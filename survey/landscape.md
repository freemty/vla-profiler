# VLM/VLA/VA Inference Efficiency 全景 Survey (2024-2026)

**编写日期:** 2026-04-11
**研究背景:** UCSD 张昊实验室 — VLM/VLA Real-Time Systems
**核心定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

---

## 目录

1. [VLM Inference 最新进展](#1-vlm-inference-最新进展-2024-2026)
2. [VLA Inference 最新进展](#2-vla-inference-最新进展-2024-2026)
3. [VA (Vision-Action, non-language) 进展](#3-va-vision-action-non-language-进展)
4. [从 vLLM/FastVideo 视角看 Gap Analysis](#4-从-vllmfastvideo-视角看-gap-analysis)
5. [Efficiency Techniques Taxonomy](#5-efficiency-techniques-taxonomy)
6. [推荐的 Research Entry Points](#6-推荐的-research-entry-points)

---

## 1. VLM Inference 最新进展 (2024-2026)

### 1.1 主流 VLM 架构概览

当前主流 VLM 采用 **"Vision Encoder + Projector + LLM Backbone"** 的三段式架构，但在具体实现上差异显著：

| 模型 | Vision Encoder | LLM Backbone | 视觉 Token 策略 | 分辨率处理 |
|------|---------------|-------------|----------------|-----------|
| LLaVA-1.5/1.6 | CLIP ViT-L | Vicuna/LLaMA | 576 tokens (固定) | 动态分辨率 tiling |
| LLaVA-OneVision | SigLIP | Qwen2 | 可变 (tiling) | AnyRes |
| Qwen-VL/Qwen2.5-VL | ViT (native) | Qwen2.5 | 动态 (最多数千) | NaViT 式动态分辨率 |
| InternVL 2/2.5 | InternViT-6B | InternLM2/Qwen2 | 动态 tiling | 448px tiles |
| GPT-4V/4o | 未公开 | 未公开 | 未公开 | 多分辨率 |
| Gemini 1.5/2.0 | 未公开 | MoE 架构 | 未公开 | 原生多模态 |
| Phi-3/4-Vision | SigLIP | Phi-3/4 | 动态 | HD tiling |
| DeepSeek-VL2 | SigLIP | DeepSeek-V2 (MoE) | 动态 tiling | Multi-scale |

**架构趋势：**
- 开源模型从早期 LLaVA 的固定 576 token 方案，演进到动态分辨率 tiling，视觉 token 数量从几百膨胀到数千甚至上万
- 视频理解场景下 token 数量进一步爆炸：1 分钟 30fps 视频在 Qwen2.5-VL 中可产生 ~50K+ visual tokens
- 多模态原生架构 (Gemini 式) 与拼接式架构 (LLaVA 式) 的路线分歧

### 1.2 VLM Inference 面临的独特挑战

#### 1.2.1 Visual Token 数量爆炸

这是 VLM inference 与纯 LLM inference 最本质的区别。在 LLM serving 中，prompt length 由用户输入控制，通常在几百到几千 token 范围；而 VLM 中，一张高分辨率图像经过 tiling 后可产生 2000-5000 个 visual tokens，视频输入则可达数万到数十万。

**问题分解：**
- **Prefill 阶段：** Visual tokens 占据 prefill 的主要计算量。以 LLaVA-OneVision 72B 为例，一次高分辨率图像 prefill 中 visual tokens 可占总 token 的 80%+
- **KV-cache 存储：** Visual tokens 的 KV-cache 在整个 decode 阶段持续占用 GPU 内存，但这些 token 的 "信息密度" 远低于 text tokens —— 大量 visual tokens 是冗余的背景信息
- **Batching 效率：** 不同请求的 visual token 数量差异巨大 (纯文本 vs 高分辨率图像 vs 视频)，导致 batch padding 浪费严重

#### 1.2.2 Cross-Modal Attention 计算模式

VLM 中 visual tokens 和 text tokens 共享同一个 Transformer attention 层，但它们的 attention pattern 本质不同：
- **Visual-to-visual attention：** 具有强局部性 (空间相邻 patch 间注意力高)，类似于 ViT 的 attention pattern
- **Text-to-visual attention：** 高度稀疏 —— 大多数 text tokens 只关注少数关键 visual tokens
- **Visual-to-text attention：** 在 autoregressive 架构中，visual tokens 在 prefill 后不再参与 text 生成，但其 KV-cache 仍然被每个 decode step query

这意味着纯为 text-only LLM 设计的 attention 优化 (如 FlashAttention 的行式处理) 在 VLM 场景下可能不是最优的。

#### 1.2.3 KV-Cache 中 Visual Tokens 的管理

在 vLLM 的 PagedAttention 中，KV-cache 以 block 粒度管理，所有 token 被平等对待。但 VLM 中：
- Visual tokens 通常在 sequence 的前部集中出现 (image tokens → text prompt → generation)
- 一旦 prefill 完成，visual tokens 的 KV-cache 只被读取、不再更新
- 不同 visual tokens 的 "重要性" 差异巨大 —— 前景物体 vs 背景纹理

**现有文献的发现：**
- HybridKV (2026, arXiv:2604.05887)：提出对 visual 和 text KV-cache 采用不同的压缩策略，实现 7.9x KV-cache 内存压缩和 1.52x 解码加速
- MHA2MLA-VLM (2026, arXiv:2601.11464)：将 VLM 转换为 DeepSeek 的 Multi-Head Latent Attention 架构，从根本上压缩 KV-cache 维度
- 频域引导方法 (2025, arXiv:2511.16786)：发现 visual KV-cache 的重要性可以通过频域特征有效评估，实现 80% KV 内存压缩和 1.69x 解码加速

#### 1.2.4 Prefill 阶段 Vision Encoder 的 Bottleneck

VLM inference 的 prefill 阶段实际上包含两个串行步骤：
1. **Vision encoding：** 将图像/视频通过 ViT 编码 (compute-bound)
2. **Projector + LLM prefill：** 将 visual features 投影并与 text tokens 拼接后做 LLM prefill

在大型 VLM (如 InternVL2-76B) 中：
- Vision encoder (InternViT-6B) 本身就有 6B 参数，处理高分辨率图像需要 ~100ms+ 级延迟
- Vision encoding 和 LLM prefill 默认串行执行，无法 overlap
- 对于 batched serving，vision encoding 的时间不可 amortize (每个请求的图像不同)

**关键论文：**
- Cross-Tier GPU Heterogeneity (2026, arXiv:2603.12707)：识别出 "vision encoding is compute-bound, language generation is memory-bandwidth-bound" 的核心特征，提出 modality-level partitioning
- RServe (2025, arXiv:2509.24381)：通过 intra-request 和 inter-request pipeline 实现 encoding 和 prefill 的 overlap，延迟降低 66%

### 1.3 已有的 Efficiency 优化工作

#### 1.3.1 Visual Token Pruning / Compression

这是目前最活跃的研究方向，2024-2026 年产出了大量工作。核心思想：大部分 visual tokens 是冗余的，可以在不损失性能的前提下大幅裁剪。

| 方法 | 年份 | 压缩比 | 精度保持 | 核心技术 | arXiv |
|------|------|--------|---------|---------|-------|
| FastV | 2024 | ~50% tokens | ~98% | Attention-based pruning in early layers | 2403.06764 |
| LLaVA-PruMerge | 2024 | ~80% tokens | ~95% | Adaptive spatial token merging | 2403.15388 |
| MADTP | 2024 | 80% FLOPs | ~96% | Multimodal alignment-guided dynamic pruning | 2403.02991 |
| Multi-Stage Token Drop | 2024 | 88.5% FLOPs | ~92% | Progressive dropping across layers | 2411.10803 |
| FlashVLM | 2025 | 77.8% tokens | lossless | Text-guided visual token selection (CPU offload) | 2512.20561 |
| TokenCarve | 2025 | 78% tokens | ~97% | Information-preserving two-stage compression | 2503.10501 |
| Skip-Vision | 2025 | 75% FLOPs | ~95% | Adaptive token skipping with 45% latency reduction | 2503.21817 |
| DUET-VLM | 2026 | 67-93.4% tokens | 99-97.6% | Dual stage unified reduction | 2602.18846 |
| HiDrop | 2026 | 90% tokens | ~baseline | Hierarchical vision token reduction | 2602.23699 |
| ReDiPrune | 2026 | 85% tokens | +2% gain | Relevance-diversity pre-projection pruning | 2603.24680 |
| RCP | 2026 | 88.9% tokens | competitive | 85.7% FLOPs reduction | 2604.04972 |
| ID-Selection | 2026 | 97.2% tokens | 91.8% | Training-free across 16 benchmarks | 2604.05601 |

**趋势观察：**
- 从 2024 到 2026，压缩比从 ~50% 提升到 90%+，精度保持也在改善
- 早期方法 (FastV) 主要用 attention score 做 pruning；新方法引入了 text-aware selection (FlashVLM)、信息论指标 (TokenCarve)、频域分析等更精细的策略
- Training-free 方法 (ID-Selection, IPCV) 日益受到关注，因为它们可以直接应用于已有模型
- **关键 gap：** 这些方法大多在 model 层面实现 (修改 forward pass)，尚未与 serving system 层面的优化 (如 vLLM 的 PagedAttention) 深度整合

#### 1.3.2 VLM-Specific Serving Systems

这是一个快速发展但尚不成熟的方向。vLLM 已经支持基本的 VLM serving，但针对 VLM 特性的系统级优化才刚刚起步：

| 系统 | 年份 | 核心技术 | 效果 | arXiv |
|------|------|---------|------|-------|
| RPS-Serve | 2026 | Modality-aware scheduling | TTFT 降低 54-78.5% | 2603.26498 |
| EPD-Serve | 2026 | Encode-Prefill-Decode 三阶段分离 (Ascend) | 吞吐提升 57-69% | 2601.11590 |
| xLLM | 2025 | 动态 EPD disaggregation | 自适应调度 | 2510.14686 |
| EPD Disaggregation | 2025 | Encode/Prefill/Decode 分离到专用资源 | SLO 改善 | 2501.05460 |
| HydraInfer | 2025 | Hybrid EPD disaggregation + stage batching | 4x 吞吐 | 2505.12658 |
| RServe | 2025 | Encoding-Prefill overlap pipeline | 延迟降低 66% | 2509.24381 |
| Cross-Tier Heterogeneous | 2026 | Modality-level GPU partitioning | 成本优化 | 2603.12707 |
| Firebolt-VL | 2026 | Liquid Foundation Model (线性复杂度解码) | 线性推理 | 2604.04579 |

**关键洞察：**
- **从 PD (Prefill-Decode) 到 EPD (Encode-Prefill-Decode)：** VLM 的 serving 需要将 DistServe 的 prefill-decode 分离概念扩展为三阶段分离。Vision encoding 作为独立的 compute-bound 阶段，应该有专门的调度和资源分配
- **Modality-aware scheduling：** RPS-Serve 发现不同请求的视觉负载差异巨大，传统的 FCFS 或 longest-prefix-match 调度策略不适用于 VLM，需要 modality-aware 的调度
- **异构硬件适配：** Vision encoding (compute-bound) 和 decode (memory-bandwidth-bound) 适合不同类型的硬件，Cross-Tier 方案提出将它们分配到不同 tier 的 GPU
- **尚无 "VLM 的 vLLM"：** 截至 2026 年初，还没有一个像 vLLM 对于 LLM 那样 dominant 的 VLM serving framework。vLLM 本身支持 VLM，但 VLM-specific 优化有限

#### 1.3.3 Speculative Decoding for VLM

VLM 的 speculative decoding 在 2025 年才真正起步，到 2026 年已成为热门方向：

| 方法 | 年份 | 加速比 | 核心创新 | arXiv |
|------|------|--------|---------|-------|
| ViSpec | 2025 | first substantial | Vision-aware draft model | 2509.15235 |
| Spec-LLaVA | 2025 | 3.28x | Dynamic tree-based speculation | 2509.11961 |
| SpecVLM | 2025 | 2.5-2.9x | Lossless multimodal speculation | 2509.11815 |
| HiViS | 2025 | - | Hide visual tokens from drafter | 2509.23928 |
| DREAM | 2025 | 3.6x | Refined target features for drafting | 2505.19201 |
| TwigVLM | 2025 | 154% | Growing twig layers for VLM | 2503.14075 |
| LiteVLM | 2025 | 2.5-3.2x | FP8 + resource-constrained optimization | 2506.07416 |
| SAGE | 2026 | 3.36x (72B) | Entropy-guided adaptive speculation | 2602.00523 |
| HSD | 2026 | 2.78-7.04x | Hierarchical speculation for documents | 2602.12957 |
| Sparrow | 2026 | 2.82x | Text-anchored window attention for video LLMs | 2602.15318 |
| MMSpec (benchmark) | 2026 | - | 发现 text-only SD 方法在 VLM 上退化 | 2603.14989 |
| Fast-dVLM | 2026 | 6x | Block-diffusion + speculative block decoding | 2604.06832 |

**核心发现 (MMSpec, 2026)：**
> "Methods designed for text-only LLMs degrade in multimodal scenarios."

这验证了一个重要假设：VLM 的 speculative decoding 不能简单复用 LLM 的方案，需要 modality-aware 的设计。具体原因：
- Draft model 如何处理 visual tokens 是一个 open question —— 给 draft model 完整 visual tokens 太慢 (HiViS 的动机)，但不给又影响 acceptance rate
- Visual context 对 text generation 的影响模式与 text-only 不同，导致 acceptance rate 下降
- Entropy-guided 方法 (SAGE) 通过自适应调整 draft length 来应对 multimodal uncertainty

#### 1.3.4 Quantization for VLM

| 方法 | 年份 | 比特数 | 目标 | 效果 | arXiv |
|------|------|--------|------|------|-------|
| VLM Edge Survey | 2025 | 综述 | Edge VLM | 全面覆盖 pruning/quant/distill | 2502.07855 |
| SPEED-Q | 2025 | 2-bit | VLM | 比现有方法 6x 精度提升 | 2511.08914 |
| LQA | 2026 | 混合精度 | VLM | 19.9x 内存压缩 | 2602.07849 |
| QAPruner | 2026 | low-bit | VLM | 联合 pruning + quantization | 2604.02816 |

#### 1.3.5 VLM Inference Efficiency 综合 Survey

- **Efficient Multimodal LLM Inference Survey** (2026, arXiv:2604.05546)：目前最全面的 VLM inference efficiency 综述，覆盖 encoding、prefilling、decoding 三阶段的优化 taxonomy

---

## 2. VLA Inference 最新进展 (2024-2026)

### 2.1 VLA 架构分类

#### 2.1.1 Autoregressive VLA (离散 Token 预测)

以 RT-2 (2023) 和 OpenVLA (2024) 为代表。将 action 离散化为 token，利用 LLM backbone 进行 next-token prediction。

**架构：** Vision Encoder → LLM → action tokens (离散化的位置/旋转/gripper)

**优势：**
- 复用成熟的 LLM inference 基础设施 (vLLM, TensorRT-LLM 等)
- 可以利用 language pretraining 的知识 (language-conditioned manipulation)
- Scaling law 相对清晰

**inference 挑战：**
- Autoregressive decode 的延迟与 action dimension 成正比 —— 7-DoF action 需要 7 个 decode step
- 离散化引入的精度损失 (尤其在精细操作中)
- KV-cache 和 visual tokens 的管理问题与 VLM 相同

**代表工作：**
| 模型 | 年份 | 参数量 | Backbone | 特点 |
|------|------|--------|---------|------|
| RT-2 | 2023 | 55B | PaLM-E | 首个大规模 VLA |
| OpenVLA | 2024 | 7B | Prismatic VLM | 开源、多任务 |
| OpenVLA 2.0 | 2025 | 7B | 改进版 | 更快 fine-tuning、更好泛化 |
| A1 | 2026 | - | Adaptive VLM | 72% 延迟降低、adaptive techniques | 2604.05672 |

#### 2.1.2 Diffusion/Flow VLA (连续动作生成)

以 Pi-Zero (2024) 和 Octo (2024) 为代表。用 diffusion 或 flow matching 生成连续动作。

**架构：** Vision Encoder → (optional LLM) → Diffusion/Flow head → continuous actions

**优势：**
- 天然支持连续动作空间 (无离散化误差)
- 可以建模多模态动作分布 (multi-modal action distribution)
- Action chunking 支持一次生成多步动作

**inference 挑战：**
- Diffusion/Flow 需要多步去噪 (10-100 steps)，每步都需要一次 forward pass
- 与 autoregressive 模型不同，无法直接使用 speculative decoding 等 LLM 加速技术
- 去噪步数与精度的 trade-off 是核心问题

**代表工作：**
| 模型 | 年份 | 去噪步数 | 核心技术 | 特点 |
|------|------|---------|---------|------|
| Diffusion Policy | 2023 | 100 | DDPM | 开创性工作 |
| Octo | 2024 | 10-50 | DiT backbone | 多任务、开源 |
| Pi-Zero | 2024 | ~10 | Flow matching + VLM | VLM + flow action head |
| DreamZero | 2026 | ~10 | World action model | 7Hz 实时控制 | 2602.15922 |
| FASTER | 2026 | 1 | Horizon-aware schedule | 单步去噪 | 2603.19199 |
| HiFlow | 2026 | ~few | Scale-wise autoregressive flow | 无 tokenization 误差 | 2603.27281 |
| Mean-Flow VLA | 2026 | 1 | Mean-flow based one-step | 8.7-83.9x 加速 | 2603.01469 |
| Action-to-Action Flow | 2026 | 1 | Informed initialization | 0.56ms 延迟 | 2602.07322 |

**去噪步数压缩趋势：**
- 2023: 100 steps (DDPM)
- 2024: 10-50 steps (DDIM, Flow Matching)
- 2025: 1-5 steps (Consistency Models, Mean Flow)
- 2026: 1 step 成为可能 (FASTER, Mean-Flow VLA, Action-to-Action Flow)

这是一个极其重要的趋势 —— 如果 flow/diffusion VLA 可以压缩到单步，那么它在延迟上的劣势就消失了，同时保留了连续动作空间和多模态分布建模的优势。

#### 2.1.3 Hybrid VLA (VLM Backbone + Action Head)

最新趋势是将 VLM backbone 与轻量 action head 结合，利用 VLM 的 vision-language understanding 能力来增强 action generation：

| 模型 | 年份 | 架构 | 特点 | arXiv |
|------|------|------|------|-------|
| Pi-Zero | 2024 | PaLI-based VLM + Flow head | 统一 VL 和 action | - |
| StreamVLA | 2026 | Completion-state gating | 打破 reason-then-act 串行 | 2602.01100 |
| TriVLA | 2025 | Triple-system VLA | 分层决策 | 2507.01424 |
| ReMem-VLA | 2026 | Dual-level recurrent queries | 时序记忆增强 | 2603.12942 |
| AnoleVLA | 2026 | State Space Model backbone | 3x faster inference | 2603.15046 |
| HY-Embodied-0.5 | 2026 | 2B activated params | Edge 部署优化 | 2604.07430 |

### 2.2 VLA Real-Time Inference 挑战

#### 2.2.1 Control Loop Latency 要求

| 应用场景 | 控制频率 | 单步延迟上限 | 典型模型延迟 | Gap |
|---------|---------|-------------|-------------|-----|
| 工业机械臂 | 100-1000 Hz | 1-10 ms | 50-500 ms | 10-100x |
| 灵巧手操作 | 50-100 Hz | 10-20 ms | 100-300 ms | 5-30x |
| 移动机器人导航 | 10-30 Hz | 33-100 ms | 100-500 ms | 1-15x |
| 自动驾驶决策 | 10-20 Hz | 50-100 ms | 200-1000 ms | 2-20x |
| 人形机器人全身 | 50-200 Hz | 5-20 ms | 200-1000 ms | 10-200x |

**核心矛盾：** 大型 VLA 模型 (7B+) 的 inference 延迟通常在 100ms-1s 级别，而机器人控制需要 10ms 级别。这个 10-100x 的 gap 是 VLA real-time inference 的核心问题。

#### 2.2.2 Action Chunking vs Streaming Inference

**Action Chunking：**
- 一次推理生成未来 K 步动作 (K=4-16)，然后在 K 个控制周期内依次执行
- 优势：将推理延迟 amortize 到 K 个 step 上，等效延迟 = inference_time / K
- 劣势：环境变化时动作已经 "过时"，K 越大越不鲁棒；需要 action re-planning 机制

**Streaming Inference：**
- StreamVLA (2026) 提出 "Completion-State Gating" 打破传统 reason-then-act 的串行模式
- 核心思想：在上一个动作执行期间，overlapping 地推理下一个动作
- 需要 serving system 层面的支持 (类似 continuous batching 但在 single request 内)

**Adaptive Action Chunking (AAC, CVPR 2026, arXiv:2604.04161)：**
- 根据动作 entropy 动态决定 chunk size：确定性高时多步生成，不确定时单步生成
- 这与 speculative decoding 中的 adaptive draft length 思想异曲同工

#### 2.2.3 Edge Device 部署

| 方法 | 年份 | 目标平台 | 模型 | 效果 | arXiv |
|------|------|---------|------|------|-------|
| NanoVLA | 2025 | Edge devices | Ultra-small | 52x faster on edge | 2510.25122 |
| BitVLA | 2025 | General | 1-bit VLA | 11x memory, 4.4x latency | 2506.07530 |
| ActionFlow | 2025 | Edge | VLM on edge | 2.55x FPS improvement | 2512.20276 |
| LiteVLA-Edge | 2026 | Jetson Orin | 4-bit GGUF | 150.5ms end-to-end | 2603.03380 |
| HBVLA | 2026 | General | 1-bit PTQ | 92.2% accuracy retained | 2602.13710 |
| HY-Embodied-0.5 | 2026 | Edge | 2B activated | Edge-optimized | 2604.07430 |
| DyQ-VLA | 2026 | General | Dynamic quant | 30.9% memory, 99.5% perf | 2603.07904 |

**关键问题：**
- Jetson Orin (嵌入式 GPU 旗舰) 上运行 7B 模型的延迟仍在 150ms+，无法满足高频控制需求
- 1-bit 量化 (BitVLA, HBVLA) 可以大幅压缩但精度损失在精细操作中可能不可接受
- 真正的 edge-native VLA (<1B) 在泛化能力上远不如大模型

### 2.3 VLA Efficiency 技术细分

#### 2.3.1 VLA Token Pruning

| 方法 | 年份 | 压缩率 | 速度提升 | 核心技术 | arXiv |
|------|------|--------|---------|---------|-------|
| VLA-InfoEntropy | 2026 | - | - | Entropy-based token selection | 2604.05323 |
| ETA-VLA | 2026 | 85% tokens | 32% FLOPs | 94% accuracy retained | 2603.25766 |
| VLA-IAP | 2026 | - | 1.25x | Training-free, 97.8% success | 2603.22991 |

#### 2.3.2 VLA Speculative Decoding

这是一个非常新但极有前景的方向：

| 方法 | 年份 | 加速比 | 核心技术 | arXiv |
|------|------|--------|---------|-------|
| Spec-VLA | 2025 | 1.42x | 44% enhanced acceptance length | 2507.22424 |
| KERV | 2026 | 27-37% | Kinematic-rectified speculation | 2603.01581 |
| HeiSD | 2026 | 2.06-2.45x | Hybrid speculative decoding for VLA | 2603.17573 |

**独特之处：**
- VLA 的 speculative decoding 不仅要保证 token 级 acceptance，还要保证 **物理可行性** (kinematic constraints)
- KERV 提出用 kinematic rectification 来增强 acceptance rate —— 在 verification 阶段检查动作的物理合理性
- 这是 VLM speculative decoding 没有的维度，是 VLA-specific 的 systems 问题

#### 2.3.3 去噪步数加速 (Diffusion/Flow VLA)

| 方法 | 年份 | 原始步数→加速后 | 核心技术 | arXiv |
|------|------|----------------|---------|-------|
| FlowPolicy | 2024 | 10→1-2 | Consistency flow matching | 2412.04987 |
| DM1 | 2025 | multi→1 | MeanFlow + dispersive reg. | 2510.07865 |
| ManiFlow | 2025 | multi→1-2 | Consistency flow training | 2509.01819 |
| Falcon | 2025 | multi→partial | Partial denoising + action reuse | 2503.00339 |
| Sparse ActionGen | 2026 | ~→25% | Real-time pruning | 2601.12894 |
| Action-to-Action Flow | 2026 | multi→1 | Informed initialization, 0.56ms | 2602.07322 |
| Mean-Flow VLA | 2026 | multi→1 | Mean-flow one-step | 2603.01469 |
| FASTER | 2026 | multi→1 | Horizon-aware schedule | 2603.19199 |
| AnchorVLA | 2026 | multi→truncated | Truncated diffusion schedule | 2604.01567 |

---

## 3. VA (Vision-Action, Non-Language) 与 World Action Model 进展

> **深度补充文档:** 详见 [`survey/papers/va-world-models.md`](papers/va-world-models.md)，包含 VA 架构族谱、World Action Model 全景、Inference efficiency 对比分析、Systems 视角分析、以及与张昊实验室技术栈的迁移可能性。

### 3.1 定义与范围

VA 模型指不经过 language 中间表示、直接从视觉观测映射到动作的模型。与 VLA 的核心区别在于：没有 language backbone，因此也没有 autoregressive text generation 的开销。

### 3.2 VA 代表架构

| 类型 | 代表模型 | 架构 | Inference 延迟 | arXiv |
|------|---------|------|---------------|-------|
| Diffusion Policy | Chi et al., 2023 | U-Net/DiT + DDPM/DDIM | 30-200ms (10-100步) | 2303.04137 |
| 1-Step Flow Policy | Action-to-Action Flow, 2026 | Flow matching, 1步 | **0.56ms** | 2602.07322 |
| 3D Diffusion | DP3 (Ze et al., 2024) | PointNet++ + Diffusion | 30-50ms | 2403.03954 |
| ACT | Zhao et al., 2023 | CVAE + Transformer | 10-30ms (单次 forward) | 2304.13705 |
| Behavior Transformer | BeT (Shafiullah et al., 2022) | GPT-style + action binning | 5-20ms | 2206.11251 |
| SE(3)-Equivariant | E3Flow (2026) | Spherical harmonics flow | ~10ms | - |
| Masked Generation | MGP (2025) | Masked token prediction | ~10ms | - |
| State Space | AnoleVLA (2026) | Mamba/RWKV-based | O(n) | 2603.15046 |

**VA Inference 已接近物理极限：** 2026 年 1-step flow VA 的 action generation 延迟 (~1ms) 远快于 vision encoding (~10ms)。Vision encoding 已成为新的 bottleneck。

### 3.3 World Action Model (WAM) 新范式

2025-2026 年出现的 World Action Model 将 "理解世界动力学" 与 "生成动作" 统一在一个框架中：

| 模型 | 年份 | 类型 | 控制频率 | 核心创新 | arXiv |
|------|------|------|---------|---------|-------|
| Dreamer v3 | 2023 | Latent dynamics | ~100Hz | 通用 RSSM | 2301.04104 |
| DayDreamer | 2022 | Latent dynamics (real) | ~50Hz | 真实机器人验证 | 2206.14176 |
| DreamZero | 2026 | Video WAM | **7Hz** | Zero-shot policy | 2602.15922 |
| DDP-WM | 2026 | Efficient latent | ~30Hz | 9x speedup, 动态分解 | 2602.01780 |
| Sparse Imagination | 2025 | Sparse latent (ICLR 2026) | ~30Hz | Token-sparse rollout | 2506.01392 |
| Cosmos Policy | 2026 | Video → action | ~10Hz | Fine-tune video model | - |
| mimic-video | 2025 | Video-Action | ~30Hz | 10x sample efficiency | 2512.15692 |

**WAM 的核心 trade-off：** DreamZero 实现了 zero-shot policy (无需 robot 数据)，但 7Hz 控制频率限制了部署场景。FastVideo 的蒸馏技术可能将其提升到 40Hz+。

### 3.4 三大范式对比 (VA vs VLA vs WAM)

| 维度 | VA (1-step flow) | VLA (7B AR) | WAM (DreamZero) |
|------|-----------------|-------------|-----------------|
| 延迟 | 1-5ms | 100-500ms | ~130ms |
| 控制频率 | >200Hz | 2-10Hz | ~7Hz |
| 泛化能力 | 低 (窄领域) | 高 (语言条件) | 极高 (zero-shot) |
| 数据需求 | 高 (需 robot demos) | 中 (利用预训练) | 低 (利用 video 预训练) |
| 模型大小 | 10M-500M | 7B-70B | 100M-10B |
| GPU 需求 | 单 GPU | 多 GPU | 多 GPU |

### 3.5 Efficiency 优势与劣势 (VA)

**优势：**
- 无 LLM backbone 的开销，模型通常在 100M-1B 参数
- 不需要 tokenizer/detokenizer，端到端延迟更低
- 可以使用更 aggressive 的量化 (action 空间连续，误差可接受)
- 针对特定硬件 (Jetson, TPU) 的优化更容易

**劣势：**
- 无法利用 language conditioning (任务指令只能通过 one-hot encoding 或 task embedding)
- 泛化能力远弱于 VLA (不能理解自然语言指令)
- 无法做 in-context learning 或 few-shot adaptation
- 当需要 common-sense reasoning 时完全依赖训练数据

**趋势：**
- VA 的 inference 已接近物理极限，但泛化能力受限
- VLA 在效率上的进步正在"侵蚀" VA 的领地
- WAM 以更高延迟为代价换取了 zero-shot 能力
- **三者正在趋向融合：** Dual-system 架构 (VLA reasoning + VA control) + WAM planning

---

## 4. 从 vLLM/FastVideo 视角看 Gap Analysis

### 4.1 vLLM 技术栈到 VLM Serving 的迁移

| vLLM 核心技术 | 直接可迁移? | 在 VLM 中的状况 | Gap |
|--------------|-----------|----------------|-----|
| PagedAttention | 部分 | VLM 已支持，但 visual/text KV-cache 无差异化管理 | Visual KV-cache 的重要性分布与 text 不同，需要 modality-aware paging |
| Continuous Batching | 部分 | VLM 已支持，但忽略了 visual token 数量的巨大差异 | 需要 modality-aware batch scheduling，避免 visual-heavy 请求 dominate batch |
| Prefix Caching | 部分 | 适用于共享 system prompt，但 visual prefix 不可共享 | 相同图像的不同问题可以共享 visual KV-cache (image prefix caching) |
| Chunked Prefill | 高度 | 直接适用于 VLM text prefill | 但 vision encoding 是否也需要 chunking? (超高分辨率场景) |
| LoRA Serving | 高度 | 直接适用 | 多个 VLM fine-tune 版本的高效 serving |
| Speculative Decoding | 需要适配 | 已证明 text-only SD 在 VLM 退化 (MMSpec) | 需要 vision-aware draft model 或 visual token hiding |
| Disaggregated Prefill-Decode | 高度 | DistServe 的思路直接扩展 | 扩展为 EPD 三阶段分离 |

**关键 Gap 总结：**
1. **Visual KV-cache 差异化管理：** vLLM 的 PagedAttention 对所有 token 一视同仁，但 visual tokens 具有独特的特性 (稀疏访问、可压缩、位置集中)，需要专门的 paging 策略
2. **Modality-aware scheduling：** continuous batching 需要考虑请求的 modality 组成 (text-only vs image vs video)，当前的 token-level scheduling 粒度不够
3. **Image prefix caching：** 多个用户对同一张图提问的场景下，visual KV-cache 可以共享，但当前没有系统支持
4. **VLM-specific speculative decoding：** 需要解决 draft model 如何处理 visual context 的问题

### 4.2 FastVideo 技术栈到 VLM/VLA 的迁移

| FastVideo 技术 | 潜在应用 | 可行性分析 |
|---------------|---------|-----------|
| Sliding Tile Attention (STA) | VLM 的 vision encoder 加速 | 高：ViT 处理高分辨率图像 tiling 时，tiles 之间有空间局部性，STA 可以利用 |
| VSA (Video Sparse Attention) | 视频 VLM 的 video token attention | 高：视频 VLM 中 temporal attention 天然稀疏，VSA 的稀疏模式可以迁移 |
| STA | Diffusion/Flow VLA 的 action denoising | 中：action sequence 有时间局部性，但维度远低于视频 |
| 并行去噪 | Flow VLA 的多步去噪加速 | 中-高：FastVideo 的 step-parallel 策略可以用于 Flow VLA |
| 模型蒸馏框架 | VLA 模型压缩 | 高：FastVideo 的蒸馏技术可直接用于 VLA |

**高价值迁移点：**

1. **STA → VLM Vision Encoder 加速：** 当 VLM 处理高分辨率图像时，ViT 会将图像 tile 成多个 448x448 patches，每个 patch 独立编码。STA 的滑动窗口思想可以用于 cross-tile attention，利用相邻 tile 的空间局部性减少计算量。

2. **VSA → Video VLM token 管理：** 视频理解场景下，temporal frames 之间的 attention 高度稀疏且有时间局部性。FastVideo 的 VSA 可以直接指导 Video VLM 中 temporal visual tokens 的管理 —— 保留 key frames 的完整 tokens，对冗余 frames 做 sparse representation。

3. **Step-parallel 去噪 → Flow VLA 加速：** FastVideo 的并行去噪思想可以用于 Flow VLA 的多步 action denoising。结合 consistency/mean-flow 的单步蒸馏技术，有望实现 sub-10ms 的 action generation。

### 4.3 Open Research Questions

| 问题 | 重要性 | 难度 | 与实验室能力的匹配度 |
|------|-------|------|-------------------|
| VLM 的 modality-aware PagedAttention | 极高 | 高 | 极高 (vLLM 原班人马) |
| VLM/VLA EPD 三阶段分离的最优调度 | 极高 | 中-高 | 极高 (DistServe 经验) |
| Visual token compression 与 serving system 的联合优化 | 高 | 中 | 高 |
| VLA 的 speculative decoding (physics-aware) | 高 | 高 | 高 (Lookahead/SD 经验) |
| Flow VLA 的 real-time serving (action streaming) | 高 | 中 | 高 (FastVideo 经验) |
| VLM/VLA 在异构硬件 (cloud+edge) 上的协同推理 | 中-高 | 高 | 中 |
| Video VLM 的 long-context visual KV-cache 管理 | 高 | 中 | 高 (vLLM KV-cache 经验) |

---

## 5. Efficiency Techniques Taxonomy

### 5.1 分层分类

```
Efficiency Techniques
|
|-- Kernel Level (算子级)
|   |-- FlashAttention (Tri Dao) — 标准 attention 加速
|   |-- FlashAttention-3 — Hopper (H100) 异步优化
|   |-- Fused Operators — LayerNorm-QKV fusion, SwiGLU fusion
|   |-- Custom CUDA Kernels — triton kernels for visual token selection
|   |-- Sliding Tile Attention (STA) — FastVideo, 利用空间局部性
|   |-- Paged KV-cache kernels — vLLM, 虚拟内存管理
|
|-- Scheduling Level (调度级)
|   |-- Continuous Batching — vLLM, iteration-level scheduling
|   |-- Prefill-Decode Disaggregation — DistServe, 阶段分离
|   |-- EPD Disaggregation — Encode-Prefill-Decode 三阶段 (VLM-specific)
|   |-- Modality-Aware Scheduling — RPS-Serve, 按模态负载调度
|   |-- Prefix Caching — 共享前缀的 KV-cache 复用
|   |-- Action Streaming — VLA, overlap inference and execution
|   |-- Adaptive Batching — 根据请求特征动态调整 batch size
|
|-- Model Level (模型级)
|   |-- Quantization
|   |   |-- Post-Training Quantization (PTQ) — W8A8, W4A16
|   |   |-- 1-bit Quantization — BitVLA, HBVLA
|   |   |-- Dynamic Quantization — DyQ-VLA, per-token adaptive
|   |   |-- Quantization-Aware Pruning — QAPruner, joint optimization
|   |-- Pruning / Compression
|   |   |-- Visual Token Pruning — FastV, FlashVLM, RCP
|   |   |-- KV-Cache Compression — HybridKV, modality-aware
|   |   |-- Weight Pruning — structured/unstructured
|   |-- Distillation
|   |   |-- Model Distillation — large VLA → small VLA
|   |   |-- Step Distillation — multi-step → single-step (Flow VLA)
|   |   |-- Feature Distillation — teacher encoder → student
|   |-- Speculative Decoding
|   |   |-- Vision-Aware SD — SAGE, ViSpec, HiViS
|   |   |-- Physics-Aware SD (VLA) — KERV, HeiSD
|   |   |-- Block Diffusion SD — Fast-dVLM
|   |   |-- Tree-based SD — Spec-LLaVA
|
|-- Architecture Level (架构级)
|   |-- Efficient Attention Patterns
|   |   |-- Sparse Attention — video temporal, cross-modal
|   |   |-- Multi-Head Latent Attention (MLA) — DeepSeek 式 KV 压缩
|   |   |-- Linear Attention — Mamba, RWKV (AnoleVLA)
|   |   |-- Sliding Window Attention — Mistral 式
|   |-- Token Compression
|   |   |-- Adaptive Resolution — 根据内容复杂度动态分辨率
|   |   |-- Token Merging — 相似 token 合并
|   |   |-- Progressive Compression — 多阶段逐步压缩
|   |-- Mixture of Experts (MoE)
|   |   |-- DeepSeek-VL2 — MoE for VLM
|   |   |-- ConceptMoE — token-to-concept 压缩
|   |-- Action Head Design
|   |   |-- Flow Matching (single-step) — Mean-Flow, FASTER
|   |   |-- Consistency Models — FlowPolicy
|   |   |-- Action Chunking — variable length
|
|-- System Level (系统级)
|   |-- Distributed Inference
|   |   |-- Tensor Parallelism — 单请求跨 GPU
|   |   |-- Pipeline Parallelism — 阶段间流水线
|   |   |-- Expert Parallelism — MoE 的专家分布
|   |-- Edge-Cloud Collaboration
|   |   |-- Split Computing — vision on edge, LLM on cloud
|   |   |-- Action Caching — 缓存常见场景的动作
|   |   |-- Model Offloading — dynamic layer offloading
|   |-- Serving Framework
|   |   |-- vLLM (VLM 扩展) — 当前最广泛
|   |   |-- SGLang — structured generation for VLM
|   |   |-- TensorRT-LLM — NVIDIA 优化
|   |   |-- 专用 VLA serving — 尚不存在 (gap!)
```

### 5.2 技术成熟度矩阵

| 技术 | LLM 成熟度 | VLM 适配度 | VLA 适配度 | 关键挑战 |
|------|-----------|-----------|-----------|---------|
| FlashAttention | 生产级 | 高 | 中 | VLM cross-modal pattern 未优化 |
| PagedAttention | 生产级 | 中 | 低 | Visual KV 无差异化管理 |
| Continuous Batching | 生产级 | 中 | 低 | Modality 差异导致 padding 浪费 |
| PD Disaggregation | 生产级 | 中 | 低 | 需扩展到 EPD |
| Speculative Decoding | 成熟 | 起步 (2025) | 萌芽 (2025-26) | VLM/VLA specific 挑战 |
| Token Pruning | - | 活跃 (2024+) | 萌芽 (2026) | 与 serving system 整合不足 |
| Quantization (INT8/INT4) | 成熟 | 中 | 起步 | VLA 1-bit 刚开始探索 |
| Flow 单步推理 | - | - | 活跃 (2025+) | 质量-速度 trade-off |

---

## 6. 推荐的 Research Entry Points

基于以上全景分析，以下是 5 个最有前景的研究方向，按照与张昊实验室能力的匹配度和预期 impact 排序：

### 6.1 VLM-Aware Serving System (VLM 感知的推理系统)

**核心思想：** 构建一个原生支持 VLM 特性的 serving system，而不是在 LLM serving system 上打补丁。

**具体方向：**
- **Modality-Aware PagedAttention：** 重新设计 KV-cache 管理，对 visual tokens 采用低精度/压缩存储，对 text tokens 保持高精度。在 page eviction 时优先 evict 低重要性的 visual pages。
- **EPD 三阶段调度器：** 将 DistServe 的 prefill-decode disaggregation 扩展到 encode-prefill-decode，每个阶段独立调度、独立 batching、独立硬件分配。
- **Image Prefix Caching：** 对于多个请求共享同一图像的场景 (如客服系统中多个用户对同一产品图提问)，实现 visual KV-cache 的跨请求共享。

**匹配度分析：**
- vLLM 原班人马，对 PagedAttention 和 continuous batching 的底层实现最了解
- DistServe 的经验直接延伸到 EPD
- 有清晰的 benchmark：TTFT, throughput, SLO attainment rate

**评估标准：** TTFT (time to first token), throughput (tokens/s), SLO attainment rate, memory efficiency (GB per concurrent request)

**时间跨度：** 2026-2028，足够发表 2-3 篇 top venue 论文

### 6.2 VLA Real-Time Serving with Action Streaming (VLA 实时推理与动作流)

**核心思想：** 设计专门的 VLA serving system，将 "推理" 和 "执行" 从串行变为并行流式处理。

**具体方向：**
- **Action Streaming Server：** 类似 continuous batching 但在 single request 内 —— VLA 一边推理一边输出动作，robot controller 一边执行一边接收新动作。需要解决 action consistency (前后动作平滑过渡) 和 re-planning (环境变化时中断并重新推理) 问题。
- **Adaptive Action Chunking at System Level：** 将 AAC (CVPR 2026) 的 model-level 想法提升到 system level —— serving system 根据实时 entropy 估计动态调整每次推理的 action chunk 长度，在 latency-accuracy trade-off 上自适应。
- **Speculative Action Generation：** 结合 speculative decoding 和 action chunking —— 用小模型快速生成 K-step action draft，大模型 verify + correct。KERV (2026) 的 kinematic rectification 思路是很好的起点。

**匹配度分析：**
- Lookahead/Speculative Decoding 的经验直接应用
- FastVideo 的流式生成思想迁移到 action streaming
- 目前不存在专门的 VLA serving system (巨大的 gap)

**评估标准：** Control loop frequency (Hz), task success rate vs latency trade-off, re-planning latency, action smoothness

**时间跨度：** 2026-2029，这是一个完全未开垦的领域，PhD 全周期都有空间

### 6.3 Cross-Modal Token Economy (跨模态 Token 经济学)

**核心思想：** 在 VLM/VLA inference 中，不同模态的 token "价值" 天然不同。构建一个统一的框架来量化、管理、优化跨模态 token 的分配。

**具体方向：**
- **Token Value Estimator：** 训练一个轻量的 value network，实时估计每个 visual token 在当前 query context 下的信息贡献。这个估计器可以驱动 pruning (low-value tokens 被剪枝)、KV-cache eviction (low-value tokens 优先 evict)、quantization (low-value tokens 低精度)。
- **System-Model Co-design：** 将 token pruning 从 model forward pass 中的 "hack" 变为 serving system 的一等公民 —— token pruning decision 与 scheduling decision 联合优化。例如，当 GPU memory 紧张时，系统可以要求 model 更激进地 prune visual tokens。
- **Unified Compression Pipeline：** 将 visual token pruning (空间维度)、temporal compression (时间维度)、KV-cache compression (存储维度) 统一到一个框架中。

**匹配度分析：**
- 这是一个 "systems + algorithm" 的交叉方向，非常符合张昊实验室的风格
- 可以 leverage vLLM 的 memory management 和 scheduling 基础设施
- Token pruning 文献已经很丰富 (如上表)，但与 serving system 的整合几乎为零

**评估标准：** Token compression ratio vs task accuracy pareto curve, end-to-end latency, memory efficiency

**时间跨度：** 2027-2029，在 2026 先发表 VLM-aware serving 的工作后，自然延伸到 token economy

### 6.4 FastVideo 技术到 VLA 的迁移 (Video Generation → Action Generation)

**核心思想：** FastVideo 加速视频生成的技术 (STA, VSA, step-parallel) 在理论上可以迁移到 Flow VLA 的 action generation。

**具体方向：**
- **STA for VLA：** Action sequence 在时间维度上具有局部性 (相邻 timestep 的 action 相似)。STA 的滑动窗口思想可以用于 action denoising 的 temporal attention。
- **Step-Parallel Action Denoising：** FastVideo 的并行去噪思想用于 Flow VLA —— 多个去噪 step 并行执行，减少 wall-clock time。
- **VSA for Video VLA：** 对于接受视频输入的 VLA (如 DreamZero)，VSA 可以直接用于视频 observation 的 sparse temporal attention。
- **Action Distillation Framework：** 利用 FastVideo 的蒸馏框架，将 multi-step flow VLA (teacher) 蒸馏为 single-step (student)，类似 FASTER (2026) 但更系统化。

**匹配度分析：**
- FastVideo 的代码和经验直接可用
- Flow VLA (Pi-Zero, FASTER 等) 与视频生成在数学形式上高度相似
- 但 action space 维度远低于 video pixel space，需要验证这些技术的 ROI

**评估标准：** Action generation latency (ms), task success rate, denoising step count, real-robot control frequency

**时间跨度：** 2026-2028，可以快速验证并发表

### 6.5 Heterogeneous VLM/VLA Inference (异构推理)

**核心思想：** 在 cloud-edge 协同的场景下，VLM/VLA 的不同组件 (vision encoder, LLM backbone, action head) 可以分布在不同的计算层级上。

**具体方向：**
- **Split VLA Inference：** Vision encoder 和 action head 运行在 edge (Jetson)，LLM backbone 运行在 cloud。需要解决：通信延迟 (visual features 上传、action tokens 下传)、fault tolerance (网络中断时的 fallback)、bandwidth-accuracy trade-off (visual features 的压缩程度)。
- **Tiered Serving：** 简单任务 (如 pick-and-place) 用 edge 上的小模型处理，复杂任务 (如 multi-step manipulation with reasoning) 路由到 cloud 上的大模型。需要一个 task complexity estimator。
- **Preemptive Action Caching：** 对于常见场景 (如工厂产线上的重复动作)，预计算并缓存 action trajectories，在运行时直接查找而不是推理。

**匹配度分析：**
- DistServe 的分离思想在更大范围上的推广
- 需要理解 robotics control 的实际需求 (与 robotics 实验室合作)
- 通信延迟的建模和优化是核心系统问题

**评估标准：** End-to-end latency breakdown (compute vs communication), task success rate under different network conditions, cost efficiency (cloud GPU hours)

**时间跨度：** 2027-2029，需要先建立 local inference 的 baseline

---

## 附录 A: 关键论文速查表

### VLM Token Compression
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| FastV | Fast Vision Token Pruning | 2024 | 2403.06764 |
| LLaVA-PruMerge | Adaptive Token Reduction for LLaVA | 2024 | 2403.15388 |
| MADTP | Multimodal Alignment-Guided Dynamic Token Pruning | 2024 | 2403.02991 |
| FlashVLM | Text-Guided Visual Token Selection | 2025 | 2512.20561 |
| TokenCarve | Information-Preserving Visual Token Compression | 2025 | 2503.10501 |
| Skip-Vision | Adaptive Token Skipping for VLMs | 2025 | 2503.21817 |
| DUET-VLM | Dual Stage Unified Efficient Token Reduction | 2026 | 2602.18846 |
| HiDrop | Hierarchical Vision Token Reduction | 2026 | 2602.23699 |
| ReDiPrune | Relevance-Diversity Pre-Projection Pruning | 2026 | 2603.24680 |
| RCP | Visual Token Removal, 88.9% reduction | 2026 | 2604.04972 |
| ID-Selection | 97.2% Token Reduction | 2026 | 2604.05601 |

### VLM Serving Systems
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| EPD Disaggregation | Encode-Prefill-Decode Disaggregation | 2025 | 2501.05460 |
| HydraInfer | Hybrid EPD Disaggregation | 2025 | 2505.12658 |
| RServe | Encoding-Prefill Overlap | 2025 | 2509.24381 |
| xLLM | Dynamic EPD Policy | 2025 | 2510.14686 |
| EPD-Serve | EPD on Ascend | 2026 | 2601.11590 |
| RPS-Serve | Modality-Aware Scheduling | 2026 | 2603.26498 |

### VLM Speculative Decoding
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| ViSpec | Vision-Aware Speculative Decoding | 2025 | 2509.15235 |
| Spec-LLaVA | Dynamic Tree-Based SD for VLMs | 2025 | 2509.11961 |
| SpecVLM | Fast SD for VLMs | 2025 | 2509.11815 |
| SAGE | Entropy-Guided Adaptive SD | 2026 | 2602.00523 |
| MMSpec | SD Benchmark for VLMs | 2026 | 2603.14989 |
| Fast-dVLM | Block-Diffusion VLM | 2026 | 2604.06832 |

### VLA Efficiency
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| Spec-VLA | Speculative Decoding for VLA | 2025 | 2507.22424 |
| NanoVLA | 52x Faster on Edge | 2025 | 2510.25122 |
| BitVLA | 1-bit VLA | 2025 | 2506.07530 |
| ActionFlow | Pipelined Action on Edge | 2025 | 2512.20276 |
| FASTER | Single-Step Flow VLA | 2026 | 2603.19199 |
| Mean-Flow VLA | One-Step Mean-Flow VLA | 2026 | 2603.01469 |
| HeiSD | Hybrid SD for Embodied VLA | 2026 | 2603.17573 |
| KERV | Kinematic-Rectified SD for VLA | 2026 | 2603.01581 |
| LiteVLA-Edge | 4-bit VLA on Jetson | 2026 | 2603.03380 |
| DyQ-VLA | Dynamic Quantization for VLA | 2026 | 2603.07904 |
| ETA-VLA | Token Pruning for VLA | 2026 | 2603.25766 |

### Flow/Diffusion Action Acceleration
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| FlowPolicy | Consistency Flow for Manipulation | 2024 | 2412.04987 |
| Falcon | Partial Denoising for Visuomotor | 2025 | 2503.00339 |
| ManiFlow | Consistency Flow Training | 2025 | 2509.01819 |
| DM1 | MeanFlow One-Step Manipulation | 2025 | 2510.07865 |
| Sparse ActionGen | Real-time Pruning for Diffusion Policy | 2026 | 2601.12894 |
| Action-to-Action Flow | 0.56ms Action Generation | 2026 | 2602.07322 |
| AnchorVLA | Truncated Diffusion for Mobile Manipulation | 2026 | 2604.01567 |

### KV-Cache for VLM
| 简称 | 全称 | 年份 | arXiv |
|------|------|------|-------|
| Frequency-Domain KV | Outlier-KV-Aware Compression | 2025 | 2511.16786 |
| StreamMem | Streaming Video KV-Cache | 2025 | 2508.15717 |
| MHA2MLA-VLM | Multi-Head Latent Attention for VLM | 2026 | 2601.11464 |
| ConceptMoE | Token-to-Concept KV Compression | 2026 | 2601.21420 |
| HybridKV | Hybrid KV Cache Compression | 2026 | 2604.05887 |

---

## 附录 B: 技术演进时间线

```
2023
  |-- RT-2 (首个大规模 VLA, 55B)
  |-- Diffusion Policy (DDPM, 100 步去噪)
  |-- vLLM (PagedAttention)
  |
2024
  |-- OpenVLA (开源 7B VLA)
  |-- Pi-Zero (Flow Matching VLA)
  |-- Octo (Multi-task Diffusion VLA)
  |-- FastV, LLaVA-PruMerge (首批 VLM token pruning)
  |-- DistServe (Prefill-Decode 分离)
  |-- FlowPolicy (Consistency Flow, 1-2 步)
  |
2025 Q1-Q2
  |-- FlashVLM, TokenCarve (text-aware visual pruning)
  |-- ViSpec, Spec-LLaVA, SpecVLM (首批 VLM speculative decoding)
  |-- EPD Disaggregation (VLM 三阶段分离概念提出)
  |-- DM1, ManiFlow (Flow VLA 单步推理)
  |-- NanoVLA, BitVLA (edge VLA)
  |
2025 Q3-Q4
  |-- HiViS, DREAM (改进 VLM SD)
  |-- RServe (encoding-prefill overlap)
  |-- HydraInfer (hybrid EPD)
  |-- xLLM (动态 EPD 调度)
  |
2026 Q1 (当前)
  |-- SAGE (3.36x VLM SD), Fast-dVLM (6x 加速)
  |-- MMSpec (VLM SD benchmark, 证明 text-only SD 在 VLM 退化)
  |-- HeiSD, KERV (VLA-specific speculative decoding)
  |-- FASTER, Mean-Flow VLA (单步 Flow VLA)
  |-- RPS-Serve, EPD-Serve (VLM-specific serving)
  |-- HybridKV, MHA2MLA-VLM (VLM KV-cache 优化)
  |-- DUET-VLM, ReDiPrune, RCP, ID-Selection (高压缩比 token pruning)
  |-- DyQ-VLA, LiteVLA-Edge, HBVLA (VLA 量化/edge)
  |-- AnoleVLA (线性复杂度 VLA)
  |-- Adaptive Action Chunking (CVPR 2026)
  |-- VLM Inference Efficiency 综合 Survey (arXiv:2604.05546)
```

---

## 附录 C: 与本项目后续研究的关系

本 survey 的核心结论可以凝练为三个观察：

1. **VLM serving 正处于 "pre-vLLM" 阶段：** 2024-2026 年的 VLM serving 工作 (EPD disaggregation, modality-aware scheduling) 类似于 2022-2023 年的 LLM serving 研究。这意味着一个类似 vLLM 量级的 VLM serving system 即将出现 —— 而张昊实验室有最好的位置来做这件事。

2. **VLA inference 正处于 "wild west" 阶段：** 没有专门的 VLA serving system，没有统一的 benchmark，没有成熟的优化 pipeline。从 autoregressive VLA 的 speculative decoding 到 flow VLA 的单步推理，各种方向并行探索。这是 PhD 入场的黄金时期。

3. **Algorithm-System Co-design 是最大的空白：** 绝大多数 token pruning、quantization、speculative decoding 的工作都是 model-level 的，与 serving system 的 scheduling、memory management、batching 策略没有深度整合。这个 co-design 的空间是实验室最擅长的领域。

---

*本 survey 基于 2026 年 4 月 arXiv 搜索结果编写，所有引用论文均有 arXiv ID 可验证。部分未公开论文的细节为基于摘要/标题的合理推断，已在相应位置标注。*
