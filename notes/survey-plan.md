# VLM/VLA Real-Time Systems Survey 行动计划

> 2026-04-11 | Yuanbo Yang | UCSD PhD (Advisor: Hao Zhang)

## Phase 0: 快速建立知识基础 (Week 1-2)

### 必读综述 (3 篇)
1. **Efficient Multimodal LLM Inference Survey** (arXiv:2604.05546) — 最全面的 VLM inference efficiency 综述
2. **A Survey on Efficient VLA Models** (arXiv:2510.24795) — VLA 效率提升方法全面综述
3. **Efficient VLA Models: A Systematic Survey** (arXiv:2510.17111) — 从模型/感知/动作/训练四维度分类

### 必读 foundational papers (张昊实验室出品)
- vLLM (PagedAttention) → DistServe (PD disaggregation) → FastVideo (STA/VSA)
- Lookahead Decoding → Dynasor (reasoning scheduling)

## Phase 1: VLM Serving 深度调研 (Week 2-4)

### 主线: 从 PD 到 EPD 的演进
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| EPD Disaggregation | E-P-D 三阶段分离 | 2501.05460 |
| HydraInfer | Hybrid EPD + stage batching | 2505.12658 |
| Nova | 单 GPU 跨阶段并行 | 2509.21301 |
| RServe | Encoding-Prefill overlap | 2509.24381 |
| xLLM | 动态 EPD policy | 2510.14686 |
| EPD-Serve | Ascend 上的 EPD | 2601.11590 |
| RPS-Serve | Modality-aware scheduling | 2603.26498 |
| HeteroServe | 异构 GPU 模态划分 | 2603.12707 |

### 副线: Visual Token 管理
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| FlashVLM | Text-guided token selection | 2512.20561 |
| HybridKV | Visual/text KV-cache 差异化压缩 | 2604.05887 |
| MHA2MLA-VLM | Multi-Head Latent Attention | 2601.11464 |
| ID-Selection | 97.2% token reduction | 2604.05601 |

### 副线: VLM Speculative Decoding
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| MMSpec | Benchmark: text-only SD 在 VLM 退化 | 2603.14989 |
| SAGE | Entropy-guided adaptive SD, 3.36x | 2602.00523 |
| Fast-dVLM | Block-diffusion + SGLang, 6x | 2604.06832 |

### 实操: 跑起来
- [ ] 在 vLLM 上 serve 一个 VLM (Qwen2.5-VL-7B)，profile latency breakdown
- [ ] 在 SGLang 上做同样的对比
- [ ] 用 vLLM 的 profiler 分析 vision encoding / prefill / decode 各阶段时间占比

## Phase 2: VLA Inference 深度调研 (Week 4-6)

### 主线: Autoregressive VLA 加速
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| OpenVLA 2.0 | 改进版开源 VLA baseline | — |
| A1 | Adaptive VLM for VLA, 72% 延迟降低 | 2604.05672 |
| HeiSD | Hybrid speculative decoding for VLA | 2603.17573 |
| KERV | Kinematic-rectified speculation | 2603.01581 |
| DySL-VLA | Dynamic-Static layer-skipping, 3.75x | 2602.22896 |

### 主线: Flow/Diffusion VLA 单步化
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| Pi-Zero | Flow matching VLA (baseline) | — |
| FASTER | Horizon-aware single-step | 2603.19199 |
| Mean-Flow VLA | One-step mean-flow, 83.9x | 2603.01469 |
| Action-to-Action Flow | Informed init, 0.56ms | 2602.07322 |
| FlowPolicy | Consistency flow matching | 2412.04987 |

### 副线: Edge 部署
| 论文 | 核心技术 | arXiv |
|------|---------|-------|
| NanoVLA | 52x faster on edge | 2510.25122 |
| BitVLA | 1-bit VLA | 2506.07530 |
| LiteVLA-Edge | 4-bit GGUF on Jetson | 2603.03380 |
| ActionFlow | Edge pipeline, 2.55x FPS | 2512.20276 |

### 实操: 跑起来
- [ ] 部署 OpenVLA，profile inference latency
- [ ] 尝试 VLAgents serving framework
- [ ] 在 simulated env (SIMPLER) 上测试 VLA latency-success trade-off

## Phase 3: Gap Analysis & Research Direction (Week 6-8)

### 核心问题 (需要通过 Phase 1-2 的实操来回答)
1. VLM serving 中，vision encoding 占总延迟的百分比？EPD 分离的实际收益？
2. VLM speculative decoding 中，visual token 对 acceptance rate 的影响有多大？
3. VLA 的 action token 生成延迟 breakdown 是怎样的？bottleneck 在哪？
4. Flow VLA 单步化后，remaining bottleneck 是 vision encoding 还是 flow head？
5. 现有 token pruning 方法集成到 serving system 的工程难度？

### 写出 Research Proposal (与张昊讨论)
基于 survey 提炼 2-3 个具体的 first-year project 方向：

**Tier 1 (最推荐 — quick win + high impact):**
- **VLM-Aware Serving System** — 做"VLM 的 vLLM"，EPD 调度 + modality-aware paging
- **VLA Real-Time Serving** — 蓝海领域，action streaming + speculative action

**Tier 2 (中期方向 — 深度 co-design):**
- **Cross-Modal Token Economy** — token value 驱动 system-level 决策
- **FastVideo → VLA 技术迁移** — STA/VSA 用于 action denoising

**Tier 3 (长期方向):**
- **异构 VLM/VLA 推理** — Cloud-Edge split inference

## 阅读优先级总表

### Must Read (Phase 0, 前 2 周)
- [ ] VLM Inference Survey (2604.05546)
- [ ] VLA Efficiency Survey (2510.24795)
- [ ] vLLM paper → DistServe → FastVideo (实验室背景)

### Should Read (Phase 1-2, 第 2-6 周)
- [ ] EPD 系列: 2501.05460 → 2505.12658 → 2509.21301
- [ ] VLM SD: MMSpec (2603.14989) → SAGE (2602.00523) → Fast-dVLM (2604.06832)
- [ ] Flow VLA 单步化: 2602.07322 → 2603.01469 → 2603.19199
- [ ] VLA SD: HeiSD (2603.17573) → KERV (2603.01581)
- [ ] Visual KV: HybridKV (2604.05887) → MHA2MLA (2601.11464)

### Nice to Have (按需)
- Edge 部署系列
- Token pruning 细分工作
- Industry reports (GR00T N1, Pi-Zero)
