# Multimodal Serving Systems (2025-2026) —— vLLM-Omni / SGLang Diffusion 实地调研

> **调研目的**: 澄清 "serving framework 级别的 EPDA / any-to-any disaggregation" 是否已被现有系统覆盖，判断 exp08 的真正空白区。
> **调研日期**: 2026-04-27
> **方法**: 直接读 GitHub 仓库代码 + 官方 blog + arXiv paper，不仅依赖摘要
> **核心结论**: **vLLM-Omni 已经实现了 any-to-any 多阶段 disaggregation，SGLang Diffusion 已经实现了 diffusion 内部的 E/Denoiser/Decoder 分离**。"写一个新 framework 做 EPDA disaggregation" 这个方向**已无空白**。exp08 必须收缩到**机制性研究 + robotics/VLA 场景扩展**，而不是做另一个 framework。

---

## 1. vLLM-Omni (vllm-project/vllm-omni)

### 1.1 定位

- **仓库**: https://github.com/vllm-project/vllm-omni
- **Paper**: Yin et al. "vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models", arXiv:2602.02204, 2026-02-02
- **Stars**: 4.5k+（2026-04-27）
- **发布历史**:
  - 2025/11: vLLM 社区正式发布，支持 omni-modality
  - 2026/01: v0.12.0rc1（diffusion 成熟 + OpenAI-compat serving）
  - 2026/02: v0.14.0 stable release（diffusion/image-video + audio/TTS stack）
  - 2026/02: v0.16.0（rebase 到 vLLM v0.16，扩展 Qwen3-Omni / Bagel / MiMo-Audio / GLM-Image / DiT）
  - 2026/03: v0.18.0（scheduler/runtime cleanup，统一 quantization + diffusion 执行）

### 1.2 Paper 核心 claim（arXiv:2602.02204 摘要关键句）

- *"Any-to-any multimodal models... typically combining multiple autoregressive LLMs, diffusion transformers, and other specialized components"*
- *"Existing serving systems are mainly tailored to a single paradigm... They lack support for any-to-any pipelines that involve multiple interconnected model components"*
- *"novel **stage abstraction** that enables users to decompose complex any-to-any architectures into interconnected **stages represented as a graph**"*
- *"**disaggregated stage execution backend** that optimizes resource utilization and throughput across stages"*
- *"Each stage is independently served by an LLM or diffusion engine with **per-stage request batching, flexible GPU allocation, and unified inter-stage connectors** for data routing"*
- *"reduces **job completion time (JCT) by up to 91.4%** compared to baseline methods"*

### 1.3 代码层实际模块（`vllm_omni/`）

```
vllm_omni/
├── core/
│   └── sched/
│       ├── omni_ar_scheduler.py              # AR 模型 scheduler
│       ├── omni_generation_scheduler.py      # Diffusion/generation scheduler
│       ├── omni_scheduling_coordinator.py    # 跨 stage 协调，管理 WAITING_FOR_CHUNK / WAITING_FOR_INPUT 状态
│       └── omni_scheduler_mixin.py           # AR/Gen 共享行为
├── distributed/
│   ├── omni_connectors/                      # inter-stage 通信
│   │   ├── connectors/                       # KV/tensor 传输实现
│   │   ├── transfer_adapter/                 # chunk vs full_payload 传输模式
│   │   ├── kv_transfer_manager.py
│   │   └── factory.py
│   ├── omni_coordinator/                     # 全局 load balancer
│   │   ├── omni_coordinator.py
│   │   ├── load_balancer.py
│   │   ├── omni_coord_client_for_hub.py
│   │   └── omni_coord_client_for_stage.py
│   └── kv_transfer/                          # KV cache 跨实例传输
├── diffusion/                                # Diffusion engine (DiT, etc.)
├── engine/                                   # OmniEngine (wraps AR + Gen)
└── executor/                                 # 多 GPU 执行
```

### 1.4 架构图命名（`docs/source/architecture/`，直接暴露设计意图）

- `ar-dit-main-architecture.png` —— **AR + DiT 合体架构**（这就是 exp08 想要做的事）
- `async-chunk-architecture.png` —— 异步 chunk 传输
- `dit-main-architecture.png` —— 纯 DiT 流
- `vllm-omni-dataflow-between-stages.png` —— **多阶段 dataflow**
- `vllm-omni-diffusion-flow.png` —— diffusion 内部流
- `qwen3-omni-async-chunk.png` / `qwen3-omni-non-async-chunk.png` —— 实际 any-to-any 模型场景

### 1.5 支持的模型（`examples/offline_inference/`）

Omni / Any-to-any:
- Qwen2.5-Omni, Qwen3-Omni, Qwen3-TTS
- Bagel (ByteDance)
- Ming-Flash-Omni / MimoAudio / CosyVoice3 / Fish-Speech
- Mammoth Modal2, Magi Human
- DynIN Omni

Image/Video generation:
- Hunyuan-Image3, Helios
- Image-to-Image, Image-to-Video, Text-to-Image, Text-to-Video

**显著缺失**: **没有 robotics / VLA / action-chunk** 相关 example。所有 output 都是 text/image/audio/video，没有 low-latency closed-loop control。

### 1.6 对 exp08 的冲击

| exp08 spec 原计划 contribution | vLLM-Omni 的现状 |
|-------------------------------|------------------|
| 提出 "EPDA stage abstraction" | ❌ 已做（`omni_ar_scheduler` + `omni_generation_scheduler` + `scheduling_coordinator`） |
| Inter-stage KV/latent transfer | ❌ 已做（`omni_connectors/transfer_adapter`） |
| 全局 placement algorithm | ❌ 已做（`omni_coordinator/load_balancer`） |
| Per-stage GPU allocation | ❌ 已做 |
| JCT / goodput as new metric | ❌ 已做（JCT 91.4% reduction） |

---

## 2. SGLang Diffusion (sgl-project/sglang/python/sglang/multimodal_gen/)

### 2.1 定位

- **Blog**: https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/ (2025-11-07)
- **路径**: `python/sglang/multimodal_gen/` —— 集成在 SGLang 主仓库
- **作者**: SGLang Team + FastVideo Team（**Hao Zhang 在作者列表**，最后一位）
- **关键合作**: **与 FastVideo 深度联合**——SGLang 负责 serving，FastVideo 负责 training/distillation
- **性能**: 1.2-5.9× 相对 HF Diffusers

### 2.2 定位 quote（blog 原文）

- *"unified, high-performance engine for both **language and diffusion** tasks"*
- *"the future of generation lies in **combining architectures**... Bagel, Transfusion, Fast-dLLM v2"*
- *"an enhanced fork of **FastVideo**... This partnership allows SGLang Diffusion to focus on delivering cutting-edge inference speed, while FastVideo provides comprehensive support for training-related tasks like model distillation"*

### 2.3 代码层实际模块

```
multimodal_gen/
├── README.md
├── apps/
│   ├── ComfyUI_SGLDiffusion
│   └── webui
├── runtime/
│   ├── cache/
│   ├── disaggregation/                       # ← 已有完整 disaggregation
│   │   ├── roles.py                          # RoleType = {ENCODER, DENOISER, DECODER, SERVER, MONOLITHIC}
│   │   ├── orchestrator.py                   # DiffusionServer (N:M:K disaggregation)
│   │   ├── dispatch_policy.py                # RoundRobin / MaxFreeSlotsFirst 等
│   │   ├── scheduler_mixin.py
│   │   ├── request_state.py
│   │   └── transport/                        # codec, buffer, engine, manager, allocator
│   ├── pipelines/                            # 每个模型一个 pipeline
│   │   ├── flux.py / flux_2.py / flux_2_klein.py / flux_2_nvfp4.py
│   │   ├── qwen_image.py
│   │   ├── sana.py / stable_diffusion_3.py
│   │   ├── hunyuan_pipeline.py / hunyuan3d_pipeline.py
│   │   ├── wan_causal_dmd_pipeline.py
│   │   └── ltx_2_pipeline.py / helios_pipeline.py / mova_pipeline.py
│   ├── managers/, models/, layers/, loader/
│   └── distributed/                          # USP (Ulysses + Ring), CFG-parallel, TP
```

### 2.4 `roles.py` 关键发现（直接读代码）

```python
class RoleType(str, Enum):
    MONOLITHIC = "monolithic"
    ENCODER = "encoder"     # text_encoder / tokenizer / image_encoder / processor / connectors
    DENOISER = "denoiser"   # transformer (the DiT denoising core)
    DECODER = "decoder"     # vae / audio_vae / video_vae / vocoder
    SERVER = "server"       # Head node (no GPU, routes)
```

这正是 exp08 原 spec 提出的 **E/P/D/A 作为独立 role** 的概念，**只是 SGLang 是单个 diffusion 内部的 encoder/denoiser/decoder 三段，不含 LLM backbone**。

### 2.5 Scope 对比（exp08 原计划 vs SGLang Diffusion）

| 方面 | SGLang Diffusion 做了 | SGLang Diffusion **没做** |
|------|----------------------|---------------------------|
| Role-based disaggregation | ENCODER / DENOISER / DECODER | LLM prefill / decode 作为独立 role |
| Inter-role transport | codec/buffer/engine 全套 | LLM KV cache 到 diffusion latent 的跨模态传输 |
| 并行策略 | USP / CFG-parallel / TP | 跨 AR+Diffusion 的混合并行 |
| 优化目标 | single diffusion request 吞吐 | robotics closed-loop SLO |
| 模型覆盖 | T2I / T2V / I2I | VLA / action chunk generation |

### 2.6 SGLang vs vLLM-Omni 分工（基于代码 + blog）

| 系统 | 覆盖范围 | 核心价值 |
|------|---------|---------|
| vLLM-Omni | **AR LLM + Diffusion 跨 model 整合**（any-to-any） | "一个 engine 既 serve 聊天也 serve 生图/生视频" |
| SGLang Diffusion | **单个 diffusion 内部 E/Denoiser/D 分离** | "加速 T2I / T2V 本身" |

两者**互补而非重叠**。vLLM-Omni 走 "AR-Diffusion 大融合"，SGLang Diffusion 走 "Diffusion 深优化"。

---

## 3. 其他相关系统（简要）

### 3.1 EPD Disaggregation (arXiv:2501.05460, ICML 2025)

- **Scope**: VLM 的 Encode / Prefill / Decode 三阶段分离
- **贡献**: multimedia token caching (15x memory reduction), intra-request encode parallelism, role-switching
- **与 exp08 的 gap**: 没有 A 阶段（diffusion action）
- **实际状态**: 论文存在，框架可能已被 vLLM-Omni 部分吸收

### 3.2 HydraInfer (arXiv:2505.12658, 2025)

- **Scope**: 混合 EPD 调度，stage-level batching
- **性能**: 4× 吞吐于 vLLM，8xH800 验证
- **与 exp08 的 gap**: 同样无 A 阶段

### 3.3 FastVideo (Hao Lab)

- **Scope**: 单个 diffusion request 内部加速（STA/VSA attention、蒸馏）
- **与 exp08 正交**: FastVideo 不考虑 serving 多 stage co-location
- **与 SGLang Diffusion 关系**: SGLang Diffusion 是 FastVideo 的 "enhanced fork"

### 3.4 VLAgents (2026-01)

- **Scope**: 模块化 VLA policy serving，集成 7 种策略（OpenVLA, Pi-Zero 等）
- **地位**: VLA 领域的 "vLLM"
- **关键**: 这是 **robotics / VLA 专门 serving framework**，与 vLLM-Omni 是**并行路线**

---

## 4. exp08 真正剩下的空白区

基于以上调研，exp08 原 spec 里的下列点**已被覆盖**，应剔除：

- ❌ "EPDA stage abstraction" —— vLLM-Omni + SGLang 都有
- ❌ "inter-stage transport" —— 两边都有
- ❌ "placement algorithm" —— vLLM-Omni 有 coordinator/load_balancer
- ❌ "写一个新 framework" —— 无空白

剩余的**真正空白**（交叉对比 vLLM-Omni paper / SGLang 代码 / exp08a pilot 数据）：

### 空白 A：Pair-wise co-location interference 的 quantitative 模型

- vLLM-Omni paper 展示 disaggregation 后 JCT 降低 91.4%，但**不量化** co-located 时各阶段互相干扰的具体数值
- 实践中 vLLM-Omni 的 coordinator 决定"哪些 stage 放一起"大概率是基于工程经验（free slot, round-robin），不是基于 contention prediction
- **exp08a pilot 已经显示**（2026-04-27）：LLM prefill 被 174M DiT 共置时膨胀 3.15×，roofline 低估 28×
- **空白**：没人建立 GPU kernel-launch-level contention 的 predictive model

### 空白 B：Robotics / VLA closed-loop 场景

- vLLM-Omni examples: TTS, T2I, T2V, any-to-any chat —— 全是 one-shot 或 streaming text
- **没有** action chunk / control frequency / real-time robot SLO
- VLA 的 **A 阶段是 closed-loop**：每个 action 必须在 ~50-100ms 内出，不是 JCT 约束
- **空白**：vLLM-Omni / SGLang 都没有机器人专用 SLO 语义

### 空白 C：Roofline 预测失败的机制性解释

- 传统 roofline（FLOPs/Byte + peak TFLOPS/BW）假设 compute 和 memory 是两个可独立饱和的资源
- exp08a pilot 证明这不成立：P（compute-bound）和 A（dispatch-bound）co-locate 时**都严重变慢**，说明还有第三个共享资源（kernel launch queue? SM scheduler? HBM controller queue?）
- **空白**：建立"GPU kernel-level contention" 的解析模型，修补 roofline

---

## 5. 对 exp08 方向的诚实重新评估

### 5.1 原定的 ⭐⭐⭐⭐⭐ 候选 D（EPDA disaggregation）→ 降至 ⭐⭐

原因：framework 部分已被 vLLM-Omni 完整占据，"disaggregation paper" 类型的 contribution 空间消失。

### 5.2 可做的新方向（按可行性排序）

| 方向 | 贡献类型 | 与 Hao 实验室关系 | 可行性 |
|------|---------|-----------------|--------|
| **B1**: 在 vLLM-Omni 上加 VLA / robotics example + SLO benchmark | 生态贡献 | 互补、非竞争 | 高 |
| **B2**: GPU kernel-level contention model + exp08a 数据延伸 | 机制研究 | 补充 vLLM-Omni paper | 中-高 |
| **B3**: FastVideo-style 加速 VLA action DiT（STA/VSA 迁移） | 算法优化 | 直接延伸 FastVideo | 中 |
| **B4**: Visual KV sparse compression (原候选 B) | co-design | 独立于 vLLM-Omni | 中 |

**推荐方向组合**: **B1 + B2**（生态贡献 + 机制研究双腿走）。B1 让你成为 vLLM-Omni 的使用者/贡献者，B2 让你有独立 research claim。

### 5.3 见 Hao 的新话术骨架

```
"我读了 vLLM-Omni paper 和最新 SGLang Diffusion 代码。
 EPDA disaggregation framework 已被你们/vLLM 社区做完。

 但我 pilot 出一个现象：LLM prefill 被 174M DiT 共置时膨胀 3.15×，
 roofline 预测低估 28×。说明 vLLM-Omni coordinator 决定'哪些 stage
 放一起'的决策目前可能缺 quantitative 基础。

 两个可能方向：
  1. 给 vLLM-Omni 补 kernel-level contention profiling/prediction
  2. 扩展到 robotics/VLA 场景（vLLM-Omni 没覆盖）

 想请教哪个更匹配实验室 roadmap？"
```

---

## 6. 对现有项目文档的修改需求

下列文档引用了"EPDA disaggregation framework" 作为 exp08 contribution，需要降档：

- `docs/specs/2026-04-26-epda-disaggregation-spec.md` —— 删除 "写 position paper / framework" 段落，改为 "benchmark + mechanism study"
- `docs/specs/2026-04-26-epda-roofline-analysis.md` —— 添加 "vLLM-Omni / SGLang 已实现 framework，本文只做 profile" disclaimer
- `survey/papers/hao-style-synthesis.md` —— 候选 D 打分 ⭐⭐⭐⭐⭐ → ⭐⭐
- `slides/epda-roofline-motivation.html` —— 加 "Related: vLLM-Omni / SGLang Diffusion" section
- `exp/exp08a/FINDINGS.md` —— 重新定位发现的价值（机制性 vs framework）

---

## 附录：数据来源

| 文件 | 内容 | 时间 |
|------|------|------|
| `lmsys.org/blog/2025-11-07-sglang-diffusion/` | SGLang Diffusion 官方介绍 + FastVideo 合作 | 2025-11-07 |
| `github.com/vllm-project/vllm-omni/` | vLLM-Omni 代码（今日仍活跃） | 访问 2026-04-27 |
| `arxiv.org/abs/2602.02204` | vLLM-Omni paper | 2026-02-02 提交 |
| `github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/` | SGLang Diffusion 代码（roles.py 等） | 访问 2026-04-27 |
| `exp/exp08a/FINDINGS.md` | exp08a pilot 干扰数据 | 2026-04-27 |
