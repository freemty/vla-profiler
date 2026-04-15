---
name: project-skill
description: "Use when advising on project architecture, experiment history, codebase navigation, or research findings."
user-invocable: false
version: v2
note: "v2 — attention overlay viz, exp01b/exp02a results, VLA controller hierarchy, timing cross-validation."
updated_at: "2026-04-15"
---

# vlla — Project Knowledge

> VLM/VLA Real-Time Systems Survey & Research
> UCSD PhD 方向调研项目 | 导师: 张昊 (Hao Zhang) — vLLM/FastVideo/Chatbot Arena 作者
> v2 — attention overlay, exp01b/exp02a, VLA controller, timing validation.

---

## 1. Project Overview & Current State

**项目名称:** vlla (VLM/VLA Real-Time Systems)
**核心问题:** 如何让 VLM/VLA 在实时约束下高效运行？
**研究定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

**动机:**
张昊的技术路线: Parameter Server -> Alpa -> vLLM -> FastVideo -> **VLM/VLA real-time systems**。每一步都是 ML Systems 前沿的下一个自然问题。VLM/VLA serving 正处于 "pre-vLLM" 阶段，存在巨大的系统研究空间。

**当前阶段:** Experiment (Phase 1 — VLM/VLA Profiling + Interpretability)
- `current_exp`: exp02a (ACT profiling — **done**)
- `stage`: experiment
- Survey 产出: 4 份核心文档，覆盖 180+ 篇论文/项目 (2024-2026)
- Framework 产出: VLM profiling + attention analysis + attention overlay 可视化 + VLA profiling 框架 (`src/`)
- 新增模块: Interpretability Mixin 体系 (`src/interpretability/`)、OverlayRenderer (`src/viz/`)、Timing Cross-Validation (`src/tasks/validation_task.py`)
- 共享核心: `model-probe-core` git submodule (`src/core/`)，同时被 rope2sink 消费
- 服务器: xdlab23 (8x RTX 5880 Ada 48GB)，3 个实验完成
- **完成的实验:** exp01a (E/P/D profiling), exp01b (attention analysis), exp02a (ACT profiling)
- **下一步:** OpenVLA profiling (配置已就绪)、attention overlay 在更多 input variant 上运行、扩展到更多 VLA 模型

**核心数据汇总:**

| Exp | Model | Key Metric |
|-----|-------|-----------|
| exp01a | Qwen2.5-VL-7B | E=253ms (58%), D=18-21ms/tok |
| exp01b | Qwen2.5-VL-7B | Pos 2 = universal sink (12-28x), Gini >0.91 |
| exp02a | ACT (LeRobot) | Total ~3ms, 850x faster than VLM |

---

## 2. Architecture

### 2.1 目录结构

```
vlla/
|-- CLAUDE.md              # 项目入口 + 索引
|-- CHANGELOG.md           # 版本日志
|-- .pipeline-state.json   # LabMate 流水线状态
|
|-- src/                   # Profiling & analysis framework
|   |-- __init__.py
|   |-- run_tasks.py       # Hydra entry point (profiling / analysis mode 路由)
|   |-- core/              # git submodule -> model-probe-core (shared with rope2sink)
|   |   |-- probe_core/
|   |       |-- controller.py   # HookMode enum, BaseController (StoreMixin + ABC)
|   |       |-- hooks.py        # HookManager: path resolution, hook registration helpers
|   |       |-- state.py        # StoreMixin: step_store/global_store lifecycle
|   |       |-- registry.py     # Registry class (dict-based dispatch)
|   |-- controllers/
|   |   |-- base_vlm_controller.py  # BaseVLMController: E/P/D phase hooks, PhaseTimer
|   |   |-- base_vla_controller.py  # BaseVLAController: E/C/A phase hooks
|   |   |-- qwen_vl_controller.py   # QwenVLController: model loading, inference, QKV hooks
|   |   |-- openvla_controller.py   # OpenVLAController: DINOv2+SigLIP→Llama-2
|   |   |-- act_controller.py       # ACTController: ResNet18→CVAE→action chunk
|   |-- interpretability/           # Attention overlay system
|   |   |-- base_mixin.py     # TokenSpatialMap, TokenType, BaseInterpretabilityMixin
|   |   |-- vlm_mixin.py      # VLMInterpretabilityMixin (Qwen2.5-VL token→image mapping)
|   |   |-- vla_mixin.py      # VLAInterpretabilityMixin (Pi-Zero placeholder)
|   |-- viz/                        # Visualization
|   |   |-- overlay_renderer.py     # OverlayRenderer: heatmap overlay, strip, GIF
|   |-- tasks/
|   |   |-- profiling_task.py       # task_epd_profiling: timing aggregation -> JSON
|   |   |-- attention_task.py       # visual_text_attn, sink_detection, per_layer_stats
|   |   |-- attention_overlay_task.py  # attention→image space→heatmap render
|   |   |-- validation_task.py      # timing cross-validation PhaseTimer vs torch.profiler
|   |-- utils/
|       |-- timing.py           # PhaseTimer: CUDA event wrapper (CPU fallback)
|
|-- configs/               # Hydra experiment configs
|   |-- base.yaml
|   |-- qwen_vl_7b/
|   |   |-- profiling.yaml         # E/P/D profiling
|   |   |-- attention.yaml         # Attention analysis
|   |   |-- attention_overlay.yaml # Overlay visualization
|   |-- act/
|   |   |-- profiling.yaml         # ACT E/A profiling
|   |-- openvla_7b/
|       |-- profiling.yaml         # OpenVLA E/P/D profiling
|       |-- attention.yaml         # OpenVLA attention analysis
|
|-- scripts/               # Server deployment scripts
|-- survey/                # 文献综述
|-- exp/summary.md         # 实验 flight recorder
|-- docs/
|   |-- superpowers/specs/
|   |   |-- 2026-04-14-vlm-profiling-framework-design.md
|   |   |-- 2026-04-15-attention-overlay-visualization-design.md
|   |-- superpowers/plans/
|   |   |-- 2026-04-14-vlm-profiling-framework.md
|   |   |-- 2026-04-15-attention-overlay-visualization.md
|   |-- knowhow/
|       |-- runbooks/deploy-to-xdlab23.md
|       |-- toolchain/cuda-profiling-patterns.md
|       |-- toolchain/hydra-config-patterns.md
|       |-- debug-solutions/phasetimer-cpu-backend-bug.md
|       |-- debug-solutions/act-action-queue-hooks.md
|       |-- debug-solutions/gqa-attention-analysis.md
|       |-- debug-solutions/qwen25vl-model-structure.md
|       |-- debug-solutions/qwen25vl-vision-token-mapping.md
|       |-- infrastructure/xdlab23-model-weights.md
```

### 2.2 Framework 继承链

```
probe_core.BaseController          # Model-agnostic: hook lifecycle, StoreMixin, HookMode
  |
  +-> BaseVLMController            # VLM-specific: E/P/D phases, PhaseTimer
  |     +-> QwenVLController       # Qwen2.5-VL: model loading, QKV hooks, VLMInterpretabilityMixin
  |     +-> OpenVLAController      # OpenVLA (AR VLA): DINOv2+SigLIP→Llama-2 7B
  |     +-> (future) InternVLController
  |
  +-> BaseVLAController            # VLA-specific: E/C/A phases (encode/context/action)
        +-> ACTController          # ACT (LeRobot): ResNet18→CVAE→action chunk, single-forward
        +-> (future) PiZeroController  # Pi-Zero: VLM backbone + flow action head
```

**VLM vs VLA Phase Models:**
- **VLM (BaseVLMController):** E/P/D — Encode / Prefill / Decode (autoregressive)
- **VLA (BaseVLAController):** E/C/A — Encode / Context / Action (C optional, A may iterate)
  - `has_context_phase()`: True for VLA-with-VLM-backbone (Pi-Zero), False for pure VA (ACT)
  - `get_denoise_steps()`: 1 for single-forward (ACT), N for flow/diffusion models

**Interpretability Mixin Architecture:**
```
BaseInterpretabilityMixin  # ABC: get_token_spatial_mappings, classify_token_types, map_attention_to_image
  +-> VLMInterpretabilityMixin   # Qwen2.5-VL: image_grid_thw → patch grid, <|image_pad|> token scan
  +-> VLAInterpretabilityMixin   # Pi-Zero placeholder (NotImplementedError)
```

**Key Design Decisions:**
- Profiling 和 Analysis 模式严格分离 (tensor copy 干扰 timing)
- Prefill vs Decode 通过 `seq_len > 1` 判断
- PhaseTimer 累加同名 phase 的多次 mark (支持 decode N steps / denoise N steps)
- 配置驱动: 新增 input variant 只需改 YAML
- Interpretability Mixin 通过多重继承注入 Controller，不修改 Controller 核心逻辑
- Timing Cross-Validation: PhaseTimer (CUDA Events) 与 torch.profiler 双测对比，verdict: PASS/WARN/FAIL

### 2.3 Server Deployment

```
Local (Mac) --git bundle+scp--> xdlab23 (8x RTX 5880 Ada)
            <--rsync results---  /data1/ybyang/vlla
```

SSH: `ssh xdlab23_yang` | Conda: `vit-probe` | HF: `/data1/ybyang/huggingface`

---

## 3. System Cognition

### 3.1 三大战略判断 (Survey 核心结论)

1. **VLM serving 处于 "pre-vLLM" 阶段** — EPD disaggregation 类似 2022 年 PD 分离
2. **VLA inference 处于 "wild west" 阶段** — 无统一 serving system/benchmark
3. **Algorithm-System Co-design 是最大空白** — token pruning/quantization 未与 scheduling 整合

### 3.2 已验证的假设

**From Literature:**
- Visual token 97.2% 可剪枝 (ID-Selection); Text-only SD 在 VLM 上退化 (MMSpec)
- Flow VLA 可压缩至单步 (0.56ms); WAM 可 zero-shot (DreamZero 7Hz)

**From exp01a (first-party):**
- **Encode 是多图场景主导瓶颈:** single_image E=253ms 占 ~58%
- **Encode 随 image 数量线性扩展:** 2 images = 2.14x single image
- **Decode per-token 跨模态稳定:** ~18-21ms (visual tokens 对 decode 速度影响有限)
- **Prefill 与 visual token 数量成正比:** 0img→20ms, 1img→156ms, 2img→332ms

**From exp01b (first-party):**
- **Pos 2 (first visual patch) 是 universal attention sink:** 12K-18K received attention across all layers, 12-28x more than #2 position
- **Text→Visual attention extreme sparsity:** Gini >0.91, 强力支持 visual token pruning
- **Layer 21 entropy 最低 (3.44 vs 4.0-4.2):** 中后层 attention 更集中

**From exp02a (first-party):**
- **ACT (single-forward VLA) total ~3ms:** 850x faster than VLM (Qwen2.5-VL ~2.5s)
- **VLA 延迟下界 baseline:** Encode (ResNet18) ~2.5-2.8ms (80%), Action ~0.4-0.8ms
- **Resolution impact minimal on encode:** 240p/480p/720p 差异不大

### 3.3 活跃假设

- [x] ~~Vision encoding 占 >50% prefill 延迟~~ → **exp01a 已验证 (58%)**
- [x] ~~Attention sink 存在于 VLM~~ → **exp01b 已验证 (Pos 2 universal sink)**
- [x] ~~Text→Visual attention sparse, supports pruning~~ → **exp01b Gini >0.91**
- [ ] EPD 三阶段分离实际收益 (disaggregation ROI)
- [ ] VLM speculative decoding 中 visual token 对 acceptance rate 的影响
- [ ] Per-layer attention entropy 分布的实际意义 (Layer 21 最低 → 是否是最佳 pruning 切入点?)
- [ ] OpenVLA (AR VLA) profiling: E/P/D 与 Qwen2.5-VL 有多大差异?
- [ ] Pi-Zero flow VLA profiling: denoise step count vs latency trade-off

---

## 4. Technical Archive

### 4.1 四大范式对比

| 维度 | VA (1-step flow) | VLA (7B AR) | WAM (DreamZero) | Latent WM |
|------|-----------------|-------------|-----------------|-----------|
| 延迟 | 1-5ms | 100-500ms | ~130ms | 10-15ms |
| 控制频率 | >200Hz | 2-10Hz | ~7Hz | 60-100Hz |
| 泛化能力 | 低 | 高 | 极高 | 中 |

**新增 first-party 基线 (exp02a):**
- ACT (single-forward VA): ~3ms total → 落入 VA 1-5ms 范围，符合预期

### 4.2 Pareto 前沿

| 方法 | 延迟 | 意义 |
|------|------|------|
| Action-to-Action Flow | 0.56ms | VA 速度下界 |
| **ACT (our measurement)** | **~3ms** | **VA baseline (first-party)** |
| Mean-Flow VLA / FASTER | ~50ms | VLA 单步化 |
| DreamZero | ~130ms, 7Hz | WAM zero-shot |
| SAGE | 3.36x speedup | VLM SD 标杆 |
| ID-Selection | 97.2% token reduction | Token pruning 上界 |

### 4.3 Rejected Alternatives & Rationale

| Decision | Alternative Considered | Why Rejected |
|----------|----------------------|-------------|
| Mixin-based interpretability | Subclass-per-model | Mixin allows mix-and-match without deep hierarchy |
| PhaseTimer CUDA Events | torch.profiler only | CUDA Events give sub-ms precision; torch.profiler added as cross-validation |
| Git bundle sync | rsync only | Bundle preserves git history on server side |
| ACT model.forward() | policy.select_action() | select_action() has action queue cache, skips forward on subsequent calls |
| GQA repeat_interleave | Separate Q/K head handling | repeat_interleave is HF standard, clean single codepath |

---

## 5. Experiment History

| Exp ID | Date | Status | Prediction | Actual | Key Finding |
|--------|------|--------|-----------|--------|-------------|
| exp01a | 2026-04-15 | **done** | E >50% of total | E=58% | E scales linearly with images; D per-token stable ~18-21ms |
| exp01b | 2026-04-15 | **done** | Sink exists (literature) | Pos 2 sink 12-28x | Universal sink at first visual patch; Gini >0.91 supports pruning |
| exp02a | 2026-04-15 | **done** | VA ~1-5ms (literature) | ~3ms total | ACT 850x faster than VLM; ResNet18 encode 80% of total |

### 5.1 Prediction Calibration: exp01a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Vision encoding 占比 | >50% | 58% | **Accurate** |
| Decode per-token | 2-5ms (A100 literature) | ~18-21ms (RTX 5880) | **Off** — 硬件差异 ~4x |
| Encode 线性扩展 | 线性 (ViT) | 2.14x for 2 images | **Confirmed** |

### 5.2 Prediction Calibration: exp01b

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Attention sink 存在 | 存在 (文献) | Pos 2 universal, 12-28x | **Confirmed** — sink 位置在 first visual patch |
| Text→Visual sparsity | Sparse (token pruning 文献) | Gini >0.91 | **Confirmed** — 极端稀疏 |

### 5.3 Prediction Calibration: exp02a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| VA latency range | 1-5ms (文献) | ~3ms | **Accurate** — 落入预测范围 |
| Encode 占比 | 主导 (CNN bottleneck) | 80% | **Confirmed** |
| Resolution sensitivity | 应该有影响 | Minimal impact | **Off** — ResNet18 对 resolution 不敏感 |

**Meta-Learning — Prediction Calibration Trends:**
- **结构性判断准确:** 哪个阶段是瓶颈、存在 attention sink、sparsity 可利用 — 3/3 correct
- **硬件映射偏差:** 文献数据 (A100/H100) 到 RTX 5880 有 3-5x gap，需要 hardware correction factor
- **量级估计偏差:** ResNet18 对 resolution 不敏感是 miss — 小模型 vs 大 ViT 的假设不能直接迁移
- **Gini 系数超预期:** 预测 "sparse" 但没预测到 >0.91 的极端稀疏度

---

## 6. Engineering Lessons (APPEND-ONLY)

### 2026-04-11: Survey 过程
1. 先全景后深入的调研策略有效
2. 180+ 篇论文覆盖
3. 单文件 <1000 行控制 context window

### 2026-04-15: Framework 开发 + exp01a
4. **从已有项目提取共享核心值得做:** rope2sink → model-probe-core submodule，关键改动是 timestep → phase 抽象泛化
5. **Profiling 和 Analysis 必须严格分离:** tensor `.to("cpu")` 触发 CUDA sync，干扰 timing
6. **Hydra 配置驱动的好处:** 新增 input variant 只需 YAML 条目
7. **Git bundle workaround 可靠:** xdlab23 firewall 挡 GitHub，`sync_to_remote.sh` 一键同步
8. **`seq_len > 1` 判断 prefill vs decode 稳定:** 跨 input type 都工作
9. **PhaseTimer decode 必须累加:** 原设计 dict 覆盖只保留最后一个 step → 修复为 list 累加
10. **Per-input profiling 隔离:** 原设计混跑所有 inputs → 修复为每个 input 独立 warmup + benchmark
11. **Immutable pattern 在 hot-path 有代价:** `[*list, item]` 是 O(n) copy，probe_core 的 hook 保留 `.append()` 并注释说明

### 2026-04-15: Attention Overlay + VLA Profiling (v0.4.0 + v0.4.1)
12. **GQA (Grouped Query Attention) 必须显式处理:** Qwen2.5-VL Q=28 heads, K=4 heads → `repeat_interleave` 对齐，否则 reshape crash。影响所有 GQA 模型 (Llama-3, Mistral 等)
13. **head_dim divisibility check 不能省:** 非 128 head_dim 模型会静默产出错误结果
14. **ACT select_action() 有 action queue 缓存陷阱:** 只有 1/chunk_size 的调用实际触发 forward pass → profiling 必须直接调 model.forward()
15. **Controller 双重注册要避免:** `__init__.py` import 和 `run_tasks.py` 重复 import 会导致 registry collision
16. **Multi-image heatmap key collision:** 多图输入的 layer→image mapping 需要唯一 key (加 image_key suffix)
17. **OmegaConf ListConfig 不能直接 JSON dump:** 需要先 `OmegaConf.to_container()` 转 plain list
18. **vision token ID 不能硬编码:** `<|image_pad|>` token ID 应从 tokenizer 动态获取 (非固定 151655)
19. **Timing cross-validation 有价值:** PhaseTimer vs torch.profiler 双测验证，发现 sum(E+P+D) vs wall clock 有 gap → 量化了 projection/sampling 等漏测时间
20. **_aggregated_timing 必须显式声明:** 动态属性 → 显式声明为 `Optional[Dict]` in `__init__`，防止 AttributeError
21. **detach before numpy:** `attn_probs.detach().numpy()` — 忘记 detach 会在 backward-enabled tensor 上 crash

---

## 7. Active Prompt Versions & Trade-offs

(No prompt versioning files yet — `prompts/` directory does not exist. When added, track here.)

---

## 8. Quick Reference

### Commands

| 命令 | 用途 |
|------|------|
| `bash scripts/sync_to_remote.sh` | Sync to xdlab23 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling` | Profiling GPU 0 |
| `bash scripts/launch_exp.sh 1 qwen_vl_7b/attention` | Attention GPU 1 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/attention_overlay` | Overlay viz GPU 0 |
| `bash scripts/launch_exp.sh 0 act/profiling` | ACT profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 openvla_7b/profiling` | OpenVLA profiling GPU 0 |
| `bash scripts/download-results.sh` | Download results |

### Server

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` |
| GPUs | 8x RTX 5880 Ada 48GB |

### Registries

**Controllers:** `qwen_vl` → QwenVLController, `openvla` → OpenVLAController, `act` → ACTController
**Tasks:** `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`, `attention_overlay`, `timing_validation`

### Knowhow Index

| File | Topic |
|------|-------|
| `docs/knowhow/runbooks/deploy-to-xdlab23.md` | xdlab23 部署流程 |
| `docs/knowhow/toolchain/cuda-profiling-patterns.md` | CUDA Event vs torch.profiler 对比 |
| `docs/knowhow/toolchain/hydra-config-patterns.md` | Hydra ListConfig/device gotchas |
| `docs/knowhow/debug-solutions/gqa-attention-analysis.md` | GQA Q/K head mismatch |
| `docs/knowhow/debug-solutions/act-action-queue-hooks.md` | ACT action queue 缓存 |
| `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` | CPU backend no-op bug |
| `docs/knowhow/debug-solutions/qwen25vl-model-structure.md` | Qwen2.5-VL 模型结构 |
| `docs/knowhow/debug-solutions/qwen25vl-vision-token-mapping.md` | Vision token 定位 |
| `docs/knowhow/infrastructure/xdlab23-model-weights.md` | ModelScope 404 issue |

---

*v2 — attention overlay viz, exp01b/exp02a results, VLA controller hierarchy, timing cross-validation. Updated: 2026-04-15*
