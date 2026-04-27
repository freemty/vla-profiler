---
name: project-skill
description: "Use when advising on project architecture, experiment history, codebase navigation, or research findings."
user-invocable: false
version: v6
note: "v6 — exp07a Pi-Zero dual-stream flow VLA profiling done. PiZeroController (uv venv, allenzren backend). DiT scaling curve: 174M=7.2ms < 300M=18ms < 350M=32ms."
updated_at: "2026-04-25"
---

# vlla — Project Knowledge

> VLM/VLA Real-Time Systems Survey & Research
> UCSD PhD 方向调研项目 | 导师: 张昊 (Hao Zhang) — vLLM/FastVideo/Chatbot Arena 作者
> v6 — exp07a Pi-Zero dual-stream flow VLA profiling done. 300M Expert per-step 18ms fills DiT scaling curve gap.

---

## 1. Project Overview & Current State

**项目名称:** vlla (VLM/VLA Real-Time Systems)
**核心问题:** 如何让 VLM/VLA 在实时约束下高效运行？
**研究定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

**动机:**
张昊的技术路线: Parameter Server -> Alpa -> vLLM -> FastVideo -> **VLM/VLA real-time systems**。每一步都是 ML Systems 前沿的下一个自然问题。VLM/VLA serving 正处于 "pre-vLLM" 阶段，存在巨大的系统研究空间。

**当前阶段:** Experiment (Phase 1 — VLM/VLA Profiling + Interpretability)
- `current_exp`: exp07a (Pi-Zero dual-stream flow VLA profiling — **done**)
- `stage`: experiment
- `version`: v0.6.1
- Survey 产出: 4 份核心文档，覆盖 180+ 篇论文/项目 (2024-2026)
- Framework 产出: VLM profiling + attention analysis + attention overlay 可视化 + VLA profiling 框架 (`src/`)
- 新增模块: Interpretability Mixin 体系 (`src/interpretability/`)、OverlayRenderer (`src/viz/`)、Timing Cross-Validation (`src/tasks/validation_task.py`)
- 共享核心: `model-probe-core` git submodule (`src/core/`)，同时被 rope2sink 消费
- Presentation viewer: `viewer/` — Flask + 3 HTML pages (hub, presentation, experiments) for advisor meeting
- 服务器: xdlab23 (8x RTX 5880 Ada 48GB)，10 个实验完成
- **完成的实验:** exp01a (E/P/D profiling), exp01b (attention analysis), exp02a (ACT profiling), exp03a (LingBot-VLA-4B profiling), exp04a (Fast-WAM ActionDiT profiling), exp04b (LingBot-VA full WAM profiling), exp05a (LingBot-VLA attention analysis), exp05b (Qwen2.5-VL-3B attention analysis), exp06a (NitroGen 500M DiT profiling), exp07a (Pi-Zero dual-stream flow VLA profiling)
- **下一步:** DreamZero profiling on RTX 5880 Ada、DreamZero DiT layer activation variance 分析、OpenVLA profiling (需 HF 下载)

**核心数据汇总:**

| Exp | Model | Key Metric |
|-----|-------|-----------|
| exp01a | Qwen2.5-VL-7B | E=253ms (58%), D=18-21ms/tok |
| exp01b | Qwen2.5-VL-7B | Pos 2 = universal sink (12-28x), Gini >0.91 |
| exp02a | ACT (LeRobot) | Total ~3ms, 850x faster than VLM |
| exp03a | LingBot-VLA-4B | E=35.7ms/C=38.3ms/A=0.48ms, total 74.5ms ≈ 13Hz |
| exp04a | Fast-WAM (ActionDiT, 6.7B) | @10step: E=7.6ms/C=36.7ms/A=362ms, total 407ms, 2.5Hz |
| exp04b | LingBot-VA (full WAM, 5B DiT) | **canonical (rerun 2026-04-27)**: E=84.7/V=697/A=1708ms, total 2518ms, 0.40Hz (median). Legacy warmup=3 低估 18%. |
| exp05a | LingBot-VLA-4B (attention) | VLA fine-tuning reshapes attention: sink migrates Pos2→Pos64, Gini 0.91→0.07, entropy V-shape→flat |
| exp05b | Qwen2.5-VL-3B (attention) | Disambiguation: Gini collapse is VLA fine-tuning effect, not model size. 3B vanilla Gini 0.80-0.98 |
| exp06a | NitroGen 500M DiT | Per-step 7.2ms, linear scaling. 174M DiT compute-bound. 174M→350M = BW transition |
| exp07a | Pi-Zero (dual-stream flow VLA) | **stable** E=9.32/C=26.40/A=164.76ms, total 200.5ms ≈ 5Hz. Action Expert (300M) dominates 82%. Per-step ~16.5ms. Cross-attn makes 300M Expert ~2.3x pricier than pure DiT. Bimodal 污染 → warmup=15 + `nvidia-smi -pm 1` |

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
|   |   |-- base_vla_controller.py  # BaseVLAController: E/C/A phase hooks, _register_capture_hook
|   |   |-- qwen_vl_controller.py   # QwenVLController: model loading, inference, QKV hooks
|   |   |-- openvla_controller.py   # OpenVLAController: DINOv2+SigLIP→Llama-2
|   |   |-- act_controller.py       # ACTController: ResNet18→CVAE→action chunk
|   |   |-- lingbot_vla_controller.py  # LingBotVLAController: Qwen2.5-VL-3B + flow action head
|   |   |-- lingbot_va_controller.py   # LingBotVAController: full WAM, E/V/A phases (VAE+5B DiT video+action)
|   |   |-- nitrogen_controller.py     # NitroGenController: SigLIP→VL-SA→DiT, E/C/A manual timing, random weights
|   |   |-- pizero_controller.py    # PiZeroController: Pi-Zero (lazy import, separate env)
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
|       |-- __init__.py         # Shared utility functions
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
|   |   |-- profiling.yaml         # OpenVLA E/P/D profiling
|   |   |-- attention.yaml         # OpenVLA attention analysis
|   |-- lingbot_vla_4b/
|   |   |-- profiling.yaml         # LingBot-VLA E/C/A profiling
|   |-- nitrogen/
|   |   |-- profiling.yaml         # NitroGen E/C/A profiling + k sweep
|   |-- pizero/
|       |-- profiling.yaml         # Pi-Zero profiling (requires openpi env)
|
|-- scripts/               # Server deployment & convenience scripts
|   |-- sync_to_remote.sh         # Git bundle sync to xdlab23
|   |-- launch_exp.sh             # Launch experiment on server GPU
|   |-- download-results.sh       # Rsync results from server
|   |-- run_local.sh              # Local GPU experiment
|   |-- run_remote.sh             # SSH → xdlab23 launch (quoted args)
|   |-- run_viewer.sh             # Flask viewer startup
|   |-- run_tests.sh              # pytest test suite
|   |-- setup_lingbot_vla.sh      # LingBot-VLA uv env + model download
|   |-- profile_lingbot_va.py     # LingBot-VA standalone profiling script (E/V/A)
|   |-- profile_fastwam.py        # Fast-WAM standalone profiling script (E/C/A)
|-- viewer/                # Research presentation viewer (Flask)
|   |-- app.py                    # Flask app + static file catch-all (api/ guard)
|   |-- static/
|       |-- index.html            # Navigation hub
|       |-- presentation.html     # 5-section advisor meeting slides
|       |-- experiments.html      # Expandable experiment detail tables
|-- survey/                # 文献综述
|-- exp/summary.md         # 实验 flight recorder
|-- docs/
|   |-- superpowers/specs/
|   |   |-- 2026-04-14-vlm-profiling-framework-design.md
|   |   |-- 2026-04-15-attention-overlay-visualization-design.md
|   |-- superpowers/plans/
|   |   |-- 2026-04-14-vlm-profiling-framework.md
|   |   |-- 2026-04-15-attention-overlay-visualization.md
|   |-- papers/
|   |   |-- landscape.md
|   |   |-- starvla-framework-deep-dive.md
|   |-- knowhow/
|       |-- runbooks/deploy-to-xdlab23.md
|       |-- runbooks/setup-uv-env-xdlab23.md
|       |-- toolchain/cuda-profiling-patterns.md
|       |-- toolchain/hydra-config-patterns.md
|       |-- debug-solutions/phasetimer-cpu-backend-bug.md
|       |-- debug-solutions/act-action-queue-hooks.md
|       |-- debug-solutions/gqa-attention-analysis.md
|       |-- debug-solutions/qwen25vl-model-structure.md
|       |-- debug-solutions/qwen25vl-vision-token-mapping.md
|       |-- debug-solutions/lingbotvla-integration.md
|       |-- infrastructure/xdlab23-model-weights.md
```

### 2.2 Framework 继承链

```
probe_core.BaseController          # Model-agnostic: hook lifecycle, StoreMixin, HookMode
  |
  +-> BaseVLMController            # VLM-specific: E/P/D phases, PhaseTimer, _register_capture_hook
  |     +-> QwenVLController       # Qwen2.5-VL: model loading, QKV hooks, VLMInterpretabilityMixin
  |     +-> OpenVLAController      # OpenVLA (AR VLA): DINOv2+SigLIP→Llama-2 7B
  |     +-> (future) InternVLController
  |
  +-> BaseVLAController            # VLA-specific: E/C/A phases, _register_capture_hook (独立定义)
        +-> ACTController          # ACT (LeRobot): ResNet18→CVAE→action chunk, single-forward
        +-> LingBotVLAController   # LingBot-VLA-4B: Qwen2.5-VL-3B + 10-step flow action head
        +-> LingBotVAController    # LingBot-VA (full WAM): VAE + 5B WanDiT, E/V/A phases
        +-> NitroGenController      # NitroGen 500M: SigLIP→VL-SA→DiT flow matching, E/C/A manual timing
        +-> PiZeroController       # Pi-Zero: VLM backbone + flow action head (lazy import, separate env)
```

**VLM vs VLA Phase Models:**
- **VLM (BaseVLMController):** E/P/D — Encode / Prefill / Decode (autoregressive)
- **VLA (BaseVLAController):** E/C/A — Encode / Context / Action (C optional, A may iterate)
  - `has_context_phase()`: True for VLA-with-VLM-backbone (LingBot-VLA, Pi-Zero), False for pure VA (ACT)
  - `get_denoise_steps()`: 1 for single-forward (ACT), N for flow/diffusion models (LingBot-VLA: 10)
- **WAM (LingBotVAController overrides):** E/V/A — Encode / Video-denoise / Action-denoise
  - Video denoising is an explicit phase (V), separate from action denoising (A)
  - Both V and A use the same shared 5B DiT; phases differ only in token routing (action_mode flag)

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
- `_register_capture_hook` 在 BaseVLMController 和 BaseVLAController 中各自独立定义 (两条继承链不共享此方法)
- PiZeroController 使用 lazy import + try/except (requires openpi conda env，避免依赖 crash)
- LingBot-VLA 使用 uv venv 而非 conda (xdlab23 非交互 SSH 下 conda 不可用)
- WAM profiling (exp04a/04b) 使用 standalone scripts 而非 Hydra controller，因为这些模型有自包含推理 pipeline

### 2.3 Server Deployment

```
Local (Mac) --git bundle+scp--> xdlab23 (8x RTX 5880 Ada)
            <--rsync results---  /data1/ybyang/vlla
```

SSH: `ssh xdlab23_yang` | Conda: `vit-probe` (legacy) | uv venv: `.venvs/lingbot-vla/` | HF: `/data1/ybyang/huggingface`

**Environment strategy:**
- `vit-probe` (conda) — shared with rope2sink, for Qwen2.5-VL / OpenVLA / ACT / LingBot-VA
- `.venvs/lingbot-vla/` (uv) — LingBot-VLA specific (lerobot compat, PyTorch 2.8)
- `.venvs/pizero/` (uv) — Pi-Zero (allenzren/open-pi-zero, vendored `pizero_src/`)
- `fastwam` (conda) — Fast-WAM (Python 3.10, PyTorch 2.7.1+cu128)

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

**From exp03a (first-party):**
- **3B backbone 比 7B 快 ~7x:** Encode 35.7ms vs 253ms (同 ViT 架构，参数量差异)
- **Context ≈ Encode:** KV cache fill (38.3ms) 与 vision encoding (35.7ms) 几乎等价
- **Multi-view encode 几乎不增加 (+1.5%):** patchified input 在 ViT 之前聚合，ViT 处理合并后的 patch 序列
- **Flow action head 极快 (0.48ms):** 10-step denoise loop 的 action_out_proj 累计仅 0.48ms，与 ACT (0.5ms) comparable
- **Total 74.5ms ≈ 13Hz:** 接近机器人 10Hz 实时需求，仍有 ~25% gap

**From exp04a (first-party) — Fast-WAM ActionDiT profiling:**
- **Action phase dominates at 89-94%:** @10step A=362ms of 407ms total — A-dominated
- **Per-step cost ~32ms/step:** 每个 Euler step 运行完整 30-layer ActionDiT 通过 MoT cross-attention
- **Paper's 190ms 是 A100/H100 + 5 steps:** RTX 5880 + 10 steps = 407ms
- **Encode 极轻 (7ms):** 单帧 VAE encode
- **Architecture: 6.7B params:** "skip-imagination" 仍加载完整 5B video expert + 350M ActionDiT

**From exp04b (first-party) — LingBot-VA full WAM profiling:**
- **Full WAM ~6x slower than skip-imagination WAM:** 2518ms vs 407ms
- **Action still dominates at 68%:** 与 Fast-WAM 一致
- **Per-step cost V≈A (~29ms/step):** 共享同一个 5B DiT
- **VAE encode 75.5ms vs Fast-WAM 7.6ms (10x):** streaming VAE with z_dim=48 vs lightweight path
- **0.5 Hz makes real-time impossible:** 26x slower than VLA (13Hz), 660x slower than ACT

**From exp05a (first-party) — LingBot-VLA attention analysis:**
- **VLA fine-tuning 彻底重塑 attention:** Sink 从 Pos 2 (vanilla VLM) 迁移到 Pos 64 (boundary token)
- **Gini 从 >0.91 崩塌至 0.07-0.45:** Text→Visual attention near-uniform (VLM pruning 不可迁移)
- **Entropy 从 V-profile 变为 flat (4.79-4.90):** 训练目标改变了 attention 结构
- **结论:** Attention structure 是 training objective property 而非 architecture property

**From exp05b (first-party) — Qwen2.5-VL-3B attention analysis (消歧实验):**
- **Gini 崩塌归因于 VLA fine-tuning, 非 model size:** 3B vanilla Gini 0.80-0.98 (与 7B 一致)
- **Pos 9 attention sink (2054), entropy V-shape (L7=2.69):** 与 7B vanilla pattern 一致
- **seq_len 差异:** vanilla 1273 (424 visual) vs VLA 136 (64 visual, spatial_merge=4)

**From exp06a (first-party) — NitroGen 500M DiT profiling:**
- **Per-step DiT cost = 7.2ms, perfectly linear across k=1..16:** 完美线性
- **174M DiT params (not 100M as estimated):** VL-SA 28M 独立于 DiT
- **Action dominates 91.6% at k=16:** 与 Fast-WAM/LingBot-VA pattern 一致
- **SigLIP encode = 8.7ms, VL self-attention = 1.9ms:** 两者都很轻
- **k=1 achieves 55.9Hz:** 通过 step distillation 可达实时
- **174M→350M: 2x params, 4.4x latency → super-linear:** compute-bound 到 memory-BW-bound 转换点在 174M-350M 之间

### 3.3 WAM Profiling 综合洞察 (exp04a + exp04b)

**完整 latency spectrum (RTX 5880 Ada, bf16):**

| Model | Paradigm | Total | E | V | A | Hz | Bottleneck |
|-------|----------|-------|---|---|---|-----|-----------|
| ACT | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 | E (80%) |
| NitroGen @k16 | VA (flow DiT) | 126ms | 8.7ms | — | 115.4ms | 7.9 | A (91.6%) |
| LingBot-VLA | Flow VLA | 74.5ms | 35.7ms | — | 38.8ms | 13 | E≈C (50/50) |
| Pi-Zero @10step | Flow VLA (dual-stream) | 200.5ms (stable) | 9.32ms | — | 164.76ms | ~5 | A (82%) |
| Fast-WAM @10step | WAM (skip) | 407ms | 7.6ms | — | 362ms | 2.5 | A (89%) |
| LingBot-VA (full, rerun) | WAM (full) | 2518ms | 84.7ms | 697ms | 1708ms | 0.40 | A (69%) |

**关键规律:**
1. **Action head scaling 决定 WAM 瓶颈:** 小 action head (0.48ms) vs 大 MoT action (362ms) — 架构决定
2. **Video generation 代价:** Full WAM vs skip 的 5x 差距几乎全来自 video denoise (592ms)
3. **Per-step cost 在同一 DiT 内高度一致:** V/A ~29ms/step — step count 是线性的
4. **"Skip-imagination" 是工程妥协:** 仍加载完整 5B video expert，省 compute 不省 memory

### 3.4 活跃假设

- [x] ~~Vision encoding 占 >50% prefill 延迟~~ → **exp01a 已验证 (58%)**
- [x] ~~Attention sink 存在于 VLM~~ → **exp01b 已验证 (Pos 2 universal sink)**
- [x] ~~Text→Visual attention sparse, supports pruning~~ → **exp01b Gini >0.91**
- [x] ~~Flow VLA (real model) 延迟在 50-100ms 范围~~ → **exp03a 验证 74.5ms**
- [x] ~~Flow action head overhead 可忽略~~ → **exp03a 验证 0.48ms (0.6% of total)**
- [x] ~~WAM latency dominated by action diffusion~~ → **exp04a/04b 验证 (68-94%)**
- [x] ~~Full WAM significantly slower than skip-imagination~~ → **exp04b 验证 5x slower**
- [x] ~~LingBot-VLA attention analysis: VLA fine-tuning 是否改变 attention pattern?~~ → **exp05a 验证: 彻底重塑。Gini 0.91→0.07, sink migrates, entropy flattens**
- [x] ~~Gini 崩塌是 model size 还是 fine-tuning 效应?~~ → **exp05b 消歧: fine-tuning effect。3B vanilla Gini 0.80-0.98**
- [x] ~~NitroGen DiT per-step cost at ~100M params?~~ → **exp06a: 174M DiT, 7.2ms/step, compute-bound**
- [x] ~~Compute-bound vs memory-BW-bound DiT size transition?~~ → **174M-350M 之间 (2x params, 4.4x latency)**
- [ ] EPD 三阶段分离实际收益 (disaggregation ROI)
- [ ] VLM speculative decoding 中 visual token 对 acceptance rate 的影响
- [ ] Per-layer attention entropy 分布的实际意义 (Layer 21 最低 → 是否是最佳 pruning 切入点?)
- [ ] OpenVLA (AR VLA) profiling: E/P/D 与 Qwen2.5-VL 有多大差异?
- [x] ~~Pi-Zero flow VLA profiling: denoise step count vs latency trade-off~~ → **exp07a stable: E=9.32/C=26.40/A=164.76ms, total 200.5ms ≈ 5Hz. Per-step ~16.5ms, cross-attn to PaliGemma KV makes 300M Expert 2.3x pricier than pure DiT. Bimodal (runs 1-12 polluted by GPU power warmup)**
- [ ] 3B→7B backbone scaling law: 是否线性? (exp01a vs exp03a 初步暗示 ~7x for 2.3x params)
- [ ] Imagination value quantification: Full WAM 的 592ms video generation 带来多少 task success rate 提升?
- [ ] DreamZero profiling: DiT caching 在 memory-BW-bound regime 的真实收益?
- [ ] DreamZero DiT layer activation variance: 哪些层可以 cache?

---

## 4. Technical Archive

### 4.1 四大范式对比

| 维度 | VA (1-step flow) | VLA (7B AR) | WAM (DreamZero) | Latent WM |
|------|-----------------|-------------|-----------------|-----------|
| 延迟 | 1-5ms | 100-500ms | ~130ms | 10-15ms |
| 控制频率 | >200Hz | 2-10Hz | ~7Hz | 60-100Hz |
| 泛化能力 | 低 | 高 | 极高 | 中 |

**First-party baselines (from experiments):**
- ACT (single-forward VA): ~3ms total → VA 1-5ms 范围 ✓
- LingBot-VLA-4B (flow VLA, 3B backbone): 74.5ms total ≈ 13Hz → VLA 高端
- Fast-WAM @10step (skip-imagination, 6.7B): 407ms → 2.5Hz
- LingBot-VA (full WAM, 5B DiT): 2518ms → 0.40Hz (rerun 2026-04-27, canonical)

### 4.2 WAM Benchmark Baselines (RTX 5880 Ada, bf16, random-init)

| Config | E | V (video) | A (action) | Total | Hz |
|--------|---|-----------|------------|-------|-----|
| Fast-WAM @5step | 7.1ms | — | 164ms | 205ms | 4.9 |
| Fast-WAM @10step | 7.6ms | — | 362ms | 407ms | 2.5 |
| Fast-WAM @20step | 7.0ms | — | 638ms | 677ms | 1.5 |
| LingBot-VA @20V+50A | 84.7ms | 697ms | 1708ms | 2518ms | 0.40 |

**Per-step cost reference:**
- Fast-WAM ActionDiT (30L, 1024 hidden, MoT cross-attn): ~32ms/step
- LingBot-VA WanDiT video (30L, 3072 hidden): ~34.5ms/step (rerun)
- LingBot-VA WanDiT action (30L, 3072 hidden): ~34.0ms/step (rerun)
- NitroGen DiT (12L, 1024 hidden, cross-attn to 256 tokens): ~7.2ms/step — compute-bound
- Pi-Zero Action Expert (18L, 1024 hidden, Gemma + cross-attn to PaliGemma KV): ~18ms/step — cross-attn overhead
- LingBot-VLA flow action head (small head, 10 steps): ~0.048ms/step

### 4.3 Pareto 前沿

| 方法 | 延迟 | 意义 |
|------|------|------|
| Action-to-Action Flow | 0.56ms | VA 速度下界 |
| **ACT (our measurement)** | **~3ms** | **VA baseline (first-party)** |
| **NitroGen @k1 (our measurement)** | **17.9ms** | **VA flow DiT baseline, 55.9Hz (first-party)** |
| **NitroGen @k16 (our measurement)** | **126ms** | **VA flow DiT, 7.9Hz (first-party)** |
| Mean-Flow VLA / FASTER | ~50ms | VLA 单步化 |
| **LingBot-VLA-4B (our measurement)** | **74.5ms** | **Flow VLA baseline (first-party, 3B)** |
| DreamZero | ~130ms, 7Hz | WAM zero-shot (A100) |
| **Pi-Zero @10step (our measurement)** | **200.5ms** (stable) | **Dual-stream flow VLA (first-party, 2.7B)** |
| **Fast-WAM @10step (our measurement)** | **407ms** | **Skip-imagination WAM (first-party)** |
| **LingBot-VA (our measurement)** | **2518ms** (rerun) | **Full WAM baseline (first-party)** |
| SAGE | 3.36x speedup | VLM SD 标杆 |
| ID-Selection | 97.2% token reduction | Token pruning 上界 |

### 4.4 Rejected Alternatives & Rationale

| Decision | Alternative Considered | Why Rejected |
|----------|----------------------|-------------|
| Mixin-based interpretability | Subclass-per-model | Mixin allows mix-and-match without deep hierarchy |
| PhaseTimer CUDA Events | torch.profiler only | CUDA Events give sub-ms precision; torch.profiler added as cross-validation |
| Git bundle sync | rsync only | Bundle preserves git history on server side |
| ACT model.forward() | policy.select_action() | select_action() has action queue cache, skips forward on subsequent calls |
| GQA repeat_interleave | Separate Q/K head handling | repeat_interleave is HF standard, clean single codepath |
| uv venv (LingBot-VLA) | conda (shared vit-probe) | conda not available in non-interactive SSH; uv is PATH-independent |
| Separate _register_capture_hook per base | Shared mixin | Two inheritance chains (VLM/VLA) need independent hook definitions; mixin adds complexity |
| eager attention (LingBot-VLA) | flash-attn / flex_attention | flex_attention dtype bug in PyTorch 2.8; flash-attn "not implemented" in lingbotvla |
| Standalone scripts (WAM profiling) | Hydra controller integration | Fast-WAM and LingBot-VA have self-contained pipelines; controller adds complexity without benefit for one-off profiling |
| Random init for WAM timing | Real checkpoint weights | Timing depends on compute graph not weight values; saves 12GB+ download |

---

## 5. Experiment History

| Exp ID | Date | Status | Prediction | Actual | Key Finding | Calibration |
|--------|------|--------|-----------|--------|-------------|-------------|
| exp01a | 2026-04-15 | **done** | E >50% of total | E=58% | E scales linearly with images; D per-token stable ~18-21ms | Accurate (structure), Off (hardware: 3-5x gap to A100 literature) |
| exp01b | 2026-04-15 | **done** | Sink exists (literature) | Pos 2 sink 12-28x | Universal sink at first visual patch; Gini >0.91 supports pruning | Confirmed (existence), Surprised (Gini >0.91 extreme) |
| exp02a | 2026-04-15 | **done** | VA ~1-5ms (literature) | ~3ms total | ACT 850x faster than VLM; ResNet18 encode 80% of total | Accurate (range), Off (resolution insensitivity) |
| exp03a | 2026-04-20 | **done** | 250-350ms (7B→3B ~2x scaling) | 74.5ms total | 3B backbone 7x faster than 7B; Context≈Encode; flow action 0.48ms; ≈13Hz | Way Off — actual 3.4-4.7x better than predicted |
| exp04a | 2026-04-21 | **done** | E=15-25ms/C=100-130ms/A=30-50ms | @10step: E=7.6ms/C=36.7ms/A=362ms, total 407ms | Action dominates 89%; per-step 32ms; paper's 190ms is A100+5step | Way Off on A (predicted 30-50ms, got 362ms) |
| exp04b | 2026-04-21 (rerun 2026-04-27) | **done** | E=10-15ms/V=600-900ms/A=500-750ms | **canonical** E=84.7/V=697/A=1708ms, total 2518ms | Full WAM ~6x skip-imagination; Action 69%; V≈A per-step (~34ms); 0.40Hz | V nailed, E 6-8x under, A 2.3x under |
| exp05a | 2026-04-21 | **done** | VLA attention similar to VLM | Gini 0.91→0.07, sink migrates, entropy flat | VLA fine-tuning reshapes attention; VLM pruning not transferable | Way Off — opposite of prediction |
| exp05b | 2026-04-21 | **done** | 3B model = different pattern | 3B vanilla Gini 0.80-0.98, consistent with 7B | Gini collapse from VLA fine-tuning, not model size | Confirmed — disambiguation successful |
| exp06a | 2026-04-22 | **done** | ~100M DiT, 5-10ms/step | 174M DiT, 7.2ms/step, linear | Per-step 7.2ms, compute-bound, 174M→350M BW transition | Accurate on per-step, Off on param count (174M not 100M) |
| exp07a | 2026-04-25 | **done** | E=15-25ms/C=40-80ms/A=30-80ms | **stable** E=9.32/C=26.40/A=164.76ms, total 200.5ms | Action Expert (300M) dominates 82%. Per-step ~16.5ms. Cross-attn 2.3x pricier than pure DiT. Bimodal (runs 1-12 polluted). | E/C overestimated, A severely underestimated (2x) |

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

### 5.4 Prediction Calibration: exp03a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Total latency | 250-350ms (7B→3B ~2x scaling) | 74.5ms | **Way Off** — 预估基于 naive 2x scaling，实际 7x |
| E/C relative cost | Encode dominant | E≈C (35.7 vs 38.3ms) | **Off** — 未预料到 context phase 如此重 |
| Action overhead | <5ms (flow is fast) | 0.48ms | **Accurate** — flow action head 确实极快 |
| Multi-view scaling | ~2x for 2 cameras | +1.5% only | **Way Off** — patchified aggregation 绕过重复 ViT |

### 5.5 Prediction Calibration: exp04a (Fast-WAM)

| Metric | Survey Prediction | Actual (@10step) | Calibration |
|--------|------------------|-----------------|-------------|
| E (VAE encode) | ~15-25ms | 7.6ms | **Off** — VAE much lighter than expected |
| C (Video prefill) | ~100-130ms | 36.7ms | **Off** — single-frame input |
| A (flow matching, 10 steps) | ~30-50ms | 362ms | **Way Off** — missed 30-layer MoT cross-attn per step |
| Total | ~150-200ms | 407ms | **Off** — 2x over due to A underestimation |

### 5.6 Prediction Calibration: exp04b (LingBot-VA)

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| E (VAE encode) | ~10-15ms | 75.5ms | **Way Off** — streaming VAE z_dim=48 much heavier |
| V (Video denoise, 20 steps) | ~600-900ms | 592.5ms | **Accurate** — nailed the lower bound |
| A (Action denoise) | ~500-750ms | 1423ms | **Off** — missed 50 steps (2.5x more than video's 20) |
| Total | ~1100-1700ms | 2091ms | **Off** — 23% over upper bound |

---

## 6. Prediction Calibration — Meta-Learning

**系统性偏差:**

1. **结构性判断准确:** 哪个阶段是瓶颈、存在 attention sink、sparsity 可利用 — 6/6 experiments correct
2. **硬件映射偏差:** 文献数据 (A100/H100) 到 RTX 5880 有 3-5x gap，需要 hardware correction factor
3. **量级估计偏差 (小模型方向):** ResNet18 对 resolution 不敏感; patchified input 绕过重复 ViT
4. **Gini 系数超预期:** 预测 "sparse" 但没预测到 >0.91 的极端稀疏度
5. **Backbone scaling 非线性:** 预测 7B→3B 是 ~2x speedup，实际 7x
6. **Context phase 被低估:** LLM forward (KV cache fill) 与 ViT encode 几乎等价
7. **WAM action phase 被严重低估:** 没有意识到每个 denoise step 都运行完整 MoT (30 layers × cross-attn)。正确推算: `steps × layers × per-layer-cost`
8. **Step count 遗漏:** LingBot-VA action 50 steps (not 20 like video)。**教训: 必须先 grep config defaults 再做预测**
9. **VAE 类型差异巨大:** Fast-WAM 7.6ms vs LingBot-VA 75.5ms — 10x 差异。不能假设 "VAE always cheap"

**校准精度趋势:**
- exp01a: 2/3 accurate — 结构 OK，硬件数字偏差
- exp01b: 2/2 confirmed — 文献预测准
- exp02a: 2/3 accurate — resolution 不敏感是 miss
- exp03a: 1/4 accurate — 最差，backbone scaling 理解不足
- exp04a: 0/4 accurate (direction correct, magnitude way off) — MoT cross-attn 代价未预料
- exp04b: 1/4 accurate (V phase nailed) — step count + streaming VAE 低估

**改进方向:**
- 对 backbone scaling 使用 FLOPs ratio 而非 param ratio
- 对非标准 input pipeline 不做 naive scaling 假设
- 将 context/prefill phase 作为独立 cost center 预测
- WAM 预测必须先确认: (1) per-step architecture (layers × heads × attn type), (2) exact step counts from config
- 预测前 grep `num_inference_steps` / `action_steps` 等 config 字段

---

## 7. Engineering Lessons (APPEND-ONLY)

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

### 2026-04-20: LingBot-VLA-4B + Codex Review (v0.4.2 + v0.4.3)
22. **两条继承链各自需要 hook 方法:** `_register_capture_hook` 在 BaseVLMController 有定义，但 BaseVLAController 继承自 BaseController 而非 BaseVLMController — LingBotVLAController 调用 `self._register_capture_hook()` 会 AttributeError。修复: 在 BaseVLAController 中也定义该方法
23. **Empty weights glob 必须 guard:** safetensors glob 返回空列表时应抛 FileNotFoundError，否则静默加载空权重导致难以调试的推理错误
24. **Shell 脚本变量必须 quote:** `run_remote.sh` 中 `$CONFIG` 未加引号 → 可被 command injection。修复: `"$CONFIG"` 双引号包裹
25. **PyTorch `.forward()` 不触发 hooks:** lingbotvla 源码调用 `self.model.forward(...)` 而非 `self.model(...)`，PyTorch hooks 只在 `__call__` 时触发。profiling hooks 静默失效
26. **uv 替代 conda 解决非交互 SSH 问题:** xdlab23 的 conda 依赖 `.bashrc` 中的 `conda init`，非交互 SSH 不 source `.bashrc`。uv 安装在 `~/.local/bin/` 无需 shell init
27. **PI0Config 字段过滤:** lingbotvla config.json 含自定义字段，PI0Config dataclass 不接受额外 kwargs。用 `dataclasses.fields()` 过滤有效字段 + `setattr` 附加
28. **Patchified input 绕过 multi-view scaling:** lingbotvla 在 ViT 之前 patchify 并聚合多视角输入 → ViT 处理合并后的 patch 序列 → multi-view encode 几乎不增加 (+1.5%)。不能假设 multi-image = multi-ViT-forward
29. **Flask catch-all 路由需要 api/ 前缀防护:** 无条件 catch-all 会遮蔽未注册的 API 路由，导致 404 静默返回 HTML 而非 JSON error

### 2026-04-21: WAM Profiling — Fast-WAM + LingBot-VA (exp04a + exp04b)
30. **先 grep config defaults 再做 latency 预测:** LingBot-VA action 用 50 steps，video 用 20 steps — step count 差异直接导致 2x 预测偏差。**规则: 所有 diffusion model 预测前必须确认 num_inference_steps**
31. **"Skip-imagination" 不意味着 "small model":** Fast-WAM 仍加载完整 5B video expert。跳过的是 test-time video denoising，不是 weight loading。6.7B params — 省 compute 不省 memory
32. **MoT cross-attention per step 代价极高:** Fast-WAM 每个 Euler step 运行 30-layer ActionDiT + MoT cross-attend to full video KV cache — ~32ms/step。正确 cost model: `steps × layers × per-layer-attn-cost`
33. **同一 DiT 的 V/A per-step cost 高度一致:** LingBot-VA V (~29.6ms) ≈ A (~28.5ms) — 在宽模型 (5120 hidden) 下 sequence length 对 per-step latency 影响可忽略
34. **Standalone profiling scripts 比 Hydra controller 更适合 external repos:** 对于有自包含推理 pipeline 的模型，直接在 repo 内写 profiling script 比强行集成进 Hydra 体系更快
35. **Random init 对 timing 结果有效:** timing 取决于 compute graph (operators, shapes, dtypes)，不取决于 weight values。节省 12GB+ checkpoint 下载
36. **High variance 可能来自 GPU power management:** 短 workload 受 GPU power state 波动影响。更长的 sustained workload 或固定 power mode 可减少方差
37. **WAM Action phase 占比 68-94% 是架构常数:** 在 Fast-WAM 和 LingBot-VA 上一致观察到。这是 diffusion action head 的内在代价

### 2026-04-21: VLA Attention Analysis (exp05a + exp05b)
38. **VLA fine-tuning 彻底重塑 attention pattern:** 不能假设 VLM 的 attention 特性 (sink, sparsity) 在 VLA 中保持。Gini 从 0.91 崩塌至 0.07
39. **消歧实验必不可少:** exp05a 发现 Gini 崩塌，但无法区分是 model size 还是 fine-tuning 效应。exp05b 用 3B vanilla VLM 消歧 → 确认是 fine-tuning
40. **Attention structure 是 training objective property:** 不是 architecture property。相同架构、不同训练目标 → 完全不同的 attention pattern

### 2026-04-22: NitroGen 500M DiT Profiling (exp06a)
41. **controller_config 中 `mode` 字段被 framework 截取:** BaseController 用 `mode` 判断 hook_mode → 自定义用途需用不同 key name (如 `weight_mode`)
42. **离线环境 `from_pretrained()` 需替换为 config-based 构建:** SigLIP random weights 不需要下载 HF 权重
43. **`__new__` 绕过 `__init__` 时需手动导入子模块:** NitroGen random build 绕过了 `__init__` 的 import chain
44. **DiT per-step cost 完美线性:** NitroGen 7.2ms/step × k steps — 无 warmup/overhead
45. **Compute-bound → memory-BW-bound 转换点在 174M-350M:** 2x params 导致 4.4x latency (super-linear) — 关键的系统设计参考点
46. **Sparse clone + tar 比 git bundle 更适合第三方 repo:** git bundle 需要完整历史；sparse clone 只下载需要的文件

### 2026-04-25: Pi-Zero Dual-Stream Flow VLA (exp07a)
47. **Vendor namespace collision requires setup-time rename:** allenzren/open-pi-zero uses `from src.model...` internally — collides with our `src/`. Solution: `setup_pizero.sh` renames `vendor/open_pi_zero/src/` → `pizero_src/` and sed-rewrites all imports
48. **Manual phase timing for opaque models:** Pi-Zero's dual-stream architecture doesn't fit base class hook-on-module pattern. Override `register_profiling_hooks` as no-op, use explicit `timer.mark_start/end` in `model_inference` (same pattern as NitroGenController)
49. **Cross-attention makes per-step cost super-linear vs pure DiT:** 300M Gemma Expert with cross-attn to PaliGemma KV = ~18ms/step, vs naive linear extrapolation from 174M DiT (7.2ms) predicting ~12ms. Cross-attn overhead adds ~50%
50. **5 warmup runs insufficient for GPU power state stabilization:** Clear bimodal distribution (runs 1-12 high, 13-20 low). Need 10-15 warmup or explicit GPU power state locking (`nvidia-smi -pm 1`)
51. **uv venv works well for vendor-specific envs:** Pi-Zero needs specific torch/transformers versions incompatible with main env. `.venvs/pizero/` keeps it isolated without conda headaches on non-interactive SSH

---

## 8. Active Prompt Versions & Trade-offs

(No prompt versioning files yet — `prompts/` directory does not exist. When added, track here.)

---

## 9. Quick Reference

### Commands

| 命令 | 用途 |
|------|------|
| `bash scripts/sync_to_remote.sh` | Sync to xdlab23 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling` | Profiling GPU 0 |
| `bash scripts/launch_exp.sh 1 qwen_vl_7b/attention` | Attention GPU 1 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/attention_overlay` | Overlay viz GPU 0 |
| `bash scripts/launch_exp.sh 0 act/profiling` | ACT profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 openvla_7b/profiling` | OpenVLA profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 lingbot_vla_4b/profiling` | LingBot-VLA profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 pizero/profiling` | Pi-Zero profiling GPU 0 |
| `bash scripts/run_remote.sh <gpu> <config>` | SSH → xdlab23 launch |
| `bash scripts/run_local.sh <gpu> <config>` | Local GPU launch |
| `bash scripts/run_viewer.sh` | Flask viewer |
| `bash scripts/run_tests.sh` | pytest suite |
| `bash scripts/download-results.sh` | Download results |
| `bash scripts/setup_lingbot_vla.sh` | Setup LingBot-VLA env on xdlab23 |
| `python scripts/profile_fastwam.py --steps 10 --gpu 0` | Fast-WAM E/C/A profiling |
| `python scripts/profile_lingbot_va.py --mode random --gpu 0` | LingBot-VA E/V/A profiling |

### Server

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` (legacy, shared with rope2sink) |
| uv venv | `.venvs/lingbot-vla/` (LingBot-VLA) |
| Conda (WAM) | `fastwam` (Fast-WAM, Python 3.10, PyTorch 2.7.1+cu128) |
| GPUs | 8x RTX 5880 Ada 48GB |
| HF cache | `/data1/ybyang/huggingface` |
| Fast-WAM repo | `/data1/ybyang/FastWAM` |
| LingBot-VA repo | `/data1/ybyang/lingbot-va` |

### Registries

**Controllers:** `qwen_vl` → QwenVLController, `openvla` → OpenVLAController, `act` → ACTController, `lingbot_vla` → LingBotVLAController, `lingbot_va` → LingBotVAController, `nitrogen` → NitroGenController, `pizero` → PiZeroController (lazy)
**Tasks:** `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`, `attention_overlay`, `timing_validation`

### Knowhow Index

| File | Topic |
|------|-------|
| `docs/knowhow/runbooks/deploy-to-xdlab23.md` | xdlab23 部署流程 |
| `docs/knowhow/runbooks/setup-uv-env-xdlab23.md` | uv venv 替代 conda (非交互 SSH) |
| `docs/knowhow/toolchain/cuda-profiling-patterns.md` | CUDA Event vs torch.profiler 对比 |
| `docs/knowhow/toolchain/hydra-config-patterns.md` | Hydra ListConfig/device gotchas |
| `docs/knowhow/debug-solutions/gqa-attention-analysis.md` | GQA Q/K head mismatch |
| `docs/knowhow/debug-solutions/act-action-queue-hooks.md` | ACT action queue 缓存 |
| `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` | CPU backend no-op bug |
| `docs/knowhow/debug-solutions/qwen25vl-model-structure.md` | Qwen2.5-VL 模型结构 |
| `docs/knowhow/debug-solutions/qwen25vl-vision-token-mapping.md` | Vision token 定位 |
| `docs/knowhow/debug-solutions/lingbotvla-integration.md` | LingBot-VLA flow VLA 集成 14 个问题 |
| `docs/knowhow/debug-solutions/nitrogen-controller-deployment.md` | NitroGen 部署 5 个问题 (Hydra, hook_mode, SigLIP offline, imports, k sweep) |
| `docs/knowhow/debug-solutions/lingbot-va-wam-integration.md` | LingBot-VA WAM 集成 |
| `docs/knowhow/runbooks/deploy-new-model-package.md` | GitHub 被防火墙封锁时部署新模型包 (sparse clone→tar→scp) |
| `docs/knowhow/toolchain/wam-standalone-profiling.md` | WAM standalone profiling 模式 |
| `docs/knowhow/infrastructure/xdlab23-model-weights.md` | ModelScope 404 issue |
| `docs/papers/starvla-framework-deep-dive.md` | StarVLA 模块化 VLA framework 深度分析 |

---

*v6 — exp07a Pi-Zero dual-stream flow VLA profiling done. DiT/Expert scaling curve: 0.048ms→7.2ms→18ms→28.5ms→32ms. 10 experiments completed. Updated: 2026-04-25*
