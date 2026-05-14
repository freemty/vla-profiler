---
name: project-skill
description: "Use when advising on project architecture, experiment history, codebase navigation, or research findings."
user-invocable: false
version: v9
note: "v9 — LIBERO eval pipeline (Cosmos 97.4%, Fast-WAM 94.5%, exp04d running). OFT bottleneck flip (action→backbone). 9 models profiled + 3 LIBERO evals complete. Two acceleration paths: A (Action DiT) vs A' (OFT + backbone compression)."
updated_at: "2026-05-14"
---

# vlla — Project Knowledge

> VLM/VLA Real-Time Systems Survey & Research
> UCSD PhD 方向调研项目 | 导师: 张昊 (Hao Zhang) — vLLM/FastVideo/Chatbot Arena 作者
> v9 — **LIBERO eval pipeline** (Cosmos 97.4%, Fast-WAM 94.5%, exp04d running) + **OFT bottleneck flip** (Action 82% → Backbone 84-99%). 9 models profiled, 3 LIBERO evals complete. Two acceleration paths: **Path A** (Action DiT 加速) vs **Path A'** (OFT + backbone compression).

---

## 1. Project Overview & Current State

**项目名称:** vlla (VLM/VLA Real-Time Systems)
**核心问题:** 如何让 VLM/VLA 在实时约束下高效运行？
**研究定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

**动机:**
张昊的技术路线: Parameter Server -> Alpa -> vLLM -> FastVideo -> **VLM/VLA real-time systems**。每一步都是 ML Systems 前沿的下一个自然问题。VLM/VLA serving 正处于 "pre-vLLM" 阶段，存在巨大的系统研究空间。

**当前阶段:** LIBERO eval + Hao meeting prep v2
- `current_exp`: reproducibility + LIBERO eval (exp06b/04c/07b real-weight done, exp04e/cosmos_libero/exp11a/exp11b done, exp04d running)
- `stage`: meeting prep final ("Fast VLA first, serving later") + LIBERO eval pipeline
- `version`: v0.11.0
- Survey 产出: 6 份核心文档 (新增 `awesome-wam-survey-2026.md` — 首篇 WAM 系统性 survey)，覆盖 180+ 篇论文/项目 (2024-2026)
- Framework 产出: VLM profiling + attention analysis + attention overlay + VLA profiling + stream-aware PhaseTimer + co-location probe (`src/`, `scripts/`)
- 新增模块: Interpretability Mixin 体系 (`src/interpretability/`)、OverlayRenderer (`src/viz/`)、Timing Cross-Validation、stream-aware PhaseTimer (`src/utils/timing.py` 支持 `stream=` 参数)
- **LIBERO eval pipeline**: `scripts/run_cosmos_libero.py`, `run_exp04d_parallel.sh`, `lerobot_stub/` (PI0Config + PreTrainedPolicy stub)
- 共享核心: `src/core/probe_core/` (inlined, no longer submodule since v0.10.0)
- 共享统计 helper: `scripts/_profiling_stats.py` — standalone profiling scripts 统一的 median/percentile 口径
- 服务器: xdlab23 (8x RTX 5880 Ada 48GB)，18+ 实验完成 + 1 running
- **完成的 profiling 实验 (9 models):** exp01a (Qwen2.5-VL-7B), exp01b (attention), exp02a (ACT), exp03a (LingBot-VLA-4B), exp04a (Fast-WAM), exp04b (LingBot-VA), exp05a/b (attention analysis), exp06a (NitroGen DiT), exp07a (Pi-Zero), exp09a (Cosmos Policy), exp11a (OpenVLA-OFT), exp11b (StarVLA-OFT)
- **完成的 LIBERO evals:** exp04e (Fast-WAM 94.5%), cosmos_libero (Cosmos Policy 97.4%), [exp04d LingBot-VA running]
- **Shelved:** exp03b (LingBot-VLA, no LIBERO finetune ckpt), exp07c (Pi-Zero, gated repo)
- **exp08a/b/c (EPDA contention):** INVALIDATED 2026-04-27, archived. See `_archive/`.
- **战略转向 (2026-04-27 "Fast VLA first, serving later"):** VLA 单请求延迟 (Pi-Zero 200ms=5Hz, 需 10-50Hz) 是当前真瓶颈。候选排序: **A (Action DiT 加速) > B (VLA benchmark) > C (DiT caching) >> D (serving, too early)**。
- **v0.10.0 OFT 发现:** OFT (parallel MLP action head) 翻转瓶颈: Action 82% → 0.1-0.2ms, backbone 成为 84-99% bottleneck → **Path A'** 出现 (OFT + backbone compression)
- **下一步:** P0 exp04d 完成 → 汇总 reproducibility matrix → Hao meeting v2 (9 models + OFT 翻转 + 双路径)

**核心数据汇总:**

| Exp | Model | Key Metric |
|-----|-------|-----------|
| exp01a | Qwen2.5-VL-7B | E=253ms (58%), D=18-21ms/tok |
| exp01b | Qwen2.5-VL-7B | Pos 2 = universal sink (12-28x), Gini >0.91 |
| exp02a | ACT (LeRobot) | Total ~3ms, 850x faster than VLM |
| exp03a | LingBot-VLA-4B | E=35.7ms/C=38.3ms/A=0.48ms, total 74.5ms ≈ 13Hz |
| exp04a | Fast-WAM (ActionDiT, 6.7B) | @10step: E=7.6ms/C=36.7ms/A=362ms, total 407ms, 2.5Hz |
| exp04b | LingBot-VA (full WAM, 5B DiT) | **canonical (rerun 2026-04-27)**: E=84.7/V=697/A=1708ms, total 2518ms, 0.40Hz |
| exp05a | LingBot-VLA-4B (attention) | VLA fine-tuning reshapes attention: sink Pos2→Pos64, Gini 0.91→0.07 |
| exp05b | Qwen2.5-VL-3B (attention) | Disambiguation: Gini collapse is VLA fine-tuning, not model size |
| exp06a | NitroGen 500M DiT | Per-step 7.2ms, linear scaling. 174M→350M = BW transition |
| exp07a | Pi-Zero (dual-stream flow VLA) | **stable** E=9.32/C=26.40/A=164.76ms, total 200.5ms ≈ 5Hz |
| exp09a | Cosmos Policy (2B DiT) | Action-only 659ms/1.5Hz. Per-step 76.8ms. 1-step→342ms/2.9Hz |
| exp11a | OpenVLA-OFT (Llama-2 7B) | E=16.8/C=92.3/A=0.24ms → 109ms (9.2Hz). Backbone 84% |
| exp11b | StarVLA-OFT (Qwen2.5-VL-3B) | E=34.7/C=28.5/A=0.13ms → 63ms (15.8Hz). Backbone 99.8% |
| exp04e | Fast-WAM LIBERO | **94.5%** (spatial 91.5/object 100/goal 97/10 89.5), 800 ep |
| cosmos_libero | Cosmos Policy LIBERO | **97.4%** (spatial 96/object 100/goal 98/10 95.5), 800 ep |

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
|   |-- core/              # Inlined probe_core (was git submodule, inlined v0.10.0)
|   |   |-- probe_core/
|   |       |-- controller.py   # HookMode enum, BaseController (StoreMixin + ABC)
|   |       |-- hooks.py        # HookManager: path resolution, hook registration helpers
|   |       |-- state.py        # StoreMixin: step_store/global_store lifecycle
|   |       |-- registry.py     # Registry class (dict-based dispatch)
|   |-- controllers/
|   |   |-- base_vlm_controller.py  # BaseVLMController: E/P/D phase hooks, PhaseTimer
|   |   |-- base_vla_controller.py  # BaseVLAController: E/C/A phase hooks, _register_capture_hook
|   |   |-- qwen_vl_controller.py   # QwenVLController: model loading, inference, QKV hooks
|   |   |-- openvla_controller.py   # OpenVLAController: DINOv2+SigLIP→Llama-2 (+ OFT head)
|   |   |-- act_controller.py       # ACTController: ResNet18→CVAE→action chunk
|   |   |-- lingbot_vla_controller.py  # LingBotVLAController: Qwen2.5-VL-3B + flow action head
|   |   |-- lingbot_va_controller.py   # LingBotVAController: full WAM, E/V/A phases (VAE+5B DiT video+action)
|   |   |-- nitrogen_controller.py     # NitroGenController: SigLIP→VL-SA→DiT, E/C/A manual timing, random weights
|   |   |-- pizero_controller.py    # PiZeroController: Pi-Zero (lazy import, separate env)
|   |   |-- starvla_controller.py   # StarVLAController: Qwen2.5-VL-3B + OFT MLP head
|   |-- interpretability/           # Attention overlay system
|   |   |-- base_mixin.py     # TokenSpatialMap, TokenType, BaseInterpretabilityMixin
|   |   |-- vlm_mixin.py      # VLMInterpretabilityMixin (Qwen2.5-VL token→image mapping)
|   |   |-- vla_mixin.py      # VLAInterpretabilityMixin (Pi-Zero placeholder)
|   |-- viz/                        # Visualization
|   |   |-- overlay_renderer.py     # OverlayRenderer: heatmap overlay, strip, GIF
|   |-- eval/                       # LIBERO eval consolidation
|   |   |-- consolidate_matrix.py   # JSON→reproducibility matrix aggregator
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
|   |-- qwen_vl_7b/        # Qwen2.5-VL-7B (profiling, attention, overlay)
|   |-- act/               # ACT (profiling)
|   |-- openvla_7b/        # OpenVLA-OFT (profiling)
|   |-- lingbot_vla_4b/    # LingBot-VLA (profiling)
|   |-- nitrogen/          # NitroGen (profiling + k sweep)
|   |-- pizero/            # Pi-Zero (profiling)
|
|-- scripts/               # Server deployment & eval scripts
|   |-- sync_to_remote.sh         # Git bundle sync to xdlab23
|   |-- launch_exp.sh             # Launch experiment on server GPU
|   |-- download-results.sh       # Rsync results from server
|   |-- run_local.sh / run_remote.sh / run_viewer.sh / run_tests.sh
|   |-- setup_lingbot_vla.sh / setup_cosmos_policy.sh / setup_libero_envs.sh
|   |-- profile_fastwam.py / profile_lingbot_va.py   # Standalone WAM profiling
|   |-- exp09a_cosmos_policy_profiling.py             # Cosmos Policy profiling
|   |-- _profiling_stats.py       # Shared statistics helper (median/percentile)
|   |-- run_cosmos_libero.py      # Cosmos LIBERO eval
|   |-- run_exp03b_libero.py      # LingBot-VLA LIBERO eval (shelved)
|   |-- run_exp04d_parallel.sh    # LingBot-VA 4-GPU LIBERO eval
|   |-- run_exp07c_libero.py      # Pi-Zero LIBERO eval (shelved)
|   |-- run_libero_all.sh         # 5-model parallel launcher
|   |-- lerobot_stub/             # Minimal PI0Config + PreTrainedPolicy (avoids version conflicts)
|-- viewer/                # Research presentation viewer (Flask)
|   |-- app.py
|   |-- static/
|       |-- index.html / presentation.html / experiments.html
|       |-- design-space.html     # Action Model Design Space dashboard
|       |-- reproducibility.html  # Latency + LIBERO success matrix
|-- survey/                # 文献综述
|-- exp/summary.md         # 实验 flight recorder
|-- slides/                # Hao meeting slides (hao-meeting-v2.html)
|-- docs/
|   |-- specs/             # Experiment specs
|   |   |-- libero-eval-inference-flows.md  # NEW: 三模型闭环推理对比
|   |-- knowhow/           # 基础设施/工具链/调试笔记
|   |-- TODO.md / learning-plan.md / hao-meeting-prep-v2.md
```

### 2.2 Framework 继承链

```
probe_core.BaseController          # Model-agnostic: hook lifecycle, StoreMixin, HookMode
  |
  +-> BaseVLMController            # VLM-specific: E/P/D phases, PhaseTimer, _register_capture_hook
  |     +-> QwenVLController       # Qwen2.5-VL: model loading, QKV hooks, VLMInterpretabilityMixin
  |     +-> OpenVLAController      # OpenVLA (AR VLA): DINOv2+SigLIP→Llama-2 7B + OFT head
  |     +-> (future) InternVLController
  |
  +-> BaseVLAController            # VLA-specific: E/C/A phases, _register_capture_hook (独立定义)
        +-> ACTController          # ACT (LeRobot): ResNet18→CVAE→action chunk, single-forward
        +-> LingBotVLAController   # LingBot-VLA-4B: Qwen2.5-VL-3B + 10-step flow action head
        +-> LingBotVAController    # LingBot-VA (full WAM): VAE + 5B WanDiT, E/V/A phases
        +-> NitroGenController      # NitroGen 500M: SigLIP→VL-SA→DiT flow matching, E/C/A manual timing
        +-> PiZeroController       # Pi-Zero: VLM backbone + flow action head (lazy import, separate env)
        +-> StarVLAController      # StarVLA-OFT: Qwen2.5-VL-3B + parallel OFT MLP head
```

**VLM vs VLA Phase Models:**
- **VLM (BaseVLMController):** E/P/D — Encode / Prefill / Decode (autoregressive)
- **VLA (BaseVLAController):** E/C/A — Encode / Context / Action (C optional, A may iterate)
  - `has_context_phase()`: True for VLA-with-VLM-backbone (LingBot-VLA, Pi-Zero), False for pure VA (ACT)
  - `get_denoise_steps()`: 1 for single-forward (ACT, OFT), N for flow/diffusion models (LingBot-VLA: 10)
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
- probe_core inlined since v0.10.0 (no longer git submodule — avoids sync issues)
- **lerobot_stub** pattern: minimal dataclass stubs to break circular dependency (lerobot 0.5.1 requires transformers <=4.51, lingbotvla needs 4.57)

### 2.3 Server Deployment

```
Local (Mac) --git bundle+scp--> xdlab23 (8x RTX 5880 Ada)
            <--rsync results---  /data1/ybyang/vlla
```

SSH: `ssh xdlab23_yang` | HF: `/data1/ybyang/huggingface`

**Environment strategy:**
- `vit-probe` (conda) — shared with rope2sink, for Qwen2.5-VL / OpenVLA / ACT / LingBot-VA / Cosmos Policy
- `.venvs/lingbot-vla/` (uv) — LingBot-VLA specific (lerobot compat, PyTorch 2.8)
- `.venvs/pizero/` (uv) — Pi-Zero (allenzren/open-pi-zero, vendored `pizero_src/`)
- `fastwam` (conda) — Fast-WAM (Python 3.10, PyTorch 2.7.1+cu128)
- **cuDNN fix**: all DiT envs need `LD_LIBRARY_PATH` prepend for pip cuDNN 9.10 (system has 9.1.1)

---

## 3. System Cognition

### 3.1 三大战略判断 (Survey 核心结论)

1. **VLM serving 处于 "pre-vLLM" 阶段** — EPD disaggregation 类似 2022 年 PD 分离
2. **VLA inference 处于 "wild west" 阶段** — 无统一 serving system/benchmark
3. **Algorithm-System Co-design 是最大空白** — token pruning/quantization 未与 scheduling 整合

### 3.1.1 系统选型三层心智模型 (from `vla-wam-efficiency-systems-deep-research.md`)

**Serving 层级三分:**
- **第一层 通用 serving 主干** (必进 shortlist): vLLM / SGLang / TensorRT-LLM — 覆盖 PagedAttention/RadixAttention/in-flight batching, 通用量化并行
- **第二层 workload 专项加速器**: vLLM-Omni (any-to-any stage disaggregation, JCT ↓91.4%) / SGLang Diffusion (Cache-DiT + TeaCache + layerwise offload, Qwen-Image ≤5x)
- **第三层 工程友好层**: LMDeploy (中文 VLM, vs vLLM 1.8x 吞吐, 4-bit 2.4x FP16)

**Robotics 层级 (与 serving 不在同一 bucket):**
- **LeRobot** (工作流基座, 23.6k★): Parquet+MP4 (LeRobotDataset v3), RTC 异步分块推理, policy_server, IsaacLab Arena GPU eval
- **OpenVLA-OFT** (高效微调 recipe, 1.2k★): parallel decoding + action chunking + continuous actions + L1; LIBERO 76.5%→97.1%, A100 **4.2Hz/240ms → 109.7Hz/73ms** (25-50x)
- **Fast-WAM** (WAM latency baseline, 579★): 项目页 190ms/单步 (vs exp04a 测 407ms total @10step = 补足了 per-step 视角), LIBERO 97.6 / RoboTwin 91.8

**统一端点 benchmark**: GenAI-Perf — TTFT/ITL/TPS/RPS, 支持多模态 synthetic/BYOD

**核心 anti-pattern**: 不要把 LeRobot/OpenVLA-OFT/Fast-WAM 与 vLLM/TRT-LLM 直接比 tok/s — 正确对比是"同任务成功率下的控制 Hz / 单步 latency / 显存"

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
- **Multi-view encode 几乎不增加 (+1.5%):** patchified input 在 ViT 之前聚合
- **Flow action head 极快 (0.48ms):** 10-step denoise loop 的 action_out_proj 累计仅 0.48ms
- **Total 74.5ms ≈ 13Hz:** 接近机器人 10Hz 实时需求

**From exp04a (first-party) — Fast-WAM ActionDiT profiling:**
- **Action phase dominates at 89-94%:** @10step A=362ms of 407ms total
- **Per-step cost ~32ms/step:** 每个 Euler step 运行完整 30-layer ActionDiT 通过 MoT cross-attention
- **Paper's 190ms 是 A100/H100 + 5 steps:** RTX 5880 + 10 steps = 407ms
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

**From exp06a (first-party) — NitroGen 500M DiT profiling:**
- **Per-step DiT cost = 7.2ms, perfectly linear across k=1..16:** 完美线性
- **174M DiT params (not 100M as estimated):** VL-SA 28M 独立于 DiT
- **Action dominates 91.6% at k=16:** 与 Fast-WAM/LingBot-VA pattern 一致
- **SigLIP encode = 8.7ms, VL self-attention = 1.9ms:** 两者都很轻
- **k=1 achieves 55.9Hz:** 通过 step distillation 可达实时
- **174M→350M: 2x params, 4.4x latency → super-linear:** compute-bound 到 memory-BW-bound 转换点在 174M-350M 之间

**From exp07a (first-party) — Pi-Zero dual-stream flow VLA:**
- **Action Expert (300M Gemma) dominates 82%:** per-step ~16.5ms
- **Cross-attn to PaliGemma KV makes 300M Expert ~2.3x pricier than pure DiT**
- **Bimodal 污染:** runs 1-12 slow 1.25x (GPU power warmup) → warmup=15 + `nvidia-smi -pm 1` standard

**From exp09a (first-party) — Cosmos Policy direct-mode profiling:**
- **Action-only 5-step: 659ms / 1.5Hz** (parallel_gen=False)
- **Full (action+future+value) 5-step: 1363ms / 0.73Hz** (parallel_gen=True)
- **Linear fit (R²=0.9975)**: fixed cost 265ms + **76.8ms/step** (2B DiT denoise)
- **1-step distilled 预测: 342ms / 2.9Hz** — still too slow for 10Hz
- **Peak VRAM 8816MB, 1.96B params**

**From exp11a/11b (first-party) — OFT VLA profiling:**
- **OFT MLP action head = 0.13-0.24ms:** 1270x faster than flow head (165ms)
- **Bottleneck flips from Action to Backbone:** OpenVLA 84% backbone (Llama-2 7B prefill), StarVLA 99.8% backbone
- **StarVLA-OFT 63ms ≈ 15.8Hz:** already meets 10Hz real-time with 3B backbone
- **Two paths emerge:** Path A (compress Action DiT) vs Path A' (OFT + compress Backbone)

**From LIBERO evals (first-party):**
- **Fast-WAM 94.5% (800 ep, paper 93.7%):** reproducible within noise
- **Cosmos Policy 97.4% (800 ep, paper 98.5%):** RTX 5880 vs A100 + seed accounts for 1.1pp gap
- **Random-weight timing faithful:** Pi-Zero real vs random Δ=12%, NitroGen Δ<2%

### 3.3 WAM Profiling 综合洞察 (exp04a + exp04b + exp09a + exp11a/b)

**完整 latency spectrum (RTX 5880 Ada, bf16):**

| Model | Paradigm | Total | E | V | A | Hz | Bottleneck |
|-------|----------|-------|---|---|---|-----|-----------|
| ACT | VA (CVAE) | 3ms | 2.5ms | — | 0.5ms | 330 | E (80%) |
| NitroGen @k16 | VA (flow DiT) | 126ms | 8.7ms | — | 115.4ms | 7.9 | A (91.6%) |
| **StarVLA-OFT** | **VLA (OFT MLP)** | **63ms** | 34.7ms | — | 0.13ms | **15.8** | **E+C (99.8%)** |
| LingBot-VLA | Flow VLA | 74.5ms | 35.7ms | — | 38.8ms | 13 | E≈C (50/50) |
| **OpenVLA-OFT** | **AR VLA (OFT MLP)** | **109ms** | 16.8ms | — | 0.24ms | **9.2** | **C (84%)** |
| Pi-Zero @10step | Flow VLA (dual-stream) | 200.5ms | 9.32ms | — | 164.76ms | ~5 | A (82%) |
| Fast-WAM @10step | WAM (skip) | 407ms | 7.6ms | — | 362ms | 2.5 | A (89%) |
| **Cosmos Policy @5step** | **WAM (EDM DiT)** | **659ms** | ~265ms(fixed) | — | ~384ms | **1.5** | **A (58%)** |
| LingBot-VA (full, rerun) | WAM (full) | 2518ms | 84.7ms | 697ms | 1708ms | 0.40 | A (69%) |

**关键规律:**
1. **OFT 翻转发现 (v0.10.0):** Action head 从 80-94% bottleneck 变成 <1% when using OFT MLP → backbone becomes the new bottleneck
2. **Action head scaling 决定 WAM 瓶颈:** 小 action head (0.48ms) vs 大 MoT action (362ms) — 架构决定
3. **Video generation 代价:** Full WAM vs skip 的 5x 差距几乎全来自 video denoise (592ms)
4. **Per-step cost 在同一 DiT 内高度一致:** V/A ~29ms/step — step count 是线性的
5. **DiT per-step scaling:** 174M=7.2ms < 300M=16.5ms (cross-attn) < 350M=32ms < 2B=76.8ms
6. **两条加速路径:** Path A (压 Action DiT: step distillation/caching/sparsity) vs Path A' (OFT + backbone compression: 3B→1.5B / quantization / token pruning)

### 3.4 活跃假设

- [x] ~~Vision encoding 占 >50% prefill 延迟~~ → **exp01a 已验证 (58%)**
- [x] ~~Attention sink 存在于 VLM~~ → **exp01b 已验证 (Pos 2 universal sink)**
- [x] ~~Text→Visual attention sparse, supports pruning~~ → **exp01b Gini >0.91**
- [x] ~~Flow VLA (real model) 延迟在 50-100ms 范围~~ → **exp03a 验证 74.5ms**
- [x] ~~Flow action head overhead 可忽略~~ → **exp03a 验证 0.48ms (0.6% of total)**
- [x] ~~WAM latency dominated by action diffusion~~ → **exp04a/04b 验证 (68-94%)**
- [x] ~~Full WAM significantly slower than skip-imagination~~ → **exp04b 验证 5x slower**
- [x] ~~VLA fine-tuning 改变 attention pattern?~~ → **exp05a 验证: 彻底重塑**
- [x] ~~Gini 崩塌是 model size 还是 fine-tuning 效应?~~ → **exp05b 消歧: fine-tuning effect**
- [x] ~~NitroGen DiT per-step cost at ~100M params?~~ → **exp06a: 174M DiT, 7.2ms/step**
- [x] ~~Compute-bound vs memory-BW-bound DiT size transition?~~ → **174M-350M 之间**
- [x] ~~Pi-Zero flow VLA: denoise step count vs latency~~ → **exp07a: 200.5ms, 16.5ms/step**
- [x] ~~OFT action head latency?~~ → **exp11a/b: 0.13-0.24ms, negligible**
- [x] ~~Cosmos Policy direct-mode latency (论文未报)?~~ → **exp09a: 659ms/1.5Hz, per-step 76.8ms**
- [x] ~~Fast-WAM LIBERO 可复现?~~ → **exp04e: 94.5% vs paper 93.7%, ✓**
- [x] ~~Cosmos Policy LIBERO 可复现?~~ → **cosmos_libero: 97.4% vs paper 98.5%, ✓**
- [ ] EPD 三阶段分离实际收益 (disaggregation ROI)
- [ ] VLM speculative decoding 中 visual token 对 acceptance rate 的影响
- [ ] Per-layer attention entropy 分布的实际意义 (Layer 21 最低 → 是否是最佳 pruning 切入点?)
- [ ] 3B→7B backbone scaling law: 是否线性? (exp01a vs exp03a 初步暗示 ~7x for 2.3x params)
- [ ] Imagination value quantification: Full WAM 的 592ms video generation 带来多少 task success rate 提升?
- [ ] LingBot-VA LIBERO accuracy (exp04d running): 预计完成后与 Fast-WAM/Cosmos 形成三角对比
- [ ] OFT + quantized backbone (4-bit 3B) 能否保持 accuracy 同时 <30ms?

---

## 4. Technical Archive

### 4.1 四大范式对比 (expanded with OFT)

| 维度 | VA (1-step flow) | VLA (OFT MLP) | VLA (flow head) | VLA (flow DiT) | WAM |
|------|-----------------|---------------|-----------------|----------------|-----|
| 延迟 | 1-5ms | **63-109ms** | 74ms | 200-660ms | 400-2500ms |
| 控制频率 | >200Hz | **9-16Hz** | ~13Hz | 1.5-5Hz | 0.4-2.5Hz |
| 瓶颈 | Encode (80%) | **Backbone (84-99%)** | E≈C | Action (82-94%) | Action (58-89%) |
| 泛化能力 | 低 | 高 | 高 | 高 | 极高 |
| 加速路径 | — | **Path A' (backbone compression)** | Smaller backbone | **Path A (step distill/caching)** | Step distill + skip-imagination |

**First-party baselines (from experiments):**
- ACT (single-forward VA): ~3ms total → VA 1-5ms 范围 ✓
- StarVLA-OFT (3B backbone, OFT MLP): 63ms ≈ 15.8Hz → **already real-time**
- OpenVLA-OFT (7B backbone, OFT MLP): 109ms ≈ 9.2Hz → close to real-time
- LingBot-VLA-4B (flow VLA, 3B backbone): 74.5ms total ≈ 13Hz
- Pi-Zero (flow DiT, 300M Expert): 200.5ms ≈ 5Hz
- Cosmos Policy (2B EDM DiT): 659ms ≈ 1.5Hz
- Fast-WAM @10step (skip-imagination, 6.7B): 407ms → 2.5Hz
- LingBot-VA (full WAM, 5B DiT): 2518ms → 0.40Hz

### 4.2 Benchmark Baselines (RTX 5880 Ada, bf16)

| Config | E | C | V (video) | A (action) | Total | Hz |
|--------|---|---|-----------|------------|-------|-----|
| ACT | 2.5ms | — | — | 0.5ms | 3ms | 330 |
| NitroGen @k1 | 8.7ms | 1.9ms | — | 7.2ms | 17.9ms | 55.9 |
| StarVLA-OFT | 34.7ms | 28.5ms | — | 0.13ms | 63ms | 15.8 |
| LingBot-VLA | 35.7ms | 38.3ms | — | 0.48ms | 74.5ms | 13 |
| OpenVLA-OFT | 16.8ms | 92.3ms | — | 0.24ms | 109ms | 9.2 |
| Pi-Zero @10step | 9.32ms | 26.40ms | — | 164.76ms | 200.5ms | ~5 |
| Fast-WAM @5step | 7.1ms | — | — | 164ms | 257ms | 3.9 |
| Fast-WAM @10step | 7.6ms | 36.7ms | — | 362ms | 407ms | 2.5 |
| Cosmos Policy @5step | ~265ms(fixed) | — | — | ~384ms | 659ms | 1.5 |
| LingBot-VA @20V+50A | 84.7ms | — | 697ms | 1708ms | 2518ms | 0.40 |

**Per-step cost reference:**
- LingBot-VLA flow action head: ~0.048ms/step
- OFT MLP head: single-forward, ~0.13-0.24ms total
- NitroGen DiT (12L, 1024 hidden, cross-attn to 256 tokens): ~7.2ms/step — compute-bound
- Pi-Zero Action Expert (18L, 1024 hidden, Gemma + cross-attn to PaliGemma KV): ~16.5ms/step
- Fast-WAM ActionDiT (30L, 1024 hidden, MoT cross-attn): ~32ms/step
- LingBot-VA WanDiT (30L, 3072 hidden): ~34ms/step
- Cosmos Policy (Cosmos-Predict2-2B, EDM): ~76.8ms/step

### 4.3 LIBERO Quality Baselines

| Model | Spatial | Object | Goal | 10 | Avg | Source |
|-------|---------|--------|------|-----|-----|--------|
| **Cosmos Policy** | 96.0% | 100% | 98.0% | 95.5% | **97.4%** | Our eval (800 ep) |
| Cosmos Policy (paper) | — | — | — | — | 98.5% | Gao+ ICLR 2026 |
| **Fast-WAM** | 91.5% | 100% | 97.0% | 89.5% | **94.5%** | Our eval (800 ep) |
| Fast-WAM (paper) | — | — | — | — | 93.7% | Paper |
| LingBot-VA | — | — | — | — | TBD | exp04d running |

### 4.4 Pareto 前沿

| 方法 | 延迟 | Hz | LIBERO% | 意义 |
|------|------|-----|---------|------|
| Action-to-Action Flow | 0.56ms | >1000 | — | VA 速度下界 |
| **ACT (our)** | **3ms** | **330** | — | VA baseline |
| **NitroGen @k1 (our)** | **17.9ms** | **55.9** | — | VA flow DiT baseline |
| Mean-Flow VLA / FASTER | ~50ms | ~20 | — | VLA 单步化 |
| **StarVLA-OFT (our)** | **63ms** | **15.8** | — | **OFT VLA real-time** |
| **LingBot-VLA (our)** | **74.5ms** | **13** | — | Flow VLA baseline |
| **OpenVLA-OFT (our)** | **109ms** | **9.2** | 97.1%* | AR VLA + OFT |
| DreamZero | ~130ms | 7 | — | WAM zero-shot (A100) |
| **Pi-Zero @10step (our)** | **200.5ms** | **~5** | — | Dual-stream flow VLA |
| **Fast-WAM @5step (our)** | **257ms** | **3.9** | **94.5%** | Skip-imagination |
| **Cosmos Policy @5step (our)** | **659ms** | **1.5** | **97.4%** | 2B EDM DiT |
| **LingBot-VA (our)** | **2518ms** | **0.40** | TBD | Full WAM |

*OpenVLA-OFT paper number on LIBERO

### 4.5 Rejected Alternatives & Rationale

| Decision | Alternative Considered | Why Rejected |
|----------|----------------------|-------------|
| Mixin-based interpretability | Subclass-per-model | Mixin allows mix-and-match without deep hierarchy |
| PhaseTimer CUDA Events | torch.profiler only | CUDA Events give sub-ms precision; torch.profiler added as cross-validation |
| Git bundle sync | rsync only | Bundle preserves git history on server side |
| ACT model.forward() | policy.select_action() | select_action() has action queue cache, skips forward on subsequent calls |
| GQA repeat_interleave | Separate Q/K head handling | repeat_interleave is HF standard, clean single codepath |
| uv venv (LingBot-VLA) | conda (shared vit-probe) | conda not available in non-interactive SSH; uv is PATH-independent |
| Separate _register_capture_hook per base | Shared mixin | Two inheritance chains (VLM/VLA) need independent hook definitions |
| eager attention (LingBot-VLA) | flash-attn / flex_attention | flex_attention dtype bug in PyTorch 2.8; flash-attn "not implemented" in lingbotvla |
| Standalone scripts (WAM profiling) | Hydra controller integration | Fast-WAM and LingBot-VA have self-contained pipelines |
| Random init for WAM timing | Real checkpoint weights | Timing depends on compute graph not weight values; saves 12GB+ download |
| Inline probe_core (v0.10.0) | Keep git submodule | Submodule adds sync friction; rope2sink diverged; 819 lines small enough to inline |
| lerobot_stub | Pin lerobot 0.5.1 | lerobot 0.5.1 forces transformers <=4.51; lingbotvla needs 4.57. Stub breaks circular dep |
| TypedDict LossKwargs shim | Downgrade transformers | transformers 4.57 needed for Qwen2.5-VL; LossKwargs removal is trivial to shim |

---

## 5. Experiment History

| Exp ID | Date | Status | Prediction | Actual | Key Finding | Calibration |
|--------|------|--------|-----------|--------|-------------|-------------|
| exp01a | 2026-04-15 | **done** | E >50% of total | E=58% | E scales linearly with images; D per-token stable ~18-21ms | Accurate (structure), Off (hardware: 3-5x gap to A100) |
| exp01b | 2026-04-15 | **done** | Sink exists (literature) | Pos 2 sink 12-28x | Universal sink at first visual patch; Gini >0.91 supports pruning | Confirmed (existence), Surprised (Gini >0.91 extreme) |
| exp02a | 2026-04-15 | **done** | VA ~1-5ms (literature) | ~3ms total | ACT 850x faster than VLM; ResNet18 encode 80% of total | Accurate (range), Off (resolution insensitivity) |
| exp03a | 2026-04-20 | **done** | 250-350ms (7B→3B ~2x scaling) | 74.5ms total | 3B backbone 7x faster than 7B; Context≈Encode; flow action 0.48ms; ≈13Hz | Way Off — actual 3.4-4.7x better than predicted |
| exp03b | 2026-05-13 | **shelved** | 0% (no LIBERO finetune) | N/A | lingbot-vla-4b is pretrained foundation, no LIBERO ckpt exists | — |
| exp04a | 2026-04-21 | **done** | E=15-25ms/C=100-130ms/A=30-50ms | @10step: E=7.6ms/C=36.7ms/A=362ms | Action dominates 89%; per-step 32ms; paper's 190ms is A100+5step | Way Off on A (predicted 30-50ms, got 362ms) |
| exp04b | 2026-04-21 (rerun 04-27) | **done** | E=10-15ms/V=600-900ms/A=500-750ms | E=84.7/V=697/A=1708ms, total 2518ms | Full WAM ~6x skip-imagination; Action 69%; 0.40Hz | V nailed, E 6-8x under, A 2.3x under |
| exp04c | 2026-04-28 | **done** | ~190ms (paper) | 257ms / 3.9Hz | Paper 190ms on A100; RTX 5880 1.35x slower. Per-step ~41ms | Accurate (direction), +35% hardware gap |
| exp04d | 2026-05-13 | **running** | TBD | TBD (4-GPU parallel) | LingBot-VA real-weight LIBERO eval, 20 ep × 4 suites | — |
| exp04e | 2026-04-29 | **done** | ~93-95% (paper 93.7%) | 94.5% avg (800 ep) | Fast-WAM LIBERO reproduced within noise | Accurate |
| exp05a | 2026-04-21 | **done** | VLA attention similar to VLM | Gini 0.91→0.07, sink migrates | VLA fine-tuning reshapes attention; VLM pruning not transferable | Way Off — opposite of prediction |
| exp05b | 2026-04-21 | **done** | 3B model = different pattern | 3B vanilla Gini 0.80-0.98 | Gini collapse from VLA fine-tuning, not model size | Confirmed disambiguation |
| exp06a | 2026-04-22 | **done** | ~100M DiT, 5-10ms/step | 174M DiT, 7.2ms/step, linear | Per-step 7.2ms, compute-bound, 174M→350M BW transition | Accurate on per-step, Off on param count |
| exp06b | 2026-04-28 | **done** | Same as 06a (random≈real) | 7.1ms/step (DiT=181M) | Confirms random-weight timing faithful | Accurate |
| exp07a | 2026-04-25 | **done** | E=15-25ms/C=40-80ms/A=30-80ms | E=9.32/C=26.40/A=164.76ms | Action Expert 82%. Per-step 16.5ms. Cross-attn 2.3x tax | E/C overestimated, A severely underestimated (2x) |
| exp07b | 2026-04-28 | **done** | ~200ms (same as 07a random) | 225ms (+12%) | Real weights slightly slower; validates random-weight timing | Accurate (12% within noise) |
| exp07c | 2026-05-13 | **shelved** | N/A | N/A | Pi-Zero LIBERO ckpt gated (Physical Intelligence) + GitHub firewall | — |
| exp09a | 2026-04-29 | **done** | ~300-500ms (2B DiT) | 659ms / 1.5Hz (action-only) | Per-step 76.8ms (R²=0.9975 linear). Fixed cost 265ms. 1-step→342ms/2.9Hz | Under by ~30% (missed VAE fixed cost) |
| cosmos_libero | 2026-05-13 | **done** | ~97-98% (paper 98.5%) | 97.4% avg (800 ep) | Reproduced on RTX 5880, -1.1pp from paper | Accurate |
| exp11a | 2026-05-11 | **done** | ~100-150ms (7B AR, OFT fast) | 109ms / 9.2Hz | Llama-2 7B prefill = 84% bottleneck. OFT MLP = 0.24ms | Accurate |
| exp11b | 2026-05-11 | **done** | ~50-80ms (3B, OFT fast) | 63ms / 15.8Hz | OFT MLP 0.13ms. Backbone 99.8% bottleneck. Already 10Hz+ | Accurate |

### 5.1 Prediction Calibration: exp01a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Vision encoding 占比 | >50% | 58% | **Accurate** |
| Decode per-token | 2-5ms (A100 literature) | ~18-21ms (RTX 5880) | **Off** — 硬件差异 ~4x |
| Encode 线性扩展 | 线性 (ViT) | 2.14x for 2 images | **Confirmed** |

### 5.2 Prediction Calibration: exp01b

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Attention sink 存在 | 存在 (文献) | Pos 2 universal, 12-28x | **Confirmed** |
| Text→Visual sparsity | Sparse (token pruning 文献) | Gini >0.91 | **Confirmed** — 极端稀疏 |

### 5.3 Prediction Calibration: exp02a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| VA latency range | 1-5ms (文献) | ~3ms | **Accurate** |
| Encode 占比 | 主导 (CNN bottleneck) | 80% | **Confirmed** |
| Resolution sensitivity | 应该有影响 | Minimal impact | **Off** — ResNet18 对 resolution 不敏感 |

### 5.4 Prediction Calibration: exp03a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Total latency | 250-350ms (7B→3B ~2x scaling) | 74.5ms | **Way Off** — 实际 7x |
| E/C relative cost | Encode dominant | E≈C (35.7 vs 38.3ms) | **Off** — 未预料 context 如此重 |
| Action overhead | <5ms (flow is fast) | 0.48ms | **Accurate** |
| Multi-view scaling | ~2x for 2 cameras | +1.5% only | **Way Off** — patchified aggregation |

### 5.5 Prediction Calibration: exp04a (Fast-WAM)

| Metric | Survey Prediction | Actual (@10step) | Calibration |
|--------|------------------|-----------------|-------------|
| E (VAE encode) | ~15-25ms | 7.6ms | **Off** — VAE much lighter |
| C (Video prefill) | ~100-130ms | 36.7ms | **Off** — single-frame |
| A (flow matching, 10 steps) | ~30-50ms | 362ms | **Way Off** — missed 30L MoT |
| Total | ~150-200ms | 407ms | **Off** — 2x over due to A underestimation |

### 5.6 Prediction Calibration: exp04b (LingBot-VA)

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| E (VAE encode) | ~10-15ms | 75.5ms | **Way Off** — streaming VAE z_dim=48 |
| V (Video denoise, 20 steps) | ~600-900ms | 592.5ms | **Accurate** — nailed the lower bound |
| A (Action denoise) | ~500-750ms | 1423ms | **Off** — missed 50 steps (2.5x more) |

### 5.7 Prediction Calibration: exp09a (Cosmos Policy)

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Total (action-only, 5-step) | ~300-500ms | 659ms | **Off** — underestimated fixed cost (265ms VAE) |
| Per-step DiT cost | ~50-80ms (2B model) | 76.8ms | **Accurate** — within predicted range |
| Linear scaling | Expected | R²=0.9975 confirmed | **Confirmed** |

### 5.8 Prediction Calibration: exp11a/b (OFT)

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| OFT action overhead | <1ms | 0.13-0.24ms | **Accurate** — OFT is trivial |
| OpenVLA total (7B) | 100-150ms | 109ms | **Accurate** |
| StarVLA total (3B) | 50-80ms | 63ms | **Accurate** |
| Bottleneck location | Backbone dominant | 84-99.8% backbone | **Confirmed** — flipped from Action |

---

## 6. Prediction Calibration — Meta-Learning

**系统性偏差:**

1. **结构性判断准确:** 哪个阶段是瓶颈、存在 attention sink、sparsity 可利用 — consistently correct
2. **硬件映射偏差:** 文献数据 (A100/H100) 到 RTX 5880 有 3-5x gap，需要 hardware correction factor
3. **量级估计偏差 (小模型方向):** ResNet18 对 resolution 不敏感; patchified input 绕过重复 ViT
4. **Gini 系数超预期:** 预测 "sparse" 但没预测到 >0.91 的极端稀疏度
5. **Backbone scaling 非线性:** 预测 7B→3B 是 ~2x speedup，实际 7x
6. **Context phase 被低估:** LLM forward (KV cache fill) 与 ViT encode 几乎等价
7. **WAM action phase 被严重低估:** 每个 denoise step 运行完整 MoT (30 layers × cross-attn)。正确推算: `steps × layers × per-layer-cost`
8. **Step count 遗漏:** LingBot-VA action 50 steps (not 20 like video)。**教训: 必须先 grep config defaults 再做预测**
9. **VAE 类型差异巨大:** Fast-WAM 7.6ms vs LingBot-VA 75.5ms vs Cosmos 265ms(fixed) — 10-35x 差异。不能假设 "VAE always cheap"
10. **OFT 预测校准提升:** exp11a/b predictions 全部 Accurate — 说明对 backbone-dominated 模型的理解已校准

**校准精度趋势:**
- exp01a: 2/3 accurate — 结构 OK，硬件数字偏差
- exp01b: 2/2 confirmed — 文献预测准
- exp02a: 2/3 accurate — resolution 不敏感是 miss
- exp03a: 1/4 accurate — 最差，backbone scaling 理解不足
- exp04a: 0/4 accurate (direction correct) — MoT cross-attn 代价未预料
- exp04b: 1/4 accurate (V phase nailed) — step count + streaming VAE 低估
- exp09a: 1/3 accurate — per-step ok, fixed cost underestimated
- **exp11a/b: 3/4 accurate — 校准显著改善** (OFT + backbone 理解已内化)
- **LIBERO evals: 2/2 accurate — quality 预测稳定**

**改进方向:**
- 对 backbone scaling 使用 FLOPs ratio 而非 param ratio
- WAM 预测必须先确认: (1) per-step architecture (layers × heads × attn type), (2) exact step counts from config
- 预测前 grep `num_inference_steps` / `action_steps` 等 config 字段
- Fixed cost (VAE/preprocessing) 需要单独估算，不要假设 cheap
- **已验证:** 对已知架构 (OFT, known backbone) 的预测精度已达到 Accurate 级别

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
12. **GQA (Grouped Query Attention) 必须显式处理:** Qwen2.5-VL Q=28 heads, K=4 heads → `repeat_interleave` 对齐，否则 reshape crash。影响所有 GQA 模型
13. **head_dim divisibility check 不能省:** 非 128 head_dim 模型会静默产出错误结果
14. **ACT select_action() 有 action queue 缓存陷阱:** 只有 1/chunk_size 的调用实际触发 forward pass → profiling 必须直接调 model.forward()
15. **Controller 双重注册要避免:** `__init__.py` import 和 `run_tasks.py` 重复 import 会导致 registry collision
16. **Multi-image heatmap key collision:** 多图输入的 layer→image mapping 需要唯一 key (加 image_key suffix)
17. **OmegaConf ListConfig 不能直接 JSON dump:** 需要先 `OmegaConf.to_container()` 转 plain list
18. **vision token ID 不能硬编码:** `<|image_pad|>` token ID 应从 tokenizer 动态获取 (非固定 151655)
19. **Timing cross-validation 有价值:** PhaseTimer vs torch.profiler 双测验证，发现 sum(E+P+D) vs wall clock 有 gap
20. **_aggregated_timing 必须显式声明:** 动态属性 → 显式声明为 `Optional[Dict]` in `__init__`，防止 AttributeError
21. **detach before numpy:** `attn_probs.detach().numpy()` — 忘记 detach 会在 backward-enabled tensor 上 crash

### 2026-04-20: LingBot-VLA-4B + Codex Review (v0.4.2 + v0.4.3)
22. **两条继承链各自需要 hook 方法:** `_register_capture_hook` 在 BaseVLMController 有定义，但 BaseVLAController 继承自 BaseController — 需在 BaseVLAController 中也定义
23. **Empty weights glob 必须 guard:** safetensors glob 返回空列表时应抛 FileNotFoundError
24. **Shell 脚本变量必须 quote:** `$CONFIG` 未加引号 → command injection。修复: `"$CONFIG"`
25. **PyTorch `.forward()` 不触发 hooks:** lingbotvla 源码调用 `self.model.forward(...)` 而非 `self.model(...)`，hooks 只在 `__call__` 时触发
26. **uv 替代 conda 解决非交互 SSH 问题:** conda 依赖 `.bashrc` 中的 `conda init`，非交互 SSH 不 source `.bashrc`
27. **PI0Config 字段过滤:** config.json 含自定义字段，用 `dataclasses.fields()` 过滤有效字段 + `setattr` 附加
28. **Patchified input 绕过 multi-view scaling:** patchify 并聚合多视角输入 → ViT 处理合并后的 patch 序列
29. **Flask catch-all 路由需要 api/ 前缀防护:** 无条件 catch-all 会遮蔽 API 路由

### 2026-04-21: WAM Profiling — Fast-WAM + LingBot-VA (exp04a + exp04b)
30. **先 grep config defaults 再做 latency 预测:** step count 差异直接导致 2x 预测偏差。**规则: 所有 diffusion model 预测前必须确认 num_inference_steps**
31. **"Skip-imagination" 不意味着 "small model":** Fast-WAM 仍加载完整 5B video expert
32. **MoT cross-attention per step 代价极高:** 30-layer ActionDiT + MoT = ~32ms/step
33. **同一 DiT 的 V/A per-step cost 高度一致:** V (~29.6ms) ≈ A (~28.5ms) — sequence length 影响可忽略
34. **Standalone profiling scripts 比 Hydra controller 更适合 external repos**
35. **Random init 对 timing 结果有效:** timing 取决于 compute graph，不取决于 weight values
36. **High variance 可能来自 GPU power management:** 短 workload 受 GPU power state 波动影响
37. **WAM Action phase 占比 68-94% 是架构常数:** 在多个模型上一致

### 2026-04-21: VLA Attention Analysis (exp05a + exp05b)
38. **VLA fine-tuning 彻底重塑 attention pattern:** 不能假设 VLM 特性在 VLA 中保持
39. **消歧实验必不可少:** 无法区分 model size vs fine-tuning → 需要 vanilla control
40. **Attention structure 是 training objective property:** 不是 architecture property

### 2026-04-22: NitroGen 500M DiT Profiling (exp06a)
41. **controller_config 中 `mode` 字段被 framework 截取:** 自定义用途需用不同 key name
42. **离线环境 `from_pretrained()` 需替换为 config-based 构建**
43. **`__new__` 绕过 `__init__` 时需手动导入子模块**
44. **DiT per-step cost 完美线性:** NitroGen 7.2ms/step × k steps — 无 warmup/overhead
45. **Compute-bound → memory-BW-bound 转换点在 174M-350M:** 2x params 导致 4.4x latency (super-linear)
46. **Sparse clone + tar 比 git bundle 更适合第三方 repo**

### 2026-04-25: Pi-Zero Dual-Stream Flow VLA (exp07a)
47. **Vendor namespace collision requires setup-time rename:** `src/` collision → rename to `pizero_src/` + sed-rewrite imports
48. **Manual phase timing for opaque models:** override `register_profiling_hooks` as no-op, use explicit timer marks
49. **Cross-attention makes per-step cost super-linear vs pure DiT:** 300M with cross-attn = 16.5ms vs linear 12ms prediction. ~35% overhead
50. **5 warmup runs insufficient for GPU power state stabilization:** Need 10-15 warmup or `nvidia-smi -pm 1`
51. **uv venv works well for vendor-specific envs:** keeps versions isolated without conda headaches

### 2026-04-27: Profiling Audit + exp08 + Scope Audit
52. **exp07a bimodal 归因 GPU 功率爬坡:** warmup=15 + `nvidia-smi -pm 1` 为后续默认
53. **两 thread + 两 CUDA stream 做 co-location probe 有效**
54. **Roofline 模型在 GPU kernel-launch-level 预测失败:** LLM 侧膨胀 >> diffusion A 侧
55. **NitroGen controller import 需要 repo root 在 sys.path**
56. **HF_HOME path = `/data1/ybyang/huggingface/Qwen/...`:** 直接是 org/repo 结构
57. **sync_to_remote.sh merge 在 untracked 冲突时静默失败**
58. **Claim "写新 framework" 前必须 code-level 扫描最新 tree:** scope audit 必须做到代码级别

### 2026-05-11: OFT VLA Profiling (exp11a/b, v0.10.0)
59. **AutoModelForVision2Seq 不等于 AutoModelForCausalLM:** OpenVLA 用前者，Prismatic 不注册到 CausalLM registry，用错类会静默加载空模型
60. **6-channel input (stacked current+goal images) 必须在 controller 手动构造:** OFT 的 image processor 不自动做 RGB concat
61. **grid_thw 必须和 pixel_values 维度对齐:** Qwen2.5-VL ViT 需要 explicit grid_thw tensor 描述每张图的 patch 网格
62. **embed_tokens 在 Llama-2 是 `model.embed_tokens` 不是 `model.model.embed_tokens`:** 层级取决于 HF 的 wrapper 结构
63. **OFT head 极简 (Linear→ReLU→Linear):** 整个 action head 可用 3 行代码手写，无需从 checkpoint 加载

### 2026-05-13: LIBERO Eval Pipeline (v0.11.0)
64. **cuDNN 版本冲突是 DiT 模型的统一 blocker:** 系统 cuDNN 9.1.1 vs torch 需要 9.10。修复: `LD_LIBRARY_PATH` 前置 pip 安装的 `nvidia-cudnn-cu12` 的 lib 路径
65. **LossKwargs 在 transformers 4.57 被移除:** lingbotvla 继承 FlashAttentionKwargs (now TypedDict in 4.57)，LossKwargs 需要用 TypedDict stub 注入回去
66. **lerobot 0.5.1 强制 transformers <=4.51:** 但 lingbotvla 需要 4.57。解法: `lerobot_stub/` 提供最小 PI0Config + PreTrainedPolicy，避免 import lerobot
67. **LIBERO env 必须用 OffScreenRenderEnv:** `bench.get_env()` 不存在，需要手动 `OffScreenRenderEnv(env_name)` + `set_init_state(init_states[0])`
68. **apply_rotary_emb shim:** transformers 4.57 修改了 rotary embedding API，需要兼容 shim
69. **use_flash_attention_2 废弃:** transformers 4.57 统一用 `attn_implementation` kwarg，旧 bool flag 触发 deprecation error
70. **4-GPU parallel eval 模式:** server-client 分离，一个 GPU 跑 model server，同 GPU 上 client 发 action 请求。`run_exp04d_parallel.sh` 实现
71. **LIBERO eval 耗时:** 20 ep × 10 tasks × 4 suites = 800 episodes，~20 小时 (单 GPU WAM 推理 2.5s/step)

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
| `bash scripts/launch_exp.sh 0 act/profiling` | ACT profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 lingbot_vla_4b/profiling` | LingBot-VLA profiling GPU 0 |
| `bash scripts/launch_exp.sh 0 pizero/profiling` | Pi-Zero profiling GPU 0 |
| `bash scripts/run_remote.sh <gpu> <config>` | SSH → xdlab23 launch |
| `bash scripts/run_local.sh <gpu> <config>` | Local GPU launch |
| `bash scripts/run_viewer.sh` | Flask viewer |
| `bash scripts/run_tests.sh` | pytest suite |
| `bash scripts/download-results.sh` | Download results |
| `python scripts/profile_fastwam.py --steps 10 --gpu 0` | Fast-WAM E/C/A profiling |
| `python scripts/profile_lingbot_va.py --mode random --gpu 0` | LingBot-VA E/V/A profiling |
| `python scripts/exp09a_cosmos_policy_profiling.py --gpu 0` | Cosmos Policy profiling |
| `python scripts/run_cosmos_libero.py` | Cosmos LIBERO eval |
| `bash scripts/run_exp04d_parallel.sh 20` | LingBot-VA 4-GPU LIBERO eval |
| `bash scripts/run_libero_all.sh` | 5-model parallel LIBERO launcher |

### Server

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` (legacy, shared with rope2sink) |
| Conda (WAM) | `fastwam` (Python 3.10, PyTorch 2.7.1+cu128) |
| uv venv | `.venvs/lingbot-vla/` / `.venvs/pizero/` |
| GPUs | 8x RTX 5880 Ada 48GB |
| HF cache | `/data1/ybyang/huggingface` |
| Fast-WAM repo | `/data1/ybyang/FastWAM` |
| LingBot-VA repo | `/data1/ybyang/lingbot-va` |
| cuDNN fix | `export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__.replace('__init__.py','lib'))"):$LD_LIBRARY_PATH` |

### Registries

**Controllers:** `qwen_vl` → QwenVLController, `openvla` → OpenVLAController, `act` → ACTController, `lingbot_vla` → LingBotVLAController, `lingbot_va` → LingBotVAController, `nitrogen` → NitroGenController, `pizero` → PiZeroController (lazy), `starvla` → StarVLAController
**Tasks:** `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`, `attention_overlay`, `timing_validation`

### Knowhow Index

| File | Topic |
|------|-------|
| `docs/knowhow/runbooks/deploy-to-xdlab23.md` | xdlab23 部署流程 |
| `docs/knowhow/runbooks/setup-uv-env-xdlab23.md` | uv venv 替代 conda |
| `docs/knowhow/runbooks/deploy-new-model-package.md` | 防火墙封锁时部署模型包 |
| `docs/knowhow/runbooks/install-libero.md` | LIBERO 环境安装 |
| `docs/knowhow/toolchain/cuda-profiling-patterns.md` | CUDA Event vs torch.profiler |
| `docs/knowhow/toolchain/hydra-config-patterns.md` | Hydra ListConfig/device gotchas |
| `docs/knowhow/toolchain/wam-standalone-profiling.md` | WAM standalone profiling 模式 |
| `docs/knowhow/toolchain/fastwam-libero-eval.md` | Fast-WAM LIBERO eval 接口 |
| `docs/knowhow/toolchain/shell-script-safety-patterns.md` | Shell 安全模式 |
| `docs/knowhow/debug-solutions/gqa-attention-analysis.md` | GQA Q/K head mismatch |
| `docs/knowhow/debug-solutions/act-action-queue-hooks.md` | ACT action queue 缓存 |
| `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` | CPU backend no-op bug |
| `docs/knowhow/debug-solutions/qwen25vl-model-structure.md` | Qwen2.5-VL 模型结构 |
| `docs/knowhow/debug-solutions/qwen25vl-vision-token-mapping.md` | Vision token 定位 |
| `docs/knowhow/debug-solutions/lingbotvla-integration.md` | LingBot-VLA 14 个问题 |
| `docs/knowhow/debug-solutions/nitrogen-controller-deployment.md` | NitroGen 5 个问题 |
| `docs/knowhow/debug-solutions/lingbot-va-wam-integration.md` | LingBot-VA WAM 集成 |
| `docs/knowhow/debug-solutions/openvla-oft-integration.md` | OpenVLA/StarVLA OFT 5 陷阱 |
| `docs/knowhow/debug-solutions/pizero-integration.md` | Pi-Zero 集成 |
| `docs/knowhow/debug-solutions/concurrent-cuda-stream-profiling-pitfalls.md` | 并发 CUDA stream profiling |
| `docs/knowhow/debug-solutions/conda-env-model-compat.md` | Conda env 兼容矩阵 |
| `docs/knowhow/debug-solutions/libero-env-creation.md` | LIBERO OffScreenRenderEnv API |
| `docs/knowhow/infrastructure/xdlab23-model-weights.md` | ModelScope 404 issue |
| `docs/knowhow/infrastructure/xdlab23-cudnn-mismatch.md` | cuDNN 9.1→9.10 fix |
| `docs/specs/libero-eval-inference-flows.md` | 三模型闭环推理对比 |

---

*v9 — LIBERO eval pipeline (Cosmos 97.4%, Fast-WAM 94.5%, exp04d running). OFT bottleneck flip (Action→Backbone). 9 models profiled, 3 LIBERO evals complete. Two paths: A (Action DiT) vs A' (OFT + backbone). 21 experiments total (18 done, 1 running, 2 shelved). Updated: 2026-05-14*
