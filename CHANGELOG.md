# Changelog

## v0.4.1 — 2026-04-15

### 新增
- **Timing Cross-Validation Task** (`timing_validation`) — 用 torch.profiler 独立测量 E/P/D phase 时长，与 PhaseTimer CUDA Events 对比验证
  - `src/tasks/validation_task.py` — `_ProfilerPhaseTracker` 在相同 phase boundary 注册 `record_function` hooks，同时运行两套测量
  - 对比报告：每 phase 的 deviation %，PASS (<5%) / WARN (5-15%) / FAIL (>15%) verdict
  - Gap 分析：`sum(E+P+D)` vs end-to-end wall clock，量化 projection/sampling 等漏测时间
  - 自动集成：profiling config 加 `timing_validation` 即启用，无需改动其他代码
- **Knowhow 归档**
  - `docs/knowhow/toolchain/cuda-profiling-patterns.md` — CUDA Event vs sync+perf_counter vs torch.profiler 三种模式对比、warmup 要求、统计方法、GPU clock 锁定
  - `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` — CPU backend record_end bug 根因与修复

## v0.4.0 — 2026-04-15

### 新增
- **Attention Overlay Visualization** — 将 VLM attention weights 映射回输入图像空间，渲染 heatmap 叠加可视化
  - `src/interpretability/` — Interpretability Mixin 体系：TokenSpatialMap、TokenType 数据结构，BaseInterpretabilityMixin 抽象接口
  - `src/interpretability/vlm_mixin.py` — Qwen2.5-VL 专用 token-to-image 空间映射，从 processor 的 `image_grid_thw` 读取 patch grid，扫描 `<|image_pad|>` token 定位 vision token 范围
  - `src/interpretability/vla_mixin.py` — Pi-Zero VLA placeholder（接口就绪，等模型权重）
  - `src/viz/overlay_renderer.py` — OverlayRenderer：JET/viridis colormap heatmap 叠加、multi-layer 横向对比 strip、GIF 动画输出
  - `src/tasks/attention_overlay_task.py` — 新 task：从 global_store 读 QK → 计算 attention → mixin 空间映射 → overlay 渲染
  - `configs/qwen_vl_7b/attention_overlay.yaml` — Hydra config，5 层 (0/7/14/21/27) attention overlay
- **E2E 验证通过** — Qwen2.5-VL-7B 单图输入，产出 5 层 overlay PNGs + multi_layer_strip.png + layers_sweep.gif + attention_data.json
- 灵感来源：[physical-AI-interpretability](https://github.com/villekuosmanen/physical-AI-interpretability) (overlay 渲染) + rope2sink (log-scale normalization, colormap 体系)

### 修复
- **GQA attention 计算** — `_compute_attention_scores()` 支持 Grouped Query Attention (K heads < Q heads)，通过 `repeat_interleave` 对齐
- **vision token ID 动态获取** — 从 tokenizer 解析 `<|image_pad|>` 而非硬编码 151655
- **layer 排序** — multi-layer strip 和 GIF 使用数字排序 (layer_7 在 layer_14 之前)
- **save_results 路径** — 使用 `base_output_path` 替代不存在的 `output_path`
- **OmegaConf 序列化** — save_results 中 ListConfig → plain list 再 JSON dump
- **launch_exp.sh** — 去掉 `device=cuda:0` override (base.yaml 已定义)
- **profiling_task.py** — 动态遍历所有 phase，不再硬编码 E/P/D
- **ACT controller** — model.forward() 直接调用 + observation.images 格式修复

## v0.3.1 — 2026-04-15

### 新增
- **ML Profiling 系统综合调研** — 8 个系统 (vLLM/SGLang/FastVideo/TensorRT-LLM/DeepSpeed/Triton/llama.cpp/MLC LLM) 的 profiling 实现对比 + CUDA timing 最佳实践
  - `survey/papers/ml-profiling-systems-comprehensive-survey.md` — 综合报告
  - `survey/papers/profiling-systems-survey.md` — 8 系统横向对比
  - `notes/sglang-profiling-deep-survey.md` — SGLang 深度调研
- **Profiling 统计增强** — median, P10/P90/P99, CV (变异系数)，CV>5% 自动警告 `[UNSTABLE]`

### 修复
- **PhaseTimer CPU backend bug** — `record_end()` 被 `if self._use_cuda:` 包裹导致 CPU backend 永远不会被调用
- **profiling_task.py mutation** — `single_run["decode"]` dict mutation 改为 immutable spread pattern

## v0.3.0 — 2026-04-15

### 新增
- **VLM Profiling Framework** (`src/`) — 完整的 VLM inference profiling + attention analysis 框架
  - `model-probe-core` 共享核心 package (git submodule at `src/core/`)，从 rope2sink 提取
  - BaseController → BaseVLMController → QwenVLController 三层继承链
  - PhaseTimer: CUDA event-based E/P/D timing (CPU fallback)
  - 4 个 analysis tasks: `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`
  - Hydra 配置驱动 (`configs/base.yaml`, `qwen_vl_7b/profiling.yaml`, `qwen_vl_7b/attention.yaml`)
- **xdlab23 服务器部署**
  - `scripts/sync_to_remote.sh` — git bundle 同步 (绕过 GitHub firewall)
  - `scripts/launch_exp.sh` — GPU-pinned 实验启动
  - `scripts/download-results.sh` — rsync 结果下载
  - `docs/knowhow/runbooks/deploy-to-xdlab23.md` — 部署 runbook
- **exp01a: Qwen2.5-VL-7B E/P/D Profiling** — 首个实验完成
  - Per-input profiling: text_only / single_image / multi_image
  - 10 benchmark runs + 3 warmup，CUDA event sub-ms 精度
  - 关键发现: Encode 随 image 线性增长，decode per-token 稳定 ~18-21ms
- **Framework Design Spec** (`docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md`)
- **Implementation Plan** (`docs/superpowers/plans/2026-04-14-vlm-profiling-framework.md`) — 12 tasks, 44 steps

### 修复
- PhaseTimer: decode timing 改为累加模式 (原 dict 覆盖只保留最后一个 step)
- PhaseTimer: CPU backend `record_end()` 改为立即记录时间 (原为 no-op)
- run_tasks.py: per-input profiling 隔离 (原混跑所有 inputs)
- run_tasks.py: Hydra struct mode + nested config unwrap
- QwenVLController: model path 修正 (`.model.language_model.layers`)
- QwenVLController: OmegaConf → plain dict 转换 for qwen_vl_utils
- probe_core hooks.py: extractor hook 错误改为 log 而非静默吞没

### 更新
- CLAUDE.md: 新增 framework 目录结构、server 信息、quick commands、current state
- project-skill SKILL.md: v0 → v1，新增 framework 架构、exp01a 数据、prediction calibration、7 条 engineering lessons
- .pipeline-state.json: stage dev → experiment, compute_env → xdlab23

## v0.2.1 — 2026-04-11

### 修复
- Dashboard: denoising chart inline styles 替换为 CSS classes
- Dashboard: 删除 canvas glow 死代码 (无效 hex→rgba)
- Dashboard: resize handler 加 rAF 节流
- Dashboard: IntersectionObserver 加 unobserve()
- Dashboard: header "8 Research Gaps" → "6 Research Entry Points" (与实际渲染一致)
- CLAUDE.md: Domain papers 路径从 `docs/papers/` 改为 `survey/papers/`

## v0.2.0 — 2026-04-11

### 新增
- **survey/papers/va-world-models-web.md** — WAM/Video WM/VA 最新论文 web 调研 (80+ 论文)，覆盖 DreamZero、NVIDIA Cosmos、DuoCore-FS 30Hz VLA、PocketDP3 等
- **viewer/static/survey-dashboard.html** — 交互式信息图 dashboard (Pareto frontier, pipeline comparison, maturity matrix, research gaps)
- **.claude/skills/project-skill/SKILL.md** — v0 bootstrap，7 章完整 context 文档 (overview, architecture, cognition, archive, experiments, lessons, reference)

### 修复
- 删除冗余 `experiments/` 目录，统一使用 `exp/`
- CLAUDE.md 目录结构描述补全实际目录
- Session startup 路径从空壳 `docs/papers/landscape.md` 改为 `survey/landscape.md`
- `.gitignore` 补充 Python/macOS/editor/ML artifacts 规则

## v0.1.0 — 2026-04-11

### 新增
- 项目初始化：LabMate 骨架 + git repo
- **survey/landscape.md** — VLM/VLA inference efficiency 全景 survey (80+ 论文)
- **survey/papers/recent-papers.md** — 2025-2026 最新论文 web 搜索 (40+ 论文)
- **survey/papers/va-world-models.md** — VA + World Action Model 深度分析 (60+ 论文)
- **notes/survey-plan.md** — 8 周调研行动计划
- CLAUDE.md 包含项目概述、survey 维度、LabMate workflow

### 其他
- selfOS wiki 同步更新：vlm-vla-real-time-systems concept 扩充、新建 world-action-model concept、方向启动 motivation source page
