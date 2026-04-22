# Changelog

## v0.6.0 @freemty — 2026-04-22

### 新增
- **exp05a 完成** — LingBot-VLA-4B attention analysis (6 layers: 0,7,14,21,28,35)
  - VLA fine-tuning 彻底重塑 attention: sink Pos 2→Pos 64, Gini 0.91→0.07, entropy V-shape→flat
  - 结论: VLM token pruning (FastV, FlashVLM) 不可迁移到 VLA
  - Attention structure 是 training objective property 而非 architecture property
- **exp05b 完成** — Qwen2.5-VL-3B-Instruct attention analysis (vanilla VLM baseline, ablation for exp05a)
  - 消歧成功: Gini 崩塌归因于 VLA fine-tuning，非 model size
  - 3B vanilla Gini 0.80-0.98 (与 7B 一致)，Pos 9 sink, entropy V-shape
  - `configs/qwen_vl_3b/attention.yaml` — 新增 vanilla 3B VLM config
- **exp06a 完成** — NitroGen 500M DiT E/C/A profiling + k sweep (k=1..16)
  - Per-step DiT 7.2ms, perfectly linear scaling
  - 174M DiT params, compute-bound → 174M-350M 之间转为 memory-BW-bound
  - k=1: 17.9ms (55.9Hz), k=16: 126ms (7.9Hz)
  - `NitroGenController` — SigLIP→VL-SA→DiT manual timing, random weights mode

### 变更
- Project skill v5 — exp05a/05b/06a findings, lessons #38-46, NitroGen in latency spectrum
- CLAUDE.md — current_exp 更新到 exp06a, 新增 exp05a/05b/06a key findings
- docs/TODO.md — NitroGen profiling 标记完成

### 文档
- `docs/knowhow/debug-solutions/nitrogen-controller-deployment.md` — NitroGen 部署 5 个问题 + Codex 审查发现
- `docs/knowhow/debug-solutions/lingbot-va-wam-integration.md` — LingBot-VA WAM 集成
- `docs/knowhow/toolchain/wam-standalone-profiling.md` — WAM standalone profiling 模式
- `docs/knowhow/runbooks/deploy-new-model-package.md` — sparse clone→tar→scp 部署流程

## v0.5.1 — 2026-04-21

### 新增
- **exp04a 完成** — Fast-WAM (skip-imagination WAM) E/C/A profiling on RTX 5880 Ada
  - @10step: E=7.6ms / C=36.7ms / A=362ms (total 407ms, 2.5Hz)
  - @20step: E=7.6ms / C=36.7ms / A=639ms (total 677ms, 1.5Hz)
  - Action phase 占 89-94%，30-layer MoT cross-attn × N steps
  - `scripts/profile_fastwam.py` — standalone profiler (random/full mode)
- **exp04b 完成** — LingBot-VA (full WAM, video imagination + action) E/V/A profiling
  - E=75.5ms / V=592.5ms (20 steps) / A=1423.1ms (50 steps), total 2091ms (0.5Hz)
  - Full WAM 比 skip-imagination 慢 ~5x
  - Video denoise ~29.6ms/step, action ~28.5ms/step (same DiT, similar per-step cost)
- **LingBotVAController** — 全新 WAM controller，E/V/A 三阶段手动 timer marks (同一 transformer 服务 video+action，module hooks 无法区分)
  - `scripts/profile_lingbot_va.py` — standalone profiler

### 修复
- Fast-WAM profiler: dummy_context 注释明确说明排除 text encoding 原因
- LingBot-VA init_pipeline: 使用独立 `repo_path` 参数而非混用 `model_name` (Codex review P1)
- LingBot-VA full mode: 不加载未使用的 4.7B text encoder，节省 ~9GB VRAM (Codex review P2)

### 文档
- Project skill v4 — exp04a/04b WAM findings, lessons #30-37, WAM benchmark baselines
- exp/summary.md — exp04a + exp04b rows done

## v0.5.0 — 2026-04-20

### 新增
- **Pi-Zero open-pi-zero 后端迁移** — 从 OpenPI (torch 2.7+) 切换到 open-pi-zero (allenzren)，与项目其他模型共享同一 uv venv
  - `PiZeroController` 全面重写：vendor path 校验、cuDNN probe-based 降级、random weights profiling 模式
  - Benchmark: 211ms total ≈ 4.7Hz (224x224, bf16, 10 denoise steps, RTX 5880 Ada)
  - Vendored code: `vendor/open_pi_zero/` (server-side, imports 从 `src.` 改为 `open_pi_zero.`)
- **uv 项目初始化** — `pyproject.toml` (torch>=2.5, transformers>=4.47, hydra, easydict)，替代 conda 作为 Pi-Zero/LingBot-VLA 的默认环境管理

### 变更
- `configs/pizero/profiling.yaml` — 简化为 open-pi-zero 后端配置 (model_name="" = random weights)
- README 更新：Pi-Zero 结果 (211ms/4.7Hz)、uv 安装说明、vendor/ 目录、server uv venv 信息

### 构建与工具链
- `pyproject.toml` — hatchling build backend, ruff config (line-length=100, py311)
- uv env on xdlab23: `.venv/` with torch 2.5+cu121, Python 3.12

## v0.4.4 — 2026-04-20

### 新增
- **exp03a 完成** — LingBot-VLA-4B E/C/A profiling (74.5ms ≈ 13Hz)
  - single_img: E=35.7ms / C=38.3ms / A=0.48ms
  - multi_view: E=36.3ms / C=38.3ms / A=0.48ms
  - 3B backbone 比 7B (exp01a) 快 ~7x
- `BaseVLAController._register_capture_hook()` — analysis hooks 支持 VLA 继承链

### 修复
- LingBotVLAController 继承链 AttributeError (`_register_capture_hook` 只在 VLM 基类)
- Empty safetensors guard — `init_pipeline` 找不到权重时抛 FileNotFoundError
- `scripts/run_remote.sh` command injection — shell 变量加引号
- `scripts/setup_lingbot_vla.sh` 补全 lingbotvla/lerobot 安装步骤，修正 model download 路径

### 文档
- `docs/knowhow/debug-solutions/lingbotvla-integration.md` — BaseVLAController 继承链问题
- Project skill v3 — exp03a findings, lessons #22-29, prediction calibration

## v0.4.3 — 2026-04-17

### 新增
- **Research Presentation Viewer** — 3 个新 HTML 页面，用于 advisor meeting 展示
  - `viewer/static/index.html` — 导航 hub，4 张卡片链接所有子页面
  - `viewer/static/presentation.html` — 5 section 汇报页 (hero, motivation, approach, experiments, roadmap)，含 timing bar chart、attention heatmap 可视化、architecture tree
  - `viewer/static/experiments.html` — 可展开的实验详情页，含数据表格和 prediction calibration
  - FARS dark slate 设计系统 (Instrument Serif + DM Sans + DM Mono, gold accent)
- `viewer/app.py` 新增 catch-all static file 路由

### 修复
- Flask catch-all 路由加 `api/` 前缀防护，避免未注册 API 路径被遮蔽
- multi_image total 数据修正 (`~1.1s` → `~3.6s` at 128 tokens)
- `index.html` body `overflow:hidden` → `overflow-x:hidden; overflow-y:auto` (移动端可滚动)
- Attention heatmap 添加 "illustrative pattern" 标注，避免与实际数据混淆

## v0.4.2 — 2026-04-17

### 文档
- **README.md 全面重写** — 新增 Status Overview (Done/TODO 表格)、3 个实验结果汇总、完整 Architecture 图、7 个 Hydra config 索引、Scripts 索引表、Adding New Models 指南
- TODO 清单覆盖 8 个待完成项（Attention overlay server run, OpenVLA, Pi-Zero, Gradient saliency 等），标注优先级和阻塞原因

### 构建与工具链
- `scripts/run_remote.sh` — 一键 SSH 到 xdlab23 启动实验（封装 launch_exp.sh）
- `scripts/run_local.sh` — 本地 GPU 实验启动（带 logs 输出）
- `scripts/run_viewer.sh` — Flask viewer 启动
- `scripts/run_tests.sh` — pytest 测试套件启动

## v0.4.1 — 2026-04-15

### 新增
- **Timing Cross-Validation Task** (`timing_validation`) — 用 torch.profiler 独立测量 E/P/D phase 时长，与 PhaseTimer CUDA Events 对比验证
  - `src/tasks/validation_task.py` — `_ProfilerPhaseTracker` 在相同 phase boundary 注册 `record_function` hooks，同时运行两套测量
  - 对比报告：每 phase 的 deviation %，PASS (<5%) / WARN (5-15%) / FAIL (>15%) verdict
  - Gap 分析：`sum(E+P+D)` vs end-to-end wall clock，量化 projection/sampling 等漏测时间
  - 自动集成：profiling config 加 `timing_validation` 即启用，无需改动其他代码

### 实验结果
- **exp01b: Qwen2.5-VL-7B attention analysis** (5 layers) — Pos 2 (first visual patch) 是 universal attention sink (12K-18K received, 12-28x vs #2)。Text→Visual Gini >0.91 (extreme sparsity, supports token pruning)。Layer 21 entropy 最低 (3.44)
- **exp02a: ACT (LeRobot) profiling** — Total ~3ms (850x faster than VLM)。Encode ~2.5-2.8ms (80%)，Action ~0.4-0.8ms。VLA latency 下界

### 修复
- 消除 controller 双重注册 (\_\_init\_\_.py + run_tasks.py 重复 import)
- `_aggregated_timing` 从动态属性改为 BaseVLM/VLAController 显式声明
- `head_dim` divisibility check — 防止非 128 head_dim 模型的静默错误
- Multi-image heatmap key collision + defensive detach before numpy

### 文档
- Knowhow 归档 (7 个文件):
  - `docs/knowhow/toolchain/cuda-profiling-patterns.md` — CUDA Event vs torch.profiler 对比
  - `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` — CPU backend bug
  - `docs/knowhow/debug-solutions/act-action-queue-hooks.md` — ACT action queue 缓存
  - `docs/knowhow/debug-solutions/gqa-attention-analysis.md` — GQA Q/K head mismatch
  - `docs/knowhow/infrastructure/xdlab23-model-weights.md` — 更新 OpenVLA ModelScope 404
  - vision token mapping、Hydra ListConfig/device gotchas

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
