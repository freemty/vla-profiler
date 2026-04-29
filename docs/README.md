# Documentation Index

Project documentation map. All paths relative to repo root.

## Survey (Literature)

| File | Description |
|------|-------------|
| `survey/landscape.md` | VLM/VLA inference efficiency 全景 survey (80+ papers) |
| `survey/papers/recent-papers.md` | 2025-2026 最新论文 web 搜索 (40+ papers) |
| `survey/papers/va-world-models.md` | VA + World Action Model 深度分析 — 五大架构族谱, WM 五层分类 (L0-L4) |
| `survey/papers/va-world-models-web.md` | WAM/Video WM/VA 最新论文 web 调研 (80+ papers, 2025-2026) |
| `survey/papers/profiling-systems-survey.md` | 8 大 ML inference system profiling 实现横向对比 |
| `survey/papers/ml-profiling-systems-comprehensive-survey.md` | 4-agent 并行调研综合报告 — PhaseTimer 审查, CUDA timing 最佳实践 |
| `survey/papers/flashdrive-deep-dive.md` | FlashDrive 论文深度分析 |
| `survey/papers/efficient-wam-va-profiling-targets.md` | WAM/VA profiling 目标模型筛选 |
| `survey/papers/nitrogen-deep-dive.md` | NitroGen 500M VA gaming foundation model 精读 |
| `survey/papers/dreamdojo-dreamzero-deep-dive.md` | DreamDojo + DreamZero 精读 (WAM, DiT caching, 7Hz on GB200) |
| `survey/papers/hao-style-fastvideo.md` | FastVideo 深度 survey — STA/VSA/蒸馏 + Hao co-design 方法论提炼 + VLA 迁移启示 |
| `survey/papers/hao-style-distserve.md` | DistServe 深度 survey — PD disaggregation, goodput 指标, EPD 三阶段 VLM 迁移启示 |
| `survey/papers/hao-style-vllm.md` | vLLM/PagedAttention 深度 survey — OS 分页借用, BlockManager+Scheduler, Visual KV 迁移启示, co-design 方法论起点 |
| `survey/papers/hao-style-synthesis.md` | **Hao 工作风格综合分析** — 三工作共享的 5-step co-design 模板, VLM/VLA 下一步实验候选 A/B/C/D 打分, 见 Hao 的 "一页话" |
| `survey/papers/multimodal-serving-systems-2026.md` | **vLLM-Omni + SGLang Diffusion 实地调研** — 代码级 scope 对比, arXiv:2602.02204 解读, exp08 原计划方向被覆盖后的真正空白区 (B1-B4) |
| `survey/papers/vla-wam-serving-systems-2026.md` | **VLA/WAM 专用 serving 系统现状** — 唯二真 system: OxyGen (arXiv:2603.14371) + VLAgents (arXiv:2601.11250); 18 arXiv ID 主进程校验零幻觉; 补 B5 (streaming VAE WAM) + B6 (OxyGen 扩 PD) 两个空白 |
| `survey/papers/vla-acceleration-tricks-2026.md` | **9 篇 model-level VLA 加速汇总** — PD-VLA/Discrete Diffusion VLA/SnapFlow/FASTER/StreamingVLA/A1/NanoVLA/OpenVLA-OFT/Fast-WAM，按 parallel-decoding/one-step-flow/model-shrinking/ft-recipe 四族分类。SnapFlow 独立验证 "A 阶段 80%" 与 exp07a 82% 吻合。只推荐 PD-VLA + OpenVLA-OFT 做 deep-dive |
| `survey/papers/pi-series-evolution.md` | **Physical Intelligence π series + Generalist GEN-1 演进** — π0 (2024-10, 我们的 exp07a baseline) → π0.5 (2025-04, co-training) → π*0.6 (2025-11, Recap RL) → **π0.7 (2026-04, pipeline system)** + GEN-1 (2026-04, "model is a system" 原话)。两家同月承认 VLA 变 system，直接升级候选 D' 的 scope |
| `survey/papers/industrial-wam-landscape-2026.md` | **工业 WAM 全景** — 用 binary 分类（推理时 video-gen vs 仅 pretrain）区分两派。runtime-video-gen 少数派 (1X 1XWM / Rhoda FutureVision "Direct Video Action" / NVIDIA DreamZero / World Labs RTFM) vs VLA 主流 (PI / Figure Helix / Generalist / Wayve / Cosmos)。关键引用逐字 curl 反验 |

## Experiment Results

| Dir | Model | Status | Key Metric |
|-----|-------|--------|------------|
| `exp/exp01a/` | Qwen2.5-VL-7B (profiling) | Done | E=253ms, D=18-21ms/tok |
| `exp/exp01b/` | Qwen2.5-VL-7B (attention) | Done | Pos 2 sink 12-28x, Gini >0.91 |
| `exp/exp02a/` | ACT (profiling) | Done | Total ~3ms, 850x vs VLM |
| `exp/exp03a/` | LingBot-VLA-4B (profiling) | Done | 74.5ms, 13Hz |
| `exp/exp04a/` | Fast-WAM (profiling) | Done | 407ms@10step, 2.5Hz |
| `exp/exp04b/` | LingBot-VA (profiling) | Done (rerun 2026-04-27) | **canonical**: E=84.7/V=697/A=1708ms (2518ms, 0.40Hz) |
| `exp/exp05a/` | LingBot-VLA-4B (attention) | Done | VLA reshapes attention: Gini 0.91→0.07 |
| `exp/exp05b/` | Qwen2.5-VL-3B (attention) | Done | Disambiguation: Gini collapse = VLA fine-tuning |
| `exp/exp06a/` | NitroGen 500M DiT (profiling) | Done | 7.2ms/step, linear, k=1: 55.9Hz |
| `exp/exp07a/` | Pi-Zero dual-stream (profiling) | Done | **stable**: E=9.32/C=26.40/A=164.76ms (200.5ms, ~5Hz) |
| `exp/exp04c/` | Fast-WAM 5-step paper-aligned | Done (v0.9.0) | 257ms / 3.9Hz (paper 190ms A100) |
| `exp/exp04d/` | LingBot-VA real-weight LIBERO | Deferred | ckpt found, env compat issue |
| `exp/exp04e/` | **Fast-WAM LIBERO-4 eval** | Done (v0.9.0) | **94.5% avg** (spatial 91.5 / object 100 / goal 97 / 10 89.5), 800 ep |
| `exp/exp06b/` | NitroGen 500M real-weight | Done (v0.9.0) | 7.1ms/step (DiT=181M, identical to exp06a) |
| `exp/exp07b/` | Pi-Zero real-weight profiling | Done (v0.9.0) | 225ms total (random 200ms, Δ=12%) |
| `exp/summary.md` | Flight recorder | — | All experiments, one row each |

## Design Specs & Plans

| File | Description |
|------|-------------|
| `docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md` | Profiling framework 设计 spec — BaseController 继承链, PhaseTimer, hook 架构 |
| `docs/superpowers/specs/2026-04-15-attention-overlay-visualization-design.md` | Attention overlay 可视化设计 — Mixin 体系, TokenSpatialMap, OverlayRenderer |
| `docs/superpowers/plans/2026-04-14-vlm-profiling-framework.md` | Framework 实施计划 — 12 tasks, 44 steps |
| `docs/superpowers/plans/2026-04-15-attention-overlay-visualization.md` | Attention overlay 实施计划 |
| `docs/specs/2026-04-26-epda-disaggregation-spec.md` | **exp08 spec** — EPDA 四阶段干扰量化 (L1→L2 跨越, DistServe-style motivation figure) |
| `docs/specs/2026-04-26-epda-roofline-analysis.md` | **exp08 roofline 分析** — E/P/D/A 在 RTX 5880 Ada 上的 AI/utilization 坐标, go/no-go 论证, 干扰矩阵预测 |
| `docs/specs/2026-04-28-reproducibility-spec.md` | **Reproducibility spec (v0.9.0)** — 7 模型官方配置合约 (权重/架构/步数/benchmark) + 已知偏差 |
| `docs/superpowers/plans/2026-04-28-full-reproducibility-libero.md` | **Reproducibility + LIBERO eval 计划** — 18 tasks, 5 phases |

## Knowhow (Operational Knowledge)

### Runbooks

| File | Description |
|------|-------------|
| `docs/knowhow/runbooks/deploy-to-xdlab23.md` | xdlab23 首次部署 + 日常 sync 流程 |
| `docs/knowhow/runbooks/setup-uv-env-xdlab23.md` | uv venv 替代 conda (非交互 SSH 场景) |
| `docs/knowhow/runbooks/install-libero.md` | LIBERO 安装 — PyPI + assets 部署 (GitHub clone→scp) + 6 个 gotcha |

### Toolchain

| File | Description |
|------|-------------|
| `docs/knowhow/toolchain/cuda-profiling-patterns.md` | CUDA Event vs torch.profiler 对比, warmup 策略, 统计方法 |
| `docs/knowhow/toolchain/hydra-config-patterns.md` | Hydra ListConfig/device gotchas |
| `docs/knowhow/toolchain/shell-script-safety-patterns.md` | Shell 脚本变量 quoting, command injection 防护 |
| `docs/knowhow/toolchain/wam-standalone-profiling.md` | WAM standalone profiling patterns — random-init, sys.path injection, manual timer marks |

### Debug Solutions

| File | Description |
|------|-------------|
| `docs/knowhow/debug-solutions/qwen25vl-model-structure.md` | Qwen2.5-VL 模型层级结构, hook path 参考 |
| `docs/knowhow/debug-solutions/qwen25vl-vision-token-mapping.md` | Vision token ID 定位 (`<\|image_pad\|>` token) |
| `docs/knowhow/debug-solutions/gqa-attention-analysis.md` | GQA Q/K head mismatch, repeat_interleave 方案 |
| `docs/knowhow/debug-solutions/act-action-queue-hooks.md` | ACT select_action() action queue 缓存陷阱 |
| `docs/knowhow/debug-solutions/phasetimer-cpu-backend-bug.md` | PhaseTimer CPU backend record_end() no-op bug |
| `docs/knowhow/debug-solutions/lingbotvla-integration.md` | LingBot-VLA flow VLA 集成 14 个问题汇总 |
| `docs/knowhow/debug-solutions/lingbot-va-wam-integration.md` | LingBot-VA full WAM 集成 7 个陷阱 (构造参数, VAE, timestep, action_mode) |
| `docs/knowhow/debug-solutions/nitrogen-controller-deployment.md` | NitroGen 部署 5 个问题 + Codex 审查发现 |
| `docs/knowhow/debug-solutions/pizero-integration.md` | Pi-Zero 集成 5 个陷阱 — vendor src/ 命名冲突、manual timing vs hooks、opaque infer_action、uv-over-conda、cv2/matplotlib 依赖 |
| `docs/knowhow/debug-solutions/conda-env-model-compat.md` | **Conda env × 模型兼容矩阵** — 哪个 env 跑哪个模型, flash-attn/cuDNN/flax blockers |
| `docs/knowhow/debug-solutions/concurrent-cuda-stream-profiling-pitfalls.md` | 并发 CUDA stream profiling 陷阱 |

### Infrastructure

| File | Description |
|------|-------------|
| `docs/knowhow/runbooks/deploy-new-model-package.md` | GitHub 被防火墙封锁时 sparse clone→tar→scp 部署新模型包 |

### Infrastructure

| File | Description |
|------|-------------|
| `docs/knowhow/infrastructure/xdlab23-model-weights.md` | ModelScope 404 issue, HF cache 路径, 权重下载方案 |

## Papers (Deep Dives)

| File | Description |
|------|-------------|
| `docs/papers/landscape.md` | VLM/VLA inference landscape (early version, superseded by survey/) |
| `docs/papers/starvla-framework-deep-dive.md` | StarVLA 模块化 VLA framework 架构分析 |

## Research Notes

| File | Description |
|------|-------------|
| `notes/survey-plan.md` | 8 周调研行动计划 |
| `notes/sglang-profiling-deep-survey.md` | SGLang profiling 深度调研 — torch.profiler 集成, Prometheus, prefill/decode 分离 |

## Viewer (Visualization)

| File | Description |
|------|-------------|
| `viewer/app.py` | Flask server + static file catch-all |
| `viewer/static/index.html` | Navigation hub (4 card links) |
| `viewer/static/presentation.html` | 5-section advisor meeting slides (hero, motivation, approach, experiments, roadmap) |
| `viewer/static/experiments.html` | Expandable experiment detail tables + prediction calibration |
| `viewer/static/survey-dashboard.html` | Interactive survey dashboard (Pareto, pipeline comparison, maturity matrix) |
| `viewer/static/scaling-curve.html` | DiT scaling scatter (log-log), latency spectrum, NitroGen k-sweep, paradigm comparison |
| `viewer/static/design-space.html` | **Action Model Design Space dashboard** (2026-04-28, Hao meeting) — 7-model paradigm scatter + phase breakdown stacked bar + DiT scaling curve |
| `viewer/static/reproducibility.html` | **Reproducibility Matrix dashboard** (v0.9.0) — latency old vs new grouped bars + LIBERO-4 success heatmap |
| `slides/hao-meeting-2026-04-28.html` | **Hao meeting deck** (11 slides, 四幕叙事 + reproducibility backup) — real-weight data + LIBERO 94.5% |
| `slides/epda-roofline-motivation.html` | **exp08 one-pager** — EPDA roofline 可视化 + 干扰矩阵预测 (advisor meeting figure) |

## Meeting & Planning

| File | Description |
|------|-------------|
| `docs/hao-meeting-prep.md` | 第一次 meeting 大纲 (四幕叙事: Design Space → 四次跳跃 → Fast VLA first → 请教) |
| `docs/meeting-cheatsheet.md` | 面谈前 15 分钟速查 (FastVideo STA/VSA/蒸馏 + DistServe PD disagg + 数据速查) |
| `docs/TODO.md` | Project action items (战略判断置顶 + P0/P1/P2 分层) |
| `docs/learning-plan.md` | GPU/MLSys 补课路径 (L0-L4 分层, Brrrr + CUDA MODE + vLLM + DistServe) |

## Other

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project instructions + index for Claude Code |
| `CHANGELOG.md` | Version history (v0.1.0 - v0.9.0) |
| `.pipeline-state.json` | LabMate pipeline state (current_exp, stage) |
| `.claude/skills/project-skill/SKILL.md` | Project knowledge base (v8, "Fast VLA first") |
| `src/eval/consolidate_matrix.py` | Latency + LIBERO JSON 结果聚合脚本 |
