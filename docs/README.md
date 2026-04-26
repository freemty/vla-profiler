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

## Experiment Results

| Dir | Model | Status | Key Metric |
|-----|-------|--------|------------|
| `exp/exp01a/` | Qwen2.5-VL-7B (profiling) | Done | E=253ms, D=18-21ms/tok |
| `exp/exp01b/` | Qwen2.5-VL-7B (attention) | Done | Pos 2 sink 12-28x, Gini >0.91 |
| `exp/exp02a/` | ACT (profiling) | Done | Total ~3ms, 850x vs VLM |
| `exp/exp03a/` | LingBot-VLA-4B (profiling) | Done | 74.5ms, 13Hz |
| `exp/exp04a/` | Fast-WAM (profiling) | Done | 407ms@10step, 2.5Hz |
| `exp/exp04b/` | LingBot-VA (profiling) | Done | 2091ms, 0.5Hz |
| `exp/exp05a/` | LingBot-VLA-4B (attention) | Done | VLA reshapes attention: Gini 0.91→0.07 |
| `exp/exp05b/` | Qwen2.5-VL-3B (attention) | Done | Disambiguation: Gini collapse = VLA fine-tuning |
| `exp/exp06a/` | NitroGen 500M DiT (profiling) | Done | 7.2ms/step, linear, k=1: 55.9Hz |
| `exp/exp07a/` | Pi-Zero dual-stream (profiling) | Done | E=9-12ms/C=26-33ms/A=165-205ms, ~201ms (~5Hz) |
| `exp/summary.md` | Flight recorder | — | All experiments, one row each |

## Design Specs & Plans

| File | Description |
|------|-------------|
| `docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md` | Profiling framework 设计 spec — BaseController 继承链, PhaseTimer, hook 架构 |
| `docs/superpowers/specs/2026-04-15-attention-overlay-visualization-design.md` | Attention overlay 可视化设计 — Mixin 体系, TokenSpatialMap, OverlayRenderer |
| `docs/superpowers/plans/2026-04-14-vlm-profiling-framework.md` | Framework 实施计划 — 12 tasks, 44 steps |
| `docs/superpowers/plans/2026-04-15-attention-overlay-visualization.md` | Attention overlay 实施计划 |
| `docs/specs/2026-04-26-epda-disaggregation-spec.md` | **exp08 spec** — EPDA 四阶段干扰量化 (L1→L2 跨越, DistServe-style motivation figure) |

## Knowhow (Operational Knowledge)

### Runbooks

| File | Description |
|------|-------------|
| `docs/knowhow/runbooks/deploy-to-xdlab23.md` | xdlab23 首次部署 + 日常 sync 流程 |
| `docs/knowhow/runbooks/setup-uv-env-xdlab23.md` | uv venv 替代 conda (非交互 SSH 场景) |

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

## Other

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project instructions + index for Claude Code |
| `CHANGELOG.md` | Version history (v0.1.0 - v0.7.0) |
| `.pipeline-state.json` | LabMate pipeline state (current_exp, stage) |
| `.claude/skills/project-skill/SKILL.md` | Project knowledge base (v6) |
