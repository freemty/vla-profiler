# VLLA — VLM/VLA Real-Time Systems Survey & Research

## Project Overview

PhD 研究方向调研项目：Vision-Language Model (VLM) / Vision-Language-Action (VLA) 实时推理系统。
导师：张昊 (Hao Zhang), UCSD — vLLM/FastVideo/Chatbot Arena 作者。

**核心问题**: 如何让 VLM/VLA 在实时约束下高效运行？
- 机器人控制: ~10ms 级 control loop
- 自动驾驶: 实时感知-决策管线
- 交互式 AI: 实时视觉对话、AR/VR
- 边缘部署: 设备端算力/功耗约束

## Research Context

张昊的技术路线: Parameter Server → Alpa → vLLM → FastVideo → **VLM/VLA real-time systems**
每一步都是 ML Systems 前沿的下一个自然问题。

## Directory Structure

```
src/                   # Profiling & analysis framework
  core/                # Git submodule → model-probe-core (shared with rope2sink)
  controllers/         # BaseVLMController, QwenVLController
  tasks/               # profiling_task, attention_task
  utils/               # PhaseTimer
  run_tasks.py         # Hydra entry point
configs/               # Hydra experiment configs
  base.yaml            # Shared defaults
  qwen_vl_7b/          # Qwen2.5-VL-7B specific configs
survey/                # Survey 文献综述 (核心产出)
  landscape.md         # VLM/VLA inference 全景图
  papers/              # 按主题分类的论文调研
exp/                   # 实验目录 (LabMate 管理)
  summary.md           # 实验 flight recorder
notes/                 # 研究笔记和想法
docs/                  # 文档
  knowhow/             # 基础设施/工具链/调试笔记
  specs/               # 实验 spec + framework design
  weekly/              # 周报
scripts/               # 实验脚本 (launch, sync, download)
viewer/                # 实验结果可视化 (Flask)
slides/                # 演示文稿
```

## Key References

- selfOS wiki: `~/selfOS/wiki/concepts/vlm-vla-real-time-systems.md`
- CSE 234 课程: `~/selfOS/wiki/sources/cse234-w25-data-systems-for-ml.md`
- Hao AI Lab 蒸馏: `~/selfOS/wiki/sources/notion-2026-04-02-hao-ai-lab-ucsd-phd-学生方向与工作蒸馏.md`
- VLA Models (Pi-Zero): `~/selfOS/wiki/sources/gem-2026-01-05-vla-models-pi-zero-and-competitors.md`
- **2025-2026 最新论文综述**: `survey/papers/recent-papers.md` — VLM/VLA inference efficiency 领域 40+ 篇论文/项目
- **VA + World Action Model 深度 Survey**: `survey/papers/va-world-models.md` — VA 五大架构族谱、World Model 五层分类 (L0-L4)、WAM efficiency 分析、ERA disaggregation 概念、speculative rollout
- **WAM/Video WM/VA 最新论文 Web 调研**: `survey/papers/va-world-models-web.md` — 2025-2026 年 80+ 篇论文，覆盖 WAM、Video World Model、单步推理加速、WM+VLA 融合、NVIDIA Cosmos/Pi0 工业进展
- **Framework design spec**: `docs/superpowers/specs/2026-04-14-vlm-profiling-framework-design.md`
- **Implementation plan**: `docs/superpowers/plans/2026-04-14-vlm-profiling-framework.md`
- **Shared core (model-probe-core)**: `src/core/` — submodule, also used by rope2sink
- **SGLang profiling 深度调研**: `notes/sglang-profiling-deep-survey.md` — SGLang 的 torch.profiler 集成、Prometheus metrics、benchmark 套件、prefill/decode 分离 profiling、内存追踪的实现级分析
- **ML Inference Profiling Systems 横向调研**: `survey/papers/profiling-systems-survey.md` — FastVideo/TensorRT-LLM/DeepSpeed/Triton/vLLM/SGLang/llama.cpp/MLC LLM 8大系统的 profiling 实现对比，timing 机制、phase 定义、warmup 策略、统计方法、memory tracking 全面分析
- **ML Profiling 综合调研报告**: `survey/papers/ml-profiling-systems-comprehensive-survey.md` — 4 agent 并行调研的综合报告，含 PhaseTimer 代码审查、CUDA timing 最佳实践、三层 profiling 架构建议

## Survey Dimensions

1. **Model Architecture**: VLM (视觉语言) / VLA (视觉语言动作) / VA (纯视觉动作) 的架构分类
2. **Inference Efficiency**: Serving, scheduling, KV-cache, speculative decoding, quantization
3. **Real-Time Constraints**: Latency requirements per application domain
4. **Hardware Adaptation**: GPU/TPU/Edge deployment strategies
5. **Open Problems**: Where the field is heading

## Quick commands

| Command | Purpose |
|---------|---------|
| `bash scripts/sync_to_remote.sh` | Sync code to xdlab23 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling` | Run profiling on GPU 0 |
| `bash scripts/launch_exp.sh 1 qwen_vl_7b/attention` | Run attention analysis on GPU 1 |
| `bash scripts/download-results.sh` | Download results from server |
| /labmate:new-experiment | Scaffold new experiment |
| /labmate:analyze-experiment | Analyze results |

## Server (xdlab23)

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` (shared with rope2sink) |
| GPUs | 8x RTX 5880 Ada 48GB |
| Model cache | `/data1/ybyang/huggingface` |
| Code sync | Git bundle (GitHub blocked by firewall) |
| Runbook | `docs/knowhow/runbooks/deploy-to-xdlab23.md` |

## Session startup

| What to do | Read first |
|-----------|-----------|
| Catch up on progress | .claude/skills/project-skill/SKILL.md |
| Check domain literature | survey/landscape.md |
| Run current experiment | exp/{current_exp}/README.md |

## Project knowledge

- **Skill hub:** .claude/skills/project-skill/SKILL.md
- **Experiment log:** exp/summary.md
- **Domain papers:** survey/papers/

## Knowhow

- `docs/knowhow/infrastructure/` — Servers, networking, disk, GPU issues
- `docs/knowhow/toolchain/` — CLI tools, docker, conda/pip, framework tips
- `docs/knowhow/debug-solutions/` — Error investigation paths and fixes
- `docs/knowhow/runbooks/` — Step-by-step operational procedures

## Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| @project-advisor | opus | Experiment history, findings, codebase navigation |
| @cc-advisor | sonnet | Claude Code workflow best practices |
| @domain-expert | opus | Reads papers, interprets experiment results |
| @exp-manager | sonnet | Monitors experiments, diagnoses failures |
| @slides-maker | sonnet | Generates HTML slides from analysis |
| @viz-frontend | sonnet | Builds analysis dashboards |

## Skills

All plugin skills use the `labmate:` prefix.

| Skill | Trigger |
|-------|---------|
| /labmate:new-experiment | Starting a new experiment |
| /labmate:analyze-experiment | After experiment completes |
| /labmate:update-project-skill | After major findings or when stale |
| /labmate:present-template | Generate overview slides |
| /labmate:weekly-progress | Summarize week's progress |
| /labmate:commit-changelog | Commit with CHANGELOG |

## Workflow

```
/labmate:new-experiment → run → /labmate:analyze-experiment
  → commit findings → /labmate:update-project-skill → repeat
```

Pipeline state tracked in .pipeline-state.json.

## Research principles

1. **Measure first** — attack the actual bottleneck, not your intuition
2. **Baselines are sacred** — every claim needs a reproducible baseline comparison
3. **Statistical rigor** — single-run results are anecdotal, track variance
4. **Ablation-driven** — multi-factor changes require per-factor isolation
5. **Respect negative results** — don't retry failed directions without new evidence
6. **Predict first** — record expected numbers before running, calibrate after

## Conventions

- **Exp naming:** exp{NN}{x} — number=major direction, letter=variant
- **Prompt versioning:** prompts/{component}/_v{NN}.md — never overwrite, always increment
- **CHANGELOG rule:** all iterating artifacts (prompts, skills, agents) must have CHANGELOG entries
- **Worktree rule:** destructive or exploratory changes use git worktree

## Current state

- **current_exp:** exp01b (Qwen2.5-VL-7B attention analysis — done)
- **stage:** experiment
- **skill_updated_at:** 2026-04-15
- **key findings:**
  - **exp01a (profiling):** text P=20ms/D=18ms; single_img E=253ms/P=156ms/D=18.6ms; multi_img E=541ms/P=332ms/D=21ms. Encode scales linearly.
  - **exp01b (attention):** Pos 2 (first visual patch) is universal attention sink (12K-18K received, 12-28x vs #2). Text→Visual Gini >0.91 (extreme sparsity → token pruning viable). Layer 21 entropy lowest (3.44).
  - **exp02a (ACT):** Total ~3ms (850x faster than VLM). Encode 80%, action 20%. VLA latency lower bound.
- **latest (v0.4.0):** Attention Overlay Visualization, Interpretability Mixin, GQA fix, BaseVLAController, ACTController.
- **next:** Run attention overlay on server. OpenVLA (need HF download). Pi-Zero controller. Gradient saliency.
