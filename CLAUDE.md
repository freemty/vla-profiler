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
- **NitroGen 精读**: `survey/papers/nitrogen-deep-dive.md` — NVIDIA 500M VA gaming foundation model, DiT+flow matching, 40K小时跨游戏训练, profiling 价值分析
- **DreamDojo + DreamZero 精读**: `survey/papers/dreamdojo-dreamzero-deep-dive.md` — NVIDIA GEAR Lab 双论文: DreamDojo (44k hr human video WM, LAM, 蒸馏 10.81FPS) + DreamZero (WAM zero-shot policy, Wan2.1 14B, DiT caching, 7Hz on GB200). 含延迟对比 exp04a/04b
- **Documentation index**: `docs/README.md` — 全项目文档结构索引 (survey, experiments, knowhow, specs, notes, viewer)

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
- **TODO:** `docs/TODO.md` — Project action items and task backlog

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

- **current_exp:** exp06a (NitroGen 500M DiT profiling — done)
- **stage:** experiment
- **skill_updated_at:** 2026-04-22
- **key findings:**
  - **exp01a (profiling):** text P=20ms/D=18ms; single_img E=253ms/P=156ms/D=18.6ms; multi_img E=541ms/P=332ms/D=21ms. Encode scales linearly.
  - **exp01b (attention):** Pos 2 (first visual patch) is universal attention sink (12K-18K received, 12-28x vs #2). Text→Visual Gini >0.91 (extreme sparsity → token pruning viable). Layer 21 entropy lowest (3.44).
  - **exp02a (ACT):** Total ~3ms (850x faster than VLM). Encode 80%, action 20%. VLA latency lower bound.
  - **exp03a (LingBot-VLA-4B):** single_img E=35.7ms/C=38.3ms/A=0.48ms (total 74.5ms). 3B backbone 比 7B 快 7x。Context ≈ Encode。Flow action head 0.48ms ≈ ACT。Total 74.5ms ≈ 13Hz。
  - **exp04a (Fast-WAM):** @10step: E=7.6ms/C=36.7ms/A=362ms (total 407ms, 2.5Hz). Action dominates 89%. Per-step ~32ms (30L MoT cross-attn).
  - **exp04b (LingBot-VA):** E=75.5ms/V=592.5ms/A=1423ms (total 2091ms, 0.5Hz). Full WAM 5x slower than skip-imagination. Action 68%.
  - **exp05a (LingBot-VLA attention):** VLA fine-tuning reshapes attention: Gini 0.91→0.07, sink Pos2→Pos64, entropy flat. VLM pruning 不可迁移到 VLA。
  - **exp05b (Qwen2.5-VL-3B attention):** 消歧: Gini 崩塌归因于 VLA fine-tuning (非 model size)。3B vanilla Gini 0.80-0.98。
  - **exp06a (NitroGen 500M DiT):** Per-step 7.2ms (174M DiT), perfectly linear. 174M→350M: 2x params, 4.4x latency → compute-bound 到 memory-BW-bound 转换。k=1: 55.9Hz。
- **latest (v0.5.1):** WAM profiling + VLA attention analysis + NitroGen DiT profiling, NitroGenController added.
- **next:** DreamZero profiling on RTX 5880 Ada. DreamZero DiT layer activation variance 分析. OpenVLA (need HF download). Pi-Zero controller.
