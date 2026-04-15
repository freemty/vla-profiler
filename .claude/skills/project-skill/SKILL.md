---
name: project-skill
description: "Use when advising on project architecture, experiment history, codebase navigation, or research findings."
user-invocable: false
version: v1
note: "v1 — updated with profiling framework + exp01a results."
updated_at: "2026-04-15"
---

# vlla — Project Knowledge

> VLM/VLA Real-Time Systems Survey & Research
> UCSD PhD 方向调研项目 | 导师: 张昊 (Hao Zhang) — vLLM/FastVideo/Chatbot Arena 作者
> v1 — updated with profiling framework + exp01a results.

---

## 1. Project Overview & Current State

**项目名称:** vlla (VLM/VLA Real-Time Systems)
**核心问题:** 如何让 VLM/VLA 在实时约束下高效运行？
**研究定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

**动机:**
张昊的技术路线: Parameter Server -> Alpa -> vLLM -> FastVideo -> **VLM/VLA real-time systems**。每一步都是 ML Systems 前沿的下一个自然问题。VLM/VLA serving 正处于 "pre-vLLM" 阶段，存在巨大的系统研究空间。

**当前阶段:** Experiment (Phase 1 — VLM Profiling)
- `current_exp`: exp01a (Qwen2.5-VL-7B E/P/D profiling — **done**)
- `stage`: experiment
- Survey 产出: 4 份核心文档，覆盖 180+ 篇论文/项目 (2024-2026)
- Framework 产出: 完整 VLM profiling + attention analysis 框架 (`src/`)，Hydra 配置驱动
- 共享核心: `model-probe-core` git submodule (`src/core/`)，同时被 rope2sink 消费
- 服务器: xdlab23 (8x RTX 5880 Ada 48GB)，已完成部署和首次实验
- **下一步:** 运行 attention analysis (sink detection, visual-text attention patterns)，然后扩展更多 input variants

**exp01a 关键数据 (Qwen2.5-VL-7B on RTX 5880 Ada, per-input):**

| Input Type | Encode (ms) | Prefill (ms) | Decode/tok (ms) |
|-----------|-------------|-------------|-----------------|
| text_only | 0 | ~20 | ~18 |
| single_image | ~253 | ~156 | ~18.6 |
| multi_image (2x) | ~541 | ~332 | ~21 |

**核心发现:** Encode 随 image 数量线性增长; Decode per-token 稳定在 ~18-21ms。

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
|   |   |-- qwen_vl_controller.py   # QwenVLController: model loading, inference, QKV hooks
|   |-- tasks/
|   |   |-- profiling_task.py   # task_epd_profiling: timing aggregation -> JSON
|   |   |-- attention_task.py   # visual_text_attn, sink_detection, per_layer_stats
|   |-- utils/
|       |-- timing.py           # PhaseTimer: CUDA event wrapper (CPU fallback)
|
|-- configs/               # Hydra experiment configs
|   |-- base.yaml
|   |-- qwen_vl_7b/
|       |-- profiling.yaml # E/P/D profiling (3 input variants, 10 benchmark runs)
|       |-- attention.yaml # Attention analysis (layers 0,7,14,21,27; Q+K capture)
|
|-- scripts/               # Server deployment scripts
|   |-- sync_to_remote.sh  # Git bundle sync to xdlab23
|   |-- launch_exp.sh      # GPU-pinned experiment launcher
|   |-- download-results.sh
|
|-- survey/                # 文献综述
|   |-- landscape.md       # VLM/VLA/VA inference 全景图 (~790行)
|   |-- papers/
|       |-- recent-papers.md, va-world-models.md, va-world-models-web.md
|
|-- exp/summary.md         # 实验 flight recorder
|-- docs/
|   |-- superpowers/specs/  # Framework 设计 spec
|   |-- superpowers/plans/  # 实现计划
|   |-- knowhow/runbooks/   # 部署 runbook
```

### 2.2 Framework 继承链

```
probe_core.BaseController          # Model-agnostic: hook lifecycle, StoreMixin, HookMode
  -> BaseVLMController             # VLM-specific: E/P/D phases, PhaseTimer
    -> QwenVLController            # Qwen2.5-VL model-specific
    -> (future) InternVLController
  -> (future) BaseVLAController    # VLA-specific: action generation phases
    -> (future) ACTController
```

**Key Design Decisions:**
- Profiling 和 Analysis 模式严格分离 (tensor copy 干扰 timing)
- Prefill vs Decode 通过 `seq_len > 1` 判断
- PhaseTimer 累加同名 phase 的多次 mark (支持 decode N steps)
- 配置驱动: 新增 input variant 只需改 YAML

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

### 3.3 活跃假设

- [x] ~~Vision encoding 占 >50% prefill 延迟~~ → **exp01a 已验证 (58%)**
- [ ] EPD 三阶段分离实际收益
- [ ] VLM speculative decoding 中 visual token 对 acceptance rate 的影响
- [ ] Qwen2.5-VL attention pattern: sink 分布? (attention analysis 待运行)
- [ ] Per-layer attention entropy 分布

---

## 4. Technical Archive

### 4.1 四大范式对比

| 维度 | VA (1-step flow) | VLA (7B AR) | WAM (DreamZero) | Latent WM |
|------|-----------------|-------------|-----------------|-----------|
| 延迟 | 1-5ms | 100-500ms | ~130ms | 10-15ms |
| 控制频率 | >200Hz | 2-10Hz | ~7Hz | 60-100Hz |
| 泛化能力 | 低 | 高 | 极高 | 中 |

### 4.2 Pareto 前沿

| 方法 | 延迟 | 意义 |
|------|------|------|
| Action-to-Action Flow | 0.56ms | VA 速度下界 |
| Mean-Flow VLA / FASTER | ~50ms | VLA 单步化 |
| DreamZero | ~130ms, 7Hz | WAM zero-shot |
| SAGE | 3.36x speedup | VLM SD 标杆 |
| ID-Selection | 97.2% token reduction | Token pruning 上界 |

### 4.3 技术成熟度 & 迁移矩阵

(See full tables in v0 — unchanged, preserved in archive)

---

## 5. Experiment History

| Exp ID | Date | Status | Key Finding |
|--------|------|--------|-------------|
| exp01a | 2026-04-15 | **done** | E scales linearly with images; D per-token stable ~18-21ms |

### 5.1 Prediction Calibration: exp01a

| Metric | Survey Prediction | Actual | Calibration |
|--------|------------------|--------|-------------|
| Vision encoding 占比 | >50% | 58% | **Accurate** |
| Decode per-token | 2-5ms (A100 literature) | ~18-21ms (RTX 5880) | **Off** — 硬件差异 ~4x |
| Encode 线性扩展 | 线性 (ViT) | 2.14x for 2 images | **Confirmed** |

**Insight:** 文献延迟数据多来自 A100/H100，RTX 5880 约慢 3-5x。结构性判断 (哪个阶段是瓶颈) 正确。

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

---

## 7. Quick Reference

### Commands

| 命令 | 用途 |
|------|------|
| `bash scripts/sync_to_remote.sh` | Sync to xdlab23 |
| `bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling` | Profiling GPU 0 |
| `bash scripts/launch_exp.sh 1 qwen_vl_7b/attention` | Attention GPU 1 |
| `bash scripts/download-results.sh` | Download results |

### Server

| Item | Value |
|------|-------|
| SSH | `ssh xdlab23_yang` (port 66) |
| Path | `/data1/ybyang/vlla` |
| Conda | `vit-probe` |
| GPUs | 8x RTX 5880 Ada 48GB |

### Registries

**Controllers:** `qwen_vl` → QwenVLController
**Tasks:** `epd_profiling`, `visual_text_attention`, `sink_detection`, `per_layer_stats`

---

*v1 — updated with framework + exp01a results. Updated: 2026-04-15*
