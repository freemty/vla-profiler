---
name: project-skill
description: "Use when advising on project architecture, experiment history, codebase navigation, or research findings."
user-invocable: false
version: v0
note: "v0 — auto-generated bootstrap, review recommended."
updated_at: "2026-04-11"
---

# vlla — Project Knowledge

> VLM/VLA Real-Time Systems Survey & Research
> UCSD PhD 方向调研项目 | 导师: 张昊 (Hao Zhang) — vLLM/FastVideo/Chatbot Arena 作者
> v0 — auto-generated bootstrap, review recommended.

---

## 1. Project Overview & Current State

**项目名称:** vlla (VLM/VLA Real-Time Systems)
**核心问题:** 如何让 VLM/VLA 在实时约束下高效运行？
**研究定位:** 从 ML Systems 视角审视 VLM/VLA inference efficiency 的技术前沿与开放问题

**动机:**
张昊的技术路线: Parameter Server -> Alpa -> vLLM -> FastVideo -> **VLM/VLA real-time systems**。每一步都是 ML Systems 前沿的下一个自然问题。VLM/VLA serving 正处于 "pre-vLLM" 阶段，存在巨大的系统研究空间。

**当前阶段:** Survey 完成 (Phase 0)，尚未进入实验阶段
- `current_exp`: null
- `stage`: dev
- Survey 产出: 4 份核心文档，覆盖 180+ 篇论文/项目 (2024-2026)
- 下一步: Phase 1 实操验证 (vLLM/SGLang profile VLM latency breakdown)

**应用场景与延迟 Gap:**

| 场景 | 控制频率需求 | 当前模型延迟 | Gap |
|------|------------|------------|-----|
| 工业机械臂 | 100-1000 Hz (1-10ms) | 50-500ms | 10-100x |
| 灵巧手操作 | 50-100 Hz (10-20ms) | 100-300ms | 5-30x |
| 自动驾驶决策 | 10-20 Hz (50-100ms) | 200-1000ms | 2-20x |
| 人形机器人全身 | 50-200 Hz (5-20ms) | 200-1000ms | 10-200x |

---

## 2. Architecture

### 2.1 目录结构

```
vlla/
|-- CLAUDE.md              # 项目入口 + 索引
|-- CHANGELOG.md           # 版本日志
|-- .pipeline-state.json   # LabMate 流水线状态
|
|-- survey/                # 核心产出: 文献综述
|   |-- landscape.md       # VLM/VLA/VA inference 全景图 (主文档, ~790行)
|   |-- papers/
|       |-- recent-papers.md       # 40+ 篇最新论文 web 搜索 (~540行)
|       |-- va-world-models.md     # VA + WAM 深度分析 (~970行)
|       |-- va-world-models-web.md # WAM/Video WM/VA 最新论文 (~560行)
|
|-- notes/
|   |-- survey-plan.md     # 8 周调研行动计划
|
|-- exp/                   # 实验目录 (空，待启动)
|   |-- summary.md         # 实验 flight recorder (空表格)
|
|-- docs/                  # 文档 (骨架已建)
|   |-- papers/landscape.md  # (旧路径，主文档在 survey/)
|   |-- specs/, weekly/, archive/
|   |-- knowhow/           # infrastructure, toolchain, debug-solutions, runbooks
|
|-- scripts/               # 实验脚本
|   |-- launch_exp.py, monitor_exp.sh, download_results.sh
|
|-- viewer/                # Flask 可视化
|   |-- app.py
|   |-- static/index.html, survey-dashboard.html
|
|-- slides/                # 演示文稿 (空)
|
|-- .claude/
    |-- skills/project-skill/SKILL.md  # 本文件
    |-- agent-memory/labmate-domain-expert/  # domain expert 记忆
```

### 2.2 Survey 文档之间的数据流

```
landscape.md (主干)
  |-- Section 1: VLM Inference       <-- recent-papers.md Section 1-3 补充
  |-- Section 2: VLA Inference       <-- recent-papers.md Section 4-5 补充
  |-- Section 3: VA & WAM            <-- va-world-models.md 全量补充
  |-- Section 4: Gap Analysis            va-world-models-web.md 全量补充
  |-- Section 5: Taxonomy
  |-- Section 6: Research Entry Points
                    |
                    v
             survey-plan.md (行动计划)
```

---

## 3. System Cognition

### 3.1 三大战略判断 (Survey 核心结论)

**判断 1: VLM serving 正处于 "pre-vLLM" 阶段**
2024-2026 年的 VLM serving 工作 (EPD disaggregation, modality-aware scheduling) 类似于 2022-2023 年的 LLM serving 研究。一个类似 vLLM 量级的 VLM serving system 即将出现 -- 张昊实验室有最好的位置来做这件事。

**判断 2: VLA inference 正处于 "wild west" 阶段**
没有专门的 VLA serving system，没有统一的 benchmark，没有成熟的优化 pipeline。从 autoregressive VLA 的 speculative decoding 到 flow VLA 的单步推理，各种方向并行探索。PhD 入场的黄金时期。

**判断 3: Algorithm-System Co-design 是最大的空白**
绝大多数 token pruning、quantization、speculative decoding 的工作都是 model-level 的，与 serving system 的 scheduling、memory management、batching 策略没有深度整合。这个 co-design 的空间是实验室最擅长的领域。

### 3.2 已验证的假设

- **Visual token 冗余度极高:** 97.2% 的 visual tokens 可以在不严重损失精度的情况下被剪枝 (ID-Selection, 2026)
- **Text-only speculative decoding 在 VLM 上退化:** MMSpec (2026) 实验证明
- **Flow/Diffusion VLA 可压缩至单步:** Action-to-Action Flow 实现 0.56ms 延迟 (2026)
- **VA inference 已接近物理极限:** 1-step flow VA 的 action generation (~1ms) 远快于 vision encoding (~10ms)
- **WAM 可实现 zero-shot policy:** DreamZero (2026) 14B 参数，7Hz 控制

### 3.3 活跃假设 (待实操验证)

- [ ] VLM serving 中 vision encoding 占总延迟的百分比 (预期 >50% on prefill)
- [ ] EPD 三阶段分离的实际收益 vs 理论收益
- [ ] VLM speculative decoding 中 visual token 对 acceptance rate 的影响
- [ ] VLA action token 生成的 bottleneck 位置
- [ ] Flow VLA 单步化后 remaining bottleneck 是 vision encoding 还是 flow head
- [ ] Token pruning 方法集成到 serving system 的工程可行性

### 3.4 八个研究入口 (Research Entry Points)

**Tier 1 (最推荐 -- quick win + high impact):**
1. **VLM-Aware Serving System** -- "VLM 的 vLLM"，EPD 调度 + modality-aware paging + image prefix caching
2. **VLA Real-Time Serving** -- 蓝海领域，action streaming + speculative action generation

**Tier 2 (中期 -- 深度 co-design):**
3. **Cross-Modal Token Economy** -- token value 驱动 system-level pruning/eviction/quantization
4. **FastVideo -> VLA 技术迁移** -- STA/VSA 用于 action denoising + 蒸馏框架
5. **WAM Serving System** -- ERA disaggregation + stateful session + speculative rollout

**Tier 3 (长期):**
6. **异构 VLM/VLA 推理** -- Cloud-Edge split inference
7. **统一 VA/VLA/WAM Serving Router** -- 按任务复杂度动态路由
8. **Speculative Rollout for World Models** -- 全新 systems 概念

---

## 4. Technical Archive

### 4.1 四大范式对比 (VA / VLA / WAM / Latent WM)

| 维度 | VA (1-step flow) | VLA (7B AR) | WAM (DreamZero) | Latent WM (Dreamer) |
|------|-----------------|-------------|-----------------|---------------------|
| 延迟 | 1-5ms | 100-500ms | ~130ms | 10-15ms |
| 控制频率 | >200Hz | 2-10Hz | ~7Hz | 60-100Hz |
| 泛化能力 | 低 (窄领域) | 高 (语言条件) | 极高 (zero-shot) | 中 (需 online RL) |
| 数据需求 | 高 (需 robot demos) | 中 (利用预训练) | 低 (利用 video 预训练) | 中 (在线学习) |
| 模型参数 | 10M-500M | 7B-70B | 100M-14B | 10M-500M |
| GPU 需求 | 单 GPU | 多 GPU | 多 GPU | 单 GPU |
| KV-cache 需求 | 不需要 | 核心瓶颈 | 不需要/轻量 | 不需要 |
| Language 能力 | 无 | 强 | 有限 | 无 |

### 4.2 Pareto 前沿关键数据点

| 模型/方法 | 延迟 | 效果 | 意义 |
|-----------|------|------|------|
| Action-to-Action Flow | **0.56ms** | 1-step VA | VA 速度下界 |
| DDP-WM | 20-40ms | 9x speedup WM | Efficient WM 标杆 |
| Mean-Flow VLA | ~50ms | 83.9x vs baseline | VLA 单步化 |
| FASTER | ~50ms | 1-step flow VLA | VLA 单步化 |
| DreamZero | ~130ms, 7Hz | zero-shot policy | WAM 能力上界 |
| SAGE | 3.36x speedup | VLM spec decoding | VLM SD 标杆 |
| Fast-dVLM | 6x speedup | block-diffusion VLM | 非 AR VLM |
| ID-Selection | 97.2% token reduction | training-free | Token pruning 上界 |
| HybridKV | 7.9x KV 压缩 | visual KV 差异化 | KV-cache 标杆 |

### 4.3 技术成熟度矩阵

| 技术 | LLM | VLM | VLA | 关键 Gap |
|------|-----|-----|-----|---------|
| FlashAttention | 生产级 | 高 | 中 | Cross-modal pattern 未优化 |
| PagedAttention | 生产级 | 中 | 低 | Visual KV 无差异化管理 |
| Continuous Batching | 生产级 | 中 | 低 | Modality 差异导致 padding 浪费 |
| PD Disaggregation | 生产级 | 中 | 低 | 需扩展到 EPD |
| Speculative Decoding | 成熟 | 起步 (2025) | 萌芽 (2025-26) | VLM/VLA specific 挑战 |
| Token Pruning | N/A | 活跃 (2024+) | 萌芽 (2026) | 与 serving system 整合不足 |
| Quantization INT8/4 | 成熟 | 中 | 起步 | VLA 1-bit 刚开始探索 |
| Flow 单步推理 | N/A | N/A | 活跃 (2025+) | 质量-速度 trade-off |

### 4.4 关键技术演进时间线

```
2023: RT-2 (首个大规模 VLA) | Diffusion Policy (100步) | vLLM | ACT
2024: OpenVLA (7B) | Pi-Zero (Flow) | DistServe (PD分离) | FastV (首批token pruning)
2025 H1: FlashVLM | 首批VLM SD (ViSpec/Spec-LLaVA) | EPD概念 | DM1/ManiFlow (单步flow)
2025 H2: HiViS/DREAM | RServe | HydraInfer | xLLM
2026 Q1: SAGE/Fast-dVLM (VLM SD) | MMSpec (证明text-only SD退化) | FASTER/Mean-Flow VLA (单步)
         HeiSD/KERV (VLA SD) | RPS-Serve/EPD-Serve | DreamZero (WAM) | 0.56ms VA
```

### 4.5 vLLM/FastVideo 技术迁移矩阵

| 实验室技术 | VLM 应用 | VLA 应用 | WAM 应用 |
|-----------|---------|---------|---------|
| PagedAttention | Visual KV 差异化管理 | - | Latent state paging |
| Continuous Batching | Modality-aware scheduling | Multi-robot action batching | Multi-robot rollout batching |
| PD Disaggregation | EPD 三阶段分离 | Encode-Infer-Action | ERA disaggregation |
| Speculative Decoding | Vision-aware SD | Physics-aware SD (KERV) | Speculative rollout (新概念) |
| Prefix Caching | Image prefix caching | - | 环境 state 缓存 |
| STA (FastVideo) | Vision encoder 加速 | Action denoising temporal attention | Video prediction spatial attention |
| VSA (FastVideo) | Video VLM temporal attention | - | Video prediction temporal attention |
| Step-parallel (FastVideo) | - | Flow VLA 并行去噪 | WAM 并行 rollout |
| Model Distillation (FastVideo) | - | Multi-step -> 1-step flow VLA | DreamZero 7Hz -> 40Hz+ |

---

## 5. Experiment History Table

| Exp ID | Motivation | Status | Key Finding |
|--------|-----------|--------|-------------|
| (none) | - | - | - |

**计划中的首批实验 (来自 survey-plan.md Phase 1-2):**

| 计划 | 目标 | 优先级 |
|------|------|--------|
| Profile VLM on vLLM | Qwen2.5-VL-7B latency breakdown (vision/prefill/decode) | P0 |
| Profile VLM on SGLang | 对比 vLLM 的 breakdown | P0 |
| Deploy OpenVLA | Profile VLA inference latency | P1 |
| Try VLAgents | 首个 VLA serving framework 体验 | P1 |
| SIMPLER env test | VLA latency-success trade-off 评估 | P2 |

---

## 6. Engineering Lessons (APPEND-ONLY)

> 此节记录项目执行过程中的工程经验教训，仅追加不删除。

### 2026-04-11: 初始 survey 过程

1. **先全景后深入的调研策略有效:** landscape.md 先覆盖全局，然后 va-world-models.md 和 va-world-models-web.md 做纵深。这种结构让新 session 可以按需加载 -- 快速了解读 landscape，深入某方向读对应子文档。

2. **论文数量统计:**
   - landscape.md: 主干综述，覆盖 ~80 篇论文/方法，含完整 taxonomy 和 gap analysis
   - recent-papers.md: 40+ 篇 web 搜索补充
   - va-world-models.md: 60+ 篇 VA/WAM 深度分析
   - va-world-models-web.md: 80+ 篇最新论文补充
   - 总计: **180+ 篇** 去重后的论文/项目覆盖

3. **Survey 文档的大小需要控制:** 单个文件接近 1000 行时，context window 消耗显著。未来应更积极地拆分子文档。

---

## 7. Quick Reference

### 7.1 关键路径

| 文件 | 用途 |
|------|------|
| `CLAUDE.md` | 项目入口，索引所有文档 |
| `survey/landscape.md` | Survey 主干 (~790行) |
| `survey/papers/recent-papers.md` | 最新论文搜索 (~540行) |
| `survey/papers/va-world-models.md` | VA/WAM 深度分析 (~970行) |
| `survey/papers/va-world-models-web.md` | WAM/Video WM 补充 (~560行) |
| `notes/survey-plan.md` | 8 周行动计划 |
| `exp/summary.md` | 实验 flight recorder |
| `.pipeline-state.json` | LabMate 状态 |

### 7.2 常用命令

| 命令 | 用途 |
|------|------|
| `/labmate:new-experiment` | 创建新实验 |
| `/labmate:analyze-experiment` | 分析实验结果 |
| `/labmate:update-project-skill` | 刷新本文件 |
| `/labmate:weekly-progress` | 生成周报 |
| `python scripts/launch_exp.py --exp <id>` | 启动实验 |

### 7.3 8 周行动计划摘要

| 阶段 | 时间 | 内容 |
|------|------|------|
| Phase 0 | Week 1-2 | 读 3 篇必读综述 + 实验室 foundational papers |
| Phase 1 | Week 2-4 | VLM Serving 深度调研 + vLLM/SGLang profile 实操 |
| Phase 2 | Week 4-6 | VLA Inference 深度调研 + OpenVLA/VLAgents 实操 |
| Phase 3 | Week 6-8 | Gap Analysis + Research Proposal 写作 (与张昊讨论) |

### 7.4 Git 里程碑

```
b26f13d fix: address code review -- dir structure, gitignore, docs
6cf329b feat: add VA/WAM web survey (80+ papers)
0906724 feat: init vlla project with VLM/VLA/WAM real-time systems survey
```

### 7.5 Agent 速查

| Agent | 用途 | 何时使用 |
|-------|------|---------|
| @project-advisor (opus) | 实验历史、代码导航 | 项目规划 |
| @domain-expert (opus) | 论文解读、实验结果分析 | 文献讨论 |
| @exp-manager (sonnet) | 实验监控、故障诊断 | 实验运行中 |
| @slides-maker (sonnet) | HTML 幻灯片 | 汇报 |

---

*v0 -- auto-generated bootstrap, review recommended. Updated: 2026-04-11*
