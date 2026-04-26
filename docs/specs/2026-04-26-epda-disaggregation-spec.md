# exp08 Spec — EPDA 四阶段干扰量化

> **方向**: 从 L1 (profile) 跳到 L2 (结构性论证)，对标 DistServe Figure 1 + FastVideo Figure 1。
> **日期**: 2026-04-26
> **前置文档**: `survey/papers/hao-style-synthesis.md`（候选 D）；`survey/papers/hao-style-distserve.md` §5
> **命名**: 这是 **exp08 系列**（新 major direction），exp07 已被 Pi-Zero profiling 占用

---

## 1. Motivation（一段话）

DistServe 在 LLM 上通过 PD disaggregation 拿到 7.4x goodput，核心是"prefill compute-bound vs decode memory-BW-bound 结构性干扰不可调和"。VLM 加入 Vision Encode (E) → 三阶段；WAM 类 VLA 再加 Action Denoising (A) → 四阶段 EPDA。**A 阶段 (Fast-WAM 89%, LingBot-VA 68%) 是 LLM 域从未出现过的新瓶颈**，且与 E/P/D 计算图完全独立，但被串行执行。

**待证明的结构性论证**（L2 命题）：
> EPDA 四个阶段在同一 GPU 上共置时存在**不可调和的资源干扰**——四者的 compute/memory-BW/HBM 占用特征两两异构，任何共置方案都是对四者的妥协。

证明这个命题本身就是一篇 systems workshop paper / short paper 的 contribution，不需要实现 disaggregation。

---

## 2. 与现有工作的差异化

| 工作 | 覆盖阶段 | 状态 |
|------|---------|------|
| DistServe (OSDI'24) | P + D (LLM) | 已做 |
| EPD Disaggregation (arXiv:2501.05460, ICML'25) | E + P + D (VLM) | 已做 |
| **exp08 (this spec)** | **E + P + D + A (VLA WAM)** | 空白 |

关键差异化点：
1. **A 阶段是 iterative compute-bound**——与 E（single-shot compute）、P（attention-heavy compute）、D（memory-BW-bound）的计算特征第四种类型
2. A 阶段的输入是 D 阶段的 latent，而 D 要等 E+P 完成——四阶段 pipeline 的 critical path 更长
3. A 阶段有内部 step loop（10-50 步 DiT denoising），与外部 pipeline 形成嵌套调度问题

---

## 3. 实验矩阵

**三个子实验，按工作量排序**：

### exp08a: EPDA 干扰量化（Motivation Figure 1）
- **目标**: 单 GPU 上并发运行 E/P/D/A 不同组合，测量彼此干扰导致的延迟膨胀
- **方法**:
  - Baseline: 每个阶段单独跑（isolated latency）
  - 干扰实验: 两两共置（E+P, E+D, E+A, P+D, P+A, D+A）并发跑，测膨胀比
  - 扩展: 三阶段共置 (EPD, EPA, EDA, PDA) 和全部共置 (EPDA)
- **度量**: 每个阶段的延迟膨胀倍数、SM 利用率、HBM 带宽利用率、DRAM 占用
- **工具**: `nvtx` range markers + `nvidia-smi dmon` + `torch.profiler` + DCGM（若可用）
- **模型组合**: Qwen2.5-VL-7B (E+P+D) + Fast-WAM Action DiT 350M (A)
  - 原因: 两个模型的阶段最典型、数据已有 baseline (exp01a + exp04a)
- **预期产出**: 一张 6×6 的干扰矩阵图（类似 DistServe Figure 1），标注每对的干扰量级

### exp08b: SLO 违反分析（Motivation Figure 2）
- **目标**: 展示 co-located baseline 在实时控制 SLO 下的违反率
- **SLO 定义（VLA 场景）**:
  - TTFA (Time To First Action): < 200ms（类比 TTFT）
  - Control frequency: > 10Hz（类比 TPOT 的倒数）
  - End-to-end p99: < 500ms
- **方法**: 用 vLLM 或自写 serving wrapper，并发 2/4/8/16 请求，测 SLO 违反率
- **预期产出**: "SLO violation rate vs concurrency" 曲线，对标 DistServe Figure 2

### exp08c: A 阶段的嵌套调度 opportunity（Motivation Figure 3）
- **目标**: 测量 A 阶段 step-level 的跨请求 overlap 潜力
- **假设**: 不同请求的 A 阶段 step n 可以在不同 GPU 上互相填补 bubble
- **方法**: 追踪 exp06a (NitroGen) 和 exp04a (Fast-WAM) 的 step-level 时间线，计算理论 overlap ratio
- **预期产出**: "如果 A 阶段独立调度，goodput 可提升 X%" 的上界估计

---

## 4. 产出物

### exp08 必须产出（对齐 Hao-style 模板）
| 产出 | 目的 | 对应方法论 Step |
|------|------|---------------|
| 干扰矩阵图（6×6 或 EPDA 四阶段两两） | Motivation figure | Step 1 (profile) + Step 2 (结构性论证) |
| SLO 违反率曲线 | Co-located baseline 坏在哪 | Step 2 (结构性论证) |
| "EPDA-goodput" 指标定义 | 新指标 | Step 5 (指标 reframe) |
| Position paper / workshop paper draft | 向 Hao 展示方向 | — |

### exp08 **不**做的事
- ❌ 实现 disaggregation scheduler（是 L4+L5 的工作，单独立项）
- ❌ 跨 GPU KV/latent transfer 机制（同上）
- ❌ Placement algorithm（同上）

**Scope 纪律**：exp08 只做 motivation，不做解法。解法留给 exp09+。

---

## 5. 工程实现

### 5.1 复用现有基础设施
- `src/controllers/QwenVLController` (E+P+D) — exp01a 已用
- Fast-WAM controller (A) — exp04a 已用
- `src/utils/PhaseTimer` — 已支持 nvtx markers
- Hydra configs — 新建 `configs/epda/` 目录

### 5.2 新增组件
```
src/
├── concurrent/
│   ├── __init__.py
│   ├── stage_runner.py      # 单阶段独立线程/进程运行器
│   └── interference_probe.py # 共置实验协调器
├── metrics/
│   ├── sm_utilization.py    # 从 nvidia-smi dmon 解析
│   └── hbm_bandwidth.py     # torch.cuda profiler 封装
configs/
└── epda/
    ├── exp08a_interference.yaml   # 干扰矩阵实验
    ├── exp08b_slo_violation.yaml  # SLO 违反分析
    └── exp08c_step_overlap.yaml   # A 阶段 overlap 分析
```

### 5.3 关键技术挑战
| 挑战 | 方案 |
|------|------|
| 同 GPU 并发运行两个模型（内存紧张） | 7B VLM + 350M DiT ≈ 17GB，RTX 5880 Ada 48GB 够用 |
| 精确测量共置时的单阶段延迟 | CUDA events per-stream + nvtx markers |
| SM 利用率细粒度采样 | nvidia-smi dmon 每 10ms 采一次 + 与 nvtx 时间线对齐 |
| 避免 warmup 污染 | 每阶段独立 warmup 5 次，测量 20 次取中位数（与 exp01-06 一致） |

---

## 6. 执行计划（建议 2-3 周）

| 阶段 | 工作量 | 可交付 |
|------|-------|--------|
| Week 1 | Concurrent runner + interference probe 基础设施 | 能同时跑两个 controller，nvtx 正确 |
| Week 2 | exp08a 完整运行（6x6 矩阵）+ exp08c 分析 | 干扰矩阵图 + overlap 上界估计 |
| Week 3 | exp08b SLO serving wrapper + 最终分析 | SLO 曲线 + position paper draft |

**里程碑**: Week 2 结束即可有足够素材与 Hao 见面。Week 3 是锦上添花。

---

## 7. 风险与备选方案

| 风险 | 概率 | 影响 | 备选 |
|------|------|------|------|
| 干扰矩阵显示干扰量级很小（<10%）→ 论证失败 | 中 | 高 | 退回候选 B（Visual KV 稀疏压缩），Gini 数据已就绪 |
| SLO 场景定义难以对齐真实 VLA 部署 | 中 | 中 | 用 DreamZero 论文 7Hz 作为锚点定义 SLO |
| 单 GPU 跑不下两个模型 | 低 | 中 | 换成 Qwen-VL-3B + NitroGen 174M（小模型组合） |
| Hao 已经有类似工作 in progress | 低 | 高 | 见面时作为请教问题提出，调整方向 |

---

## 8. 成功标准

**最低标准**（必须达到才去见 Hao）:
- 一张能让人一眼看懂 "EPDA 四阶段干扰存在" 的 motivation 图
- 用 Hao 工作风格的语言（goodput / disaggregation / structural）表达问题

**理想标准**:
- Position paper 草稿（8 页），引用 DistServe + EPD Disaggregation + FastVideo 作为前沿
- 具体 exp09 (disaggregation 实现) 的 one-pager

---

## 9. 开放问题（见 Hao 时请教）

1. EPDA 的 "A 阶段" 是否已经在 Hao Lab 的 roadmap？
2. Visual feature transfer（E→P）和 latent transfer（D→A）的带宽瓶颈是否严重？是否值得专门设计 interconnect 原语？
3. "EPDA-goodput" 指标定义中，Action SLO 用 control frequency 还是 per-step budget？哪个更符合机器人学场景？
4. 实际部署中，多个机器人请求是否 co-located 在同一 GPU？还是专机专用？这影响 disaggregation 收益预估的基础假设。
