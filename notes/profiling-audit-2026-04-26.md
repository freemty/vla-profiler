# Profiling Audit — 2026-04-26

> 审计范围：exp01a / exp02a / exp03a / exp04a / exp04b / exp06a / exp07a 的 E/P/D 和 E/C/A profiling 部分
> 目的：为 exp08 EPDA 干扰量化奠定干净的 baseline

---

## Summary (TL;DR)

- **本地数据残缺率高**：exp01a / exp02a / exp06a 本地只有 README，原始 JSON 结果在 server 端，未下载。
- **统计口径分裂**：exp03a / exp07a 有完整 median/P10/P90/P99/all_ms；exp04a / exp04b 只有 mean/std/min/max，无 raw data，无 median。
- **exp07a 双峰问题已记录但未修复**：5 次 warmup 不足以稳定 GPU 功率状态，前 12 次运行比后 8 次慢 ~25%；稳定态数值（runs 13-20）更可信，但 reported aggregated stats 混入了不稳定运行。
- **exp04b 统计最脆弱**：3 次 warmup + 10 次迭代 + CV>20%（encode），没有 raw 数据，无法做分位数分析。
- **PhaseTimer CPU backend bug 已修复**（2026-04-15），CUDA 模式下计时逻辑正确，已有 cross-validation task 验证。

---

## 1. 完整性问题

| 实验 | 本地 README | 本地 JSON 结果 | predictions.md | analysis.md | 严重度 |
|------|------------|---------------|---------------|------------|--------|
| **exp01a** | ✅ | ❌ 缺失（在 server） | ❌ | ❌ | **HIGH** — 最重要的 E/P/D baseline，本地无法复现核心数字 |
| **exp02a** | ❌ 本地无 exp 目录 | ❌ | ❌ | ❌ | **HIGH** — ACT baseline 完全不在本地 |
| **exp03a** | ✅ | ✅ (2 input variants) | 内嵌于 README | 内嵌于 README | LOW — 完整 |
| **exp04a** | ✅ | ✅ (10/20 steps) | 内嵌于 README | 内嵌于 README | MEDIUM — JSON 缺 raw data / 分位数 |
| **exp04b** | ✅ | ✅ | 内嵌于 README | 内嵌于 README | MEDIUM — 同上 + warmup 不足 |
| **exp06a** | ✅ | ❌ 缺失（在 server） | 内嵌于 README | 内嵌于 README | MEDIUM — 数据在 server，本地只有 README 中的汇总表 |
| **exp07a** | ✅ | ✅ | 内嵌于 README | 内嵌于 README | MEDIUM — 双峰问题 |

**Phase 命名混用情况：**

| 实验 | 实际 JSON phase key |
|------|-------------------|
| exp03a | `encode` / `context` / `action` |
| exp04a | `encode` / `context` / `action` |
| exp04b | `encode` / `video_denoise` / `action_denoise` |（不同！非 E/V/A）
| exp07a | `encode` / `context` / `action` |

exp04b 的 JSON 中 phase 命名为 `video_denoise` / `action_denoise`，与其他实验的 `action` 不一致。README 及 SKILL.md 里标记为 E/V/A，但 JSON key 更详细。对 exp08 跨实验对比时需注意字段名不同。

**warmup / iterations 汇总：**

| 实验 | warmup | iterations | 框架 |
|------|--------|-----------|------|
| exp01a | 3 | 10 | Hydra + PhaseTimer |
| exp02a | 5 | 20 | Hydra + PhaseTimer |
| exp03a | 3 | 10 | Hydra + PhaseTimer |
| exp04a | 5 | 20 | Standalone script |
| exp04b | 3 | 10 | Standalone script |
| exp06a | 5 | 20 | Hydra + PhaseTimer |
| exp07a | 5 | 20 | Hydra + PhaseTimer |

Warmup 无统一标准：早期实验（exp01a/exp03a/exp04b）用 3 次，后期（exp04a/exp06a/exp07a）改为 5 次。对于有长期 GPU 功率爬坡问题的模型（exp07a），5 次仍不够。

---

## 2. Solid 程度（统计/warmup/单位）

| 实验 | median/P10/P90/P99 | raw all_ms | CV 水平 | warmup 充分性 | 单位 | 主要问题 |
|------|-------------------|-----------|---------|--------------|----|---------|
| exp01a | 未知（本地无数据） | 未知 | 未知 | 未知 | ms | **本地无数据，无法审计** |
| exp02a | 未知（本地无目录） | 未知 | 未知 | 未知 | ms | **本地无数据，无法审计** |
| exp03a | ✅ 全有 | ✅ 10 runs | CV<3%（极稳定） | 充分 | ms | 无显著问题 |
| exp04a | ❌ 只有 mean/std | ❌ | Context CV=13.3%，Action CV=11.3% | 基本充分 | ms | 缺 median/P90，high CV 无法用 median 排除异常值 |
| exp04b | ❌ 只有 mean/std | ❌ | Encode CV=21.2% | **不充分（3 次 warmup）** | ms | 最脆弱；encode 方差极大（std=16ms on mean=75ms）；无分位数 |
| exp06a | 未知（本地无 JSON） | 未知 | 待查（README 汇总无 std）| 5 次，可能充分 | ms | 需下载 server 原始数据确认 |
| exp07a | ✅ 全有 | ✅ 20 runs | 整体 CV=10-12%，但源于双峰 | **不充分（5 次不够）** | ms | **双峰：runs 1-12 约为 runs 13-20 的 1.25-1.27x**；报告的 aggregated 混入不稳定态 |

**exp07a 双峰量化：**
- 不稳定态（runs 1-12）：E=11.84ms / C=32.77ms / A=205.28ms
- 稳定态（runs 13-20）：E=9.32ms / C=26.22ms / A=164.76ms
- 比率：约 1.27x（E）/ 1.25x（C/A）
- 结论：应以稳定态数值（E≈9.3ms / C≈26ms / A≈165ms / total≈200ms）为 baseline，而非 README 中给出的 mean（E=10.8ms / C=30.3ms / A=189ms）

**单位一致性：** 所有本地可查实验均使用 ms，无 us/s 混用问题。

**Decode 归一化：** exp01a 的 decode per-token 时间（~18-21ms/tok）未检查计算方法（本地无 JSON），需确认 PhaseTimer decode 累加后是否除以 step 数。根据 SKILL.md Lesson 9，PhaseTimer 已修复为 list 累加，但归一化是否在 `profiling_task.py` 中做除法需人工确认。

---

## 3. Bug 与风险点

| 问题 | 位置 | 状态 | 修复建议 |
|------|------|------|---------|
| **PhaseTimer CPU backend no-op bug** | `src/utils/timing.py` (已修复) | ✅ 已修复（2026-04-15） | 无需操作；已验证 |
| **exp04b warmup=3 + CV>20%** | `scripts/profile_lingbot_va.py` + exp04b JSON | ⚠️ 已完成但数据质量低 | 如需精确数字需重跑，至少 warmup=10；或使用当前 mean±std 加宽置信区间 |
| **exp07a bimodal（GPU 功率状态未稳定）** | exp07a 结果 JSON | ⚠️ 已记录，未重跑 | P0 for baseline 精度；需 `nvidia-smi -pm 1` 锁定功率 + 增加 warmup=15 后重跑；或明确以稳定态（runs 13-20）作为 canonical baseline |
| **exp04a/04b 无 raw iteration data** | standalone profiling scripts | ⚠️ 设计缺陷 | 重要实验需增加 `all_ms` 字段输出；exp04b 最需要（高方差） |
| **exp04a/04b 无 median/P10/P90/P99** | standalone profiling scripts | ⚠️ 格式不统一 | 与 Hydra 路径的 PhaseTimer 统计口径统一；补充分位数 |
| **exp03a README 命名混淆（E/P/D vs E/C/A）** | `exp/exp03a/README.md` | ⚠️ 文档问题 | README 动机段落写 "E/P/D"，应统一为 E/C/A（Flow VLA 无 decode） |
| **exp04b phase key 不一致**（`video_denoise`/`action_denoise` vs `encode`/`context`/`action`）| exp04b JSON | ⚠️ 命名不统一 | 不影响已有分析（README 已说明），但 exp08 跨实验代码需注意字段映射 |
| **并发/多 stream timing 未验证** | `src/utils/timing.py` | ⚠️ 设计盲区 | PhaseTimer 使用单 CUDA stream 的 Event；多 stream 并发（如 exp08 EPDA）时，每个 phase 需绑定到对应 stream 的 Event，否则 `end_event.synchronize()` 可能等待错误的 stream。需要重构为 stream-aware PhaseTimer |
| **exp01a / exp02a / exp06a 本地无原始 JSON** | `exp/` 目录 | ❌ 数据缺失 | 运行 `bash scripts/download-results.sh` 同步 server 数据 |
| **decode per-token 归一化逻辑未本地验证** | `src/tasks/profiling_task.py` | 未检查 | 确认 `elapsed_ms("decode") / timer.decode_step_count` 是否正确 |

**关于 exp07a 双峰是 timing bug 还是真实现象：**
根据 SKILL.md Lesson 50 和数据分析，这是**真实 GPU 功率爬坡现象**，而非 timing code bug。所有 phase（E/C/A）同步从 run 12→13 发生跳变（1.25-1.27x），且跳变后数值极其稳定（stable runs std 几乎消失）。这是 GPU 从节能模式切换到高性能模式的典型特征。不是测量误差。

---

## 4. 对 exp08 EPDA 的影响

### 可直接复用的 baseline

| 实验 | 可复用原因 | 推荐值 |
|------|----------|-------|
| **exp03a LingBot-VLA** | 完整统计、CV<3%、稳定 | single_img: E=35.7ms / C=38.3ms / A=0.48ms |
| **exp04a Fast-WAM** | 充足迭代（20次）、有文档、结构清晰 | @10step: E=7.6ms / C=36.7ms / A=362ms（使用 mean±std，了解无 median） |
| **exp06a NitroGen** | 线性结构简单、per-step 精确 | 7.2ms/step（但需从 server 下载原始 JSON 验证分位数） |

### 需要重跑或注意的

| 实验 | 问题 | 建议 |
|------|------|------|
| **exp07a Pi-Zero** | 双峰，aggregated stats 包含不稳定态 | 使用稳定态数值（E=9.3ms/C=26ms/A=165ms）或重跑（`nvidia-smi -pm 1` + warmup=15） |
| **exp04b LingBot-VA** | warmup=3，CV=21%，无分位数 | 如需精确 E 时间需重跑；V/A phase 相对稳定（CV≈14%），勉强可用 |
| **exp01a Qwen2.5-VL-7B** | 本地无 JSON | 下载后补充审计 |

### exp08 EPDA 测量缺少的基础设施

1. **Stream-aware PhaseTimer**：exp08 要量化 E/P/D/A 四阶段的干扰，可能需要在不同 CUDA stream 上并发运行多个阶段。当前 PhaseTimer 假设单 stream，必须扩展。具体：`_CudaTimerBackend` 需要接受 `stream` 参数，`mark_end` 时同步对应 stream 而非默认 stream。

2. **Wall-clock vs phase-sum 差值跟踪**：现有 timing_validation task 已对比 PhaseTimer 与 torch.profiler，但没有追踪 "sum(E+C+A) vs 实际 wall clock" 的比率。exp08 需要明确这个 gap（framework overhead、host-device sync、空闲 bubble），否则无法精确量化干扰带来的额外延迟。

3. **单 GPU 多进程干扰实验**：需要能在同一 GPU 上同时 launch 两个进程并测量各自延迟的测试脚本。当前 framework 是单进程单模型。需要新增 `concurrent_profiling_task.py`。

---

## 5. 优先级建议

### P0 — 必须修才能做 exp08

1. **下载 server 数据**（exp01a/exp02a/exp06a JSON）
   - `bash scripts/download-results.sh` — 不需要代码改动，1 分钟内可完成

2. **确认 exp07a canonical baseline**
   - 明确以 stable-window（runs 13-20）作为官方数字，更新 README 和 SKILL.md 对应数据
   - 或重跑（`nvidia-smi -pm 1` + warmup=15），更彻底

3. **stream-aware PhaseTimer 设计**
   - exp08 并发测量的核心基础设施，无此不能做 EPDA 干扰量化

### P1 — 应该修但可以绕开

4. **exp04b 重跑**（warmup=3 不足，CV=21%）
   - 如果 exp08 需要 full WAM 作为对比，这个数据可信度偏低
   - 绕开方式：用 mean±2σ 宽置信区间，标注数据质量为 "低"

5. **统计口径统一**（exp04a/exp04b 补 raw data + median/P10/P90/P99）
   - 修改 `scripts/profile_fastwam.py` 和 `profile_lingbot_va.py`，输出与 Hydra 路径一致的 JSON 格式

### P2 — 优化项

6. **exp03a README 命名修正**（E/P/D → E/C/A，文档内一致性）

7. **exp04b JSON phase key 标准化**（`video_denoise` → `video`，或在 exp08 代码中加映射层）

8. **wall-clock vs phase-sum 差值 tracking**（为 exp08 干扰量化提供更精确的测量基准）

---

*审计时间：2026-04-26 | 审计人：Claude | 数据源：本地 `/Users/sum_young/code/projects/vlla/exp/`*
