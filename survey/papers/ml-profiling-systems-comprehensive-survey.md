# ML Inference Profiling 系统综合调研

> **调研日期**: 2026-04-15
> **调研范围**: 8 个 ML inference 系统 + CUDA profiling 最佳实践
> **调研方法**: 4 个并行 research agent，15+ web 源，覆盖 GitHub 源码、官方文档、技术博客
> **目的**: 为 VLLA VLM profiling framework 建立可靠的 measurement infrastructure

---

## Executive Summary

**核心发现**: 生产级 ML inference 系统中，**没有一个系统同时具备我们需要的三个特性**：
1. CUDA Event 级精度的 GPU timing
2. VLM 特有的 Encode/Prefill/Decode 三阶段分解
3. Per-input 级别的细粒度 profiling

我们的 PhaseTimer 架构方向是正确的，但需要参考这些系统做出关键改进。

**最重要的三个发现**:
- vLLM/SGLang/TensorRT-LLM **都不用 CUDA Events 做 latency benchmarking** — 它们用 `synchronize() + perf_counter()`，因为这更简单且对 end-to-end 足够
- 只有 DeepSpeed 的 `SynchronizedWallClockTimer` 同时支持 CPU 和 CUDA Event 双模式
- 我们的 `timing.py` 有一个 **确认的 bug**：CPU backend 的 `record_end()` 在 CUDA 模式下不会被调用

---

## 1. 系统概览

### 1.1 覆盖系统

| 系统 | 维护者 | 定位 | Profiling 特点 |
|------|--------|------|---------------|
| **vLLM** | vLLM Project | LLM serving | EncoderTimingStats, LayerwiseProfileResults, 事件驱动 phase tracking |
| **SGLang** | SGLang Project | LLM serving | Mixin 架构, stage-based profiling, 三级 phase tracking, 最全 observability |
| **FastVideo** | Hao AI Lab | Video generation | 完全委托 PyTorch profiler, 零自建 timer |
| **TensorRT-LLM** | NVIDIA | LLM serving | 最成熟的 phase 分解 (TTFT/TPOT), 40+ Prometheus metrics |
| **DeepSpeed** | Microsoft | 训练+推理 | FlopsProfiler (理论 FLOPs), dual-backend timer, trim_mean() |
| **Triton** | NVIDIA | 推理服务 | 最"生产级" — 三种 latency 模式, perf_analyzer stabilization |
| **llama.cpp** | ggml-org | 边缘推理 | 最简洁 — RAII timer, 直接 t_p_eval/t_eval |
| **MLC LLM** | MLC AI | 跨平台推理 | Serving benchmark client, P25-P99 完整统计 |

---

## 2. Timing 机制深度对比

### 2.1 谁用什么计时？

| 系统 | CUDA Events | CPU Timer | torch.profiler | 精度级别 |
|------|:-----------:|:---------:|:--------------:|:--------:|
| vLLM | Multi-stream sync only | `perf_counter()` + `time.time()` | Yes (TorchProfilerWrapper) | ms (CPU) / kernel (profiler) |
| SGLang | - | `perf_counter()` + `monotonic()` | Yes (SchedulerProfilerMixin) | ms |
| FastVideo | - | - | Yes (TorchProfilerController) | kernel (via trace) |
| TensorRT-LLM | - | `time.time()` | - | ms |
| DeepSpeed | **Yes** (CudaEventTimer) | `time.time()` | - | sub-ms (CUDA mode) |
| Triton | N/A | N/A | N/A | request-level |
| llama.cpp | - | `int64_t µs` | N/A | µs |
| MLC LLM | - | `time.monotonic()` | - | ms |
| **VLLA (ours)** | **Yes** | `perf_counter()` | - | **sub-ms** |

### 2.2 关键洞察

**为什么生产系统不用 CUDA Events 做 benchmarking？**

1. **vLLM 的做法**: `model.generate()` 是同步的 — 返回时 GPU 已完成。所以 `perf_counter()` 够用。
   ```python
   # vLLM benchmark_latency.py
   start = time.perf_counter()
   llm.generate(prompts)  # 内部已 sync
   elapsed = time.perf_counter() - start
   ```

2. **vLLM EncoderTimingStats**: 对 vision encoder 用 `synchronize() + perf_counter()` bracket：
   ```python
   torch.accelerator.synchronize()  # barrier BEFORE
   start = time.perf_counter()
   yield  # encoder forward
   torch.accelerator.synchronize()  # barrier AFTER
   elapsed = time.perf_counter() - start
   ```

3. **SGLang bench_one_batch**: 同样用 `synchronize() + perf_counter()`：
   ```python
   model_runner.synchronize()
   tic = time.perf_counter()
   model_runner.extend(reqs)  # prefill
   model_runner.synchronize()
   prefill_latency = time.perf_counter() - tic
   ```

**结论**: 对于 end-to-end phase timing，`synchronize() + perf_counter()` 是业界主流。CUDA Events 更适合：
- 同一 stream 内的 **微观** 操作（单 kernel / 单 layer）
- **多 stream** 同步协调
- 需要 **纯 GPU 时间** 不含 CPU overhead 的场景

我们的 CUDA Event approach 不是错的，但要理解它测的是什么：CUDA Event 不包含 CPU-GPU 交互开销，而 `sync + perf_counter` 包含。对于 VLM inference 的 E/P/D 分解，两种方式差距通常 < 1ms。

---

## 3. Phase 定义与分离机制

### 3.1 各系统的 Phase 定义

**TensorRT-LLM** (最清晰的四时间戳模型):
```
arrival → first_scheduled → first_token → last_token
         |--- queue ---|--- prefill ---|--- decode ---|
```

**vLLM** (事件驱动):
```python
@dataclass
class FinishedRequestStats:
    queued_time: float    # QUEUED → first SCHEDULED
    prefill_time: float   # first SCHEDULED → first NEW_TOKEN
    decode_time: float    # first NEW_TOKEN → last NEW_TOKEN
    inference_time: float # first SCHEDULED → last NEW_TOKEN
```

**SGLang** (三级层次):
```
Level 1 (原子): tokenize → prefill_forward → decode_forward
Level 2 (调度): api_dispatch → request_process
Level 3 (复合): decode_loop → prefill_chunked
```

**llama.cpp** (最简洁):
```c
t_p_eval_ms  // prefill 总时间
t_eval_ms    // decode 总时间
t_sample_ms  // sampling 总时间
```

### 3.2 我们的 Phase 定义 vs 业界

| Phase | 我们 (VLLA) | vLLM | SGLang | TensorRT-LLM |
|-------|-------------|------|--------|--------------|
| **Vision Encode** | ✅ 独立阶段 | ✅ EncoderTimingStats | ❌ 不区分 | ❌ 不区分 |
| **Prefill** | ✅ LLM 首次 forward | ✅ SCHEDULED→FIRST_TOKEN | ✅ prefill_forward | ✅ first_scheduled→first_token |
| **Decode** | ✅ LLM 后续 forward (累加) | ✅ FIRST_TOKEN→LAST_TOKEN | ✅ decode_forward | ✅ first_token→last_token |
| **Projection gap** | ❌ 未捕获 | ❌ 未捕获 | ❌ | ❌ |
| **Sampling** | ❌ 未分离 | ❌ 含在 decode 中 | ❌ | ❌ |

**我们的独特优势**: Vision Encode 作为独立阶段的 profiling，这是 vLLM 之外唯一这样做的。vLLM 的 `EncoderTimingStats` 是 batch-level 均分的，我们是 per-input 精确计时。

**待改进**: Encode → Prefill 之间的 projection layer gap 没有被任何系统显式捕获。

### 3.3 SGLang 的 Stage-based Profiling (值得借鉴)

SGLang 实现了一个精巧的状态机，自动根据 `ForwardMode` 切分 profiler trace：

```python
def _profile_batch_predicate(self, batch):
    if batch.forward_mode.is_prefill():
        if self.profiler_prefill_ct == 0:
            self.start_profile(batch.forward_mode)
        self.profiler_prefill_ct += 1
        if self.profiler_prefill_ct > target:
            self.stop_profile(stage=ForwardMode.EXTEND)
    elif batch.forward_mode.is_decode():
        # stop prefill profiler, start decode profiler
```

这会生成**独立的 prefill trace 和 decode trace 文件**，直接可用 TensorBoard 分析。

---

## 4. Warmup 策略对比

| 系统 | Warmup 方法 | 次数/时长 | 特殊处理 |
|------|------------|----------|---------|
| vLLM | 配置 `delay_iterations` | 默认 10 次 | - |
| SGLang | 1 次 warmup run | 1 次 (缩短 output_len) | `min(32, output_len)` |
| TensorRT-LLM | 显式 warmup 请求 | N 次 + backend 重启 | warmup 后重启 |
| DeepSpeed | `warm_up=N` 参数 | 可配置 | 跳过前 N iterations |
| FastVideo | PyTorch schedule | wait + warmup + active | 三阶段 |
| Triton perf_analyzer | Stabilization passes | 动态收敛 | 结果稳定才停 |
| Triton `do_bench` | 25ms warmup 时间 | 自适应次数 | 基于时间不是次数 |
| **VLLA (ours)** | **`num_warmup_runs`** | **1 次 (不足)** | **待改进** |

### Warmup 需要消除的因素

1. **CUDA context 初始化**: 首次 CUDA 调用 ~100ms-1s
2. **cuBLAS/cuDNN handle 创建**: 首次 matmul/conv 的额外开销
3. **JIT 编译**: `torch.compile` 首次编译, `cudnn.benchmark` 算法搜索
4. **Memory allocator 预热**: PyTorch caching allocator 首次分配
5. **Lazy kernel 加载**: 某些 CUDA kernel 首次使用时才加载

**建议**: 将 `num_warmup_runs` 从 1 增加到 **至少 3**。

---

## 5. 统计方法对比

| 系统 | Mean | Median | Percentiles | Trimmed Mean | Outlier Removal |
|------|:----:|:------:|:-----------:|:------------:|:---------------:|
| vLLM | ✅ | ✅ | P10/P25/P50/P75/P90/P99 | - | - |
| SGLang | ✅ | ✅ | P95/P99 | - | - |
| TensorRT-LLM | ✅ | - | P50/P90/P95/P99 | - | - |
| DeepSpeed | ✅ | - | - | **✅ 10% trim** | ✅ (via trim) |
| Triton perf_analyzer | ✅ | - | P50/P90/P95/P99 | - | **✅ stabilization** |
| MLC LLM | ✅ | - | P25/P50/P75/P90/P95/P99 | - | - |
| Triton `do_bench` | - | **✅** | P20/P80 (quantiles) | - | - |
| **VLLA (ours)** | ✅ | - | - | - | - |

### DeepSpeed 的 trim_mean (推荐借鉴)

```python
def trim_mean(data, trim_percent=0.1):
    """排序后剔除两端 10%，计算均值"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    trim = int(n * trim_percent)
    return np.mean(sorted_data[trim:n-trim])
```

### Triton do_bench 的做法 (GPU microbenchmark 黄金标准)

- 默认 100ms 的 repetition 窗口（自适应迭代次数）
- 返回 **median** + 20th/80th percentile
- 每次测量前可选 L2 cache flush

### 建议的统计方案

```python
stats = {
    "median_ms": np.median(arr),       # 主要报告指标
    "mean_ms": np.mean(arr),
    "std_ms": np.std(arr),
    "p10_ms": np.percentile(arr, 10),
    "p90_ms": np.percentile(arr, 90),
    "p99_ms": np.percentile(arr, 99),
    "cv": np.std(arr) / np.mean(arr),  # 变异系数 > 5% 说明不稳定
    "n_samples": len(arr),
}
```

---

## 6. Memory Profiling 对比

| 系统 | Host Memory | GPU Memory | KV Cache | Memory Snapshot | Peak Tracking |
|------|:-----------:|:----------:|:--------:|:---------------:|:-------------:|
| vLLM | - | `MemorySnapshot` dataclass | ✅ (automatic sizing) | ✅ (context manager) | ✅ |
| SGLang | - | `get_available_gpu_memory()` | ✅ (radix cache metrics) | ✅ (via torch profiler) | - |
| TensorRT-LLM | `psutil` USS | `pynvml` | ✅ (block counters) | - | ✅ |
| DeepSpeed | `psutil` | `accelerator.memory_usage()` | - | - | ✅ (parameter counting) |
| Triton | `/proc/meminfo` | DCGM | - | - | max over period |
| llama.cpp | - | per-device breakdown | - | - | - |
| **VLLA (ours)** | **❌** | **❌** | **❌** | **❌** | **❌** |

**vLLM MemorySnapshot 模式 (推荐)**:
```python
@dataclass
class MemorySnapshot:
    torch_peak: int      # PyTorch 分配峰值
    free_memory: int     # 设备空闲
    total_memory: int    # 设备总量
    cuda_memory: int     # CUDA 管理总量
    torch_memory: int    # torch.cuda.memory_reserved()
    non_torch_memory: int  # CUDA - PyTorch 部分
```

---

## 7. CUDA Profiling 最佳实践

### 7.1 正确的 CUDA Event Timing

```python
# ✅ CORRECT
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)  # 纯 GPU 时间，~0.5µs 精度

# ❌ WRONG: CPU timer 测 GPU 异步操作
start = time.perf_counter()
output = model(input)  # GPU 异步执行，CPU 立刻返回
elapsed = time.perf_counter() - start  # 测的是 kernel launch 时间

# ⚠️ ACCEPTABLE: sync + CPU timer (业界主流)
start = time.perf_counter()
output = model(input)
torch.cuda.synchronize()  # 等 GPU 完成
elapsed = time.perf_counter() - start  # 包含 sync overhead
```

### 7.2 NVTX 标注 (用于 Nsight Systems 可视化)

```python
# 推荐在 PhaseTimer hooks 中集成
def _encode_pre_hook(module, inputs):
    torch.cuda.nvtx.range_push("ENCODE")
    timer.mark_start("encode")

def _encode_post_hook(module, inputs, output):
    timer.mark_end("encode")
    torch.cuda.nvtx.range_pop()
```

### 7.3 GPU Clock 锁定 (可重复 benchmark)

```bash
# 锁定 GPU 时钟频率，避免 thermal throttling 导致的波动
nvidia-smi -i 0 -lgc 1500,1500
# 运行 benchmark ...
nvidia-smi -i 0 -rgc  # 恢复
```

### 7.4 Phase Boundary 的 Synchronization

**不要在 phase boundary 加 synchronize** (对 benchmark 模式):
- CUDA Event 已保证正确时序
- 额外的 synchronize 会破坏 pipeline 并行性，测出来的数不是真实性能
- 只有需要确保 phases 完全隔离时才加

### 7.5 torch.profiler vs CUDA Events 的定位

| 用途 | 推荐工具 |
|------|---------|
| **Latency benchmarking** (E/P/D ms) | CUDA Events 或 sync + perf_counter |
| **Kernel-level 分析** (哪个 kernel 慢) | torch.profiler |
| **系统级瓶颈定位** (CPU idle? memory?) | Nsight Systems |
| **单 kernel 优化** (occupancy, bandwidth) | Nsight Compute |
| **理论计算效率** (actual/theoretical FLOPs) | DeepSpeed FlopsProfiler |

---

## 8. 我们的 PhaseTimer 代码审查

### 8.1 确认的 Bug: CPU backend record_end 未调用

**文件**: `src/utils/timing.py:128-131`

```python
def mark_end(self, phase: str) -> None:
    backend = self._active[phase]
    if self._use_cuda:          # ← BUG: CPU mode 时跳过了 record_end()
        backend.record_end()
```

`backend` 可能是 `_CpuTimerBackend`，但 `record_end()` 只在 `self._use_cuda` 为 True 时调用。

**修复**: `backend.record_end()` 应无条件调用。

### 8.2 已有优势

1. **CUDA Event timing** — 比 vLLM/SGLang/TensorRT-LLM 的 CPU timer 更精确
2. **Immutable patterns** — `self._active = {**self._active, phase: backend}` 符合编码规范
3. **Accumulative decode** — 正确累加多步 decode 时间
4. **Backend abstraction** — CPU/CUDA 双模式切换干净

### 8.3 缺失项 (按优先级)

| 优先级 | 缺失项 | 参考系统 | 建议 |
|--------|--------|---------|------|
| **P0** | CPU backend bug | - | 修复 mark_end |
| **P0** | Warmup 不足 (1次) | vLLM (10次), DeepSpeed (N次) | 增加到 3+ |
| **P0** | 统计不完整 (只有 mean) | DeepSpeed (trim_mean), vLLM (percentiles) | 加 median + P90/P99 + CV |
| **P0** | Benchmark 次数不足 (3次) | vLLM (30次), Triton do_bench (100ms 窗口) | 增加到 10+ |
| **P1** | 无 memory tracking | vLLM MemorySnapshot, TensorRT-LLM psutil+pynvml | 每 phase 记录 GPU memory |
| **P1** | 无 NVTX 标注 | SGLang hook_manager, 通用最佳实践 | 在 hooks 中加 nvtx.range |
| **P1** | Encode→Prefill gap | 所有系统都没覆盖 | 新增 "project" phase 或检测 gap |
| **P2** | 无 FLOPs 估算 | DeepSpeed FlopsProfiler | GPU utilization 分析 |
| **P2** | 无 torch.profiler 集成 | FastVideo, vLLM, SGLang | 可选 kernel-level trace 模式 |
| **P2** | 无 Prometheus 导出 | SGLang ~50 metrics, TensorRT-LLM 40+ | 批量实验监控 |

---

## 9. 架构建议: 三层 Profiling 体系

参考这些系统的共同模式，建议将 VLLA profiling 分为三层：

```
┌──────────────────────────────────────────────────┐
│  Layer 3: Observability                           │
│  (可选, 批量实验监控)                              │
│  - Prometheus metrics 导出                        │
│  - Memory timeline HTML                           │
│  - Experiment flight recorder                     │
├──────────────────────────────────────────────────┤
│  Layer 2: Detailed Analysis                       │
│  (可选, kernel 级分析)                             │
│  - torch.profiler 集成 (参考 SGLang Mixin)         │
│  - NVTX annotation (参考 SGLang hook_manager)      │
│  - Nsight Systems trace                           │
│  - Per-layer CUDA time (参考 vLLM Layerwise)       │
├──────────────────────────────────────────────────┤
│  Layer 1: Phase Benchmarking  ← PhaseTimer 所在层  │
│  (核心, 每次实验必用)                              │
│  - CUDA Event timing for E/P/D                    │
│  - Warmup (3+ iterations)                         │
│  - Multi-run statistics (median, percentiles)     │
│  - Memory snapshot (before/after each phase)      │
│  - GPU clock locking for reproducibility          │
└──────────────────────────────────────────────────┘
```

---

## 10. 结论

### 10.1 Profiling 系统的 Root of Trust

你最初的直觉是对的: **profiling 系统是整个研究的 root of trust**。

这些生产系统告诉我们:
- **简单可靠 > 复杂精确**: vLLM/SGLang 用 `sync + perf_counter()` 而非 CUDA Events，因为它够用且不容易出错
- **Warmup 是必须的**: 没有一个系统跳过 warmup
- **统计严格性是标配**: percentiles + 多次运行是基本要求
- **Memory tracking 是标配**: 所有系统都追踪 GPU memory

### 10.2 下一步行动

1. **立即修复** `timing.py` CPU backend bug
2. **增加** warmup 次数 (1 → 3) 和 benchmark 次数 (3 → 10)
3. **增加** median + percentile + CV 统计
4. **增加** 每 phase 的 memory snapshot
5. **交叉验证**: 用 torch.profiler trace 验证 PhaseTimer 数字是否一致
6. 然后 profiling 系统才算 trustworthy，后续 attention analysis 数据才有意义

---

## 详细子报告索引

- **SGLang 深度调研**: `notes/sglang-profiling-deep-survey.md`
- **8 系统横向对比**: `survey/papers/profiling-systems-survey.md`
- vLLM 和 CUDA 最佳实践的详细发现已整合到本报告中

## 参考源 (15+ 源)

### GitHub 源码
1. `vllm-project/vllm` — model_runner, profiler/, benchmarks/, metrics/
2. `sgl-project/sglang` — observability/, scheduler_profiler_mixin, bench_*
3. `hao-ai-lab/FastVideo` — profiler.py, composed_pipeline_base.py
4. `NVIDIA/TensorRT-LLM` — profiler.py, metrics/, bench/
5. `microsoft/DeepSpeed` — flops_profiler/, timer.py, inference/engine.py
6. `triton-inference-server/server` — metrics docs
7. `ggml-org/llama.cpp` — llama.h, common.h
8. `mlc-ai/mlc-llm` — bench/request_record.py

### 官方文档
9. PyTorch — torch.cuda.Event, CUDA Semantics, Memory Management
10. PyTorch — Profiler Tutorial, TensorBoard Profiler
11. NVIDIA — Nsight Systems / Nsight Compute docs
12. DeepSpeed — FlopsProfiler Tutorial

### 技术博客
13. Speechmatics — "Timing Operations in PyTorch" (GPU timing pitfalls)
14. PyTorch Blog — "Understanding GPU Memory" (Part 1 & 2)
15. vLLM Blog — "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
