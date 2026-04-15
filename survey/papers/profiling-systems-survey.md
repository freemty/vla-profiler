# ML Inference Profiling Systems 深度调研

> 调研目标：分析主流 ML 推理系统的 profiling 实现，为 VLLA 的 VLM profiling framework 提供设计参考。
> 覆盖系统：FastVideo, TensorRT-LLM, DeepSpeed, Triton Inference Server, vLLM, SGLang, llama.cpp, MLC LLM
> 调研日期：2026-04-15

---

## 1. FastVideo (Hao AI Lab)

**仓库**: `hao-ai-lab/FastVideo`
**关键文件**: `fastvideo/profiler.py`, `fastvideo/pipelines/composed_pipeline_base.py`

### 1.1 Profiling 架构

FastVideo 完全委托 PyTorch 原生 profiler，没有自建 timing 基础设施：

```python
# fastvideo/profiler.py 核心类
ProfilerRegion          # Dataclass: name, description, default_enabled
TorchProfilerConfig     # 环境变量驱动的 region 配置
TorchProfilerController # 主控制器，管理 profiler 生命周期
```

**关键设计决策**：
- **零自定义 timer**：不使用 CUDA events 或 `time.perf_counter()`，完全依赖 `torch.profiler.profile()`
- **Activities 覆盖**：同时 profile `ProfilerActivity.CPU` 和 `ProfilerActivity.CUDA`
- **Environment-driven**：通过 `FASTVIDEO_TORCH_PROFILER_DIR` 等环境变量控制，零代码修改

### 1.2 Diffusion Step Profiling

FastVideo 对 diffusion denoising loop 的 profiling 通过 **pipeline stage 抽象** 实现：

```python
# composed_pipeline_base.py
for stage in self.stages:
    batch = stage(batch, fastvideo_args)
```

- 每个 `PipelineStage` 对应一个 denoising step
- Profiling regions 用 context manager 包裹：`with self.profiler_controller.region('...'):`
- 结果通过 `torch.profiler.tensorboard_trace_handler` 输出到 TensorBoard

### 1.3 Warmup 策略

三阶段 schedule（PyTorch 标准模式）：
1. **Wait steps** — profiler 空闲期，不收集任何数据
2. **Warmup steps** — 开始收集但丢弃数据，消除冷启动噪声
3. **Active steps** — 正式记录的 profiling 数据

```python
torch.profiler.schedule(wait=..., warmup=..., active=...)
```

### 1.4 Nested Region 追踪

防止 profiler 重复开关的深度计数器：

```python
self._active_region_depth += 1
if self._active_region_depth == 1:
    self._set_collection(True)
```

### 1.5 统计方法

**无自建统计** — 所有分析委托 TensorBoard。通过 `key_averages().table(sort_by='self_cuda_time_total')` 获取 CUDA kernel 级别汇总。

### 1.6 Memory Tracking

通过 PyTorch profiler 的 `profile_memory` 参数启用，由环境变量控制。

**评价**: FastVideo 的 profiling 设计非常简约，适合训练/调优场景。但缺乏 per-step 精细 timing（如每个 denoising step 的 ms 级延迟），对实时推理 profiling 不够用。

---

## 2. TensorRT-LLM (NVIDIA)

**仓库**: `NVIDIA/TensorRT-LLM`
**关键文件**: `tensorrt_llm/profiler.py`, `tensorrt_llm/metrics/collector.py`, `tensorrt_llm/metrics/perf_utils.py`, `tensorrt_llm/metrics/enums.py`, `tensorrt_llm/bench/`

### 2.1 核心 Profiler (`profiler.py`)

```python
class Timer:
    _start_times: dict      # tag -> start_time
    _total_elapsed_times: dict  # tag -> accumulated time
    # 方法: start(tag), stop(tag), elapsed_time_in_sec(tag), reset(tag), summary()

class PyNVMLContext:
    # Context manager for pynvml init/shutdown
```

**Timing 机制**：CPU-only (`time.time()`)，没有 CUDA events。适合高层 profiling，不适合 kernel 级精度。

**Memory Tracking** — 双层监控：
- **Host**: `psutil` 的 USS (Unique Set Size)
- **GPU**: `pynvml` 库查询设备内存统计
- Graceful degradation：库不可用时返回零

### 2.2 Metrics Collector (Prometheus 集成)

`MetricsCollector` 类通过 Prometheus 暴露 40+ 指标：

**Per-Request Metrics (10 类)**:
| Metric | Description |
|--------|-------------|
| `TTFT` | Time to First Token |
| `TPOT` | Time per Output Token |
| `E2E` | End-to-end latency |
| `PREFILL_TIME` | `first_token - first_scheduled` |
| `DECODE_TIME` | `last_token - first_token` |
| `INFERENCE_TIME` | `last_token - first_scheduled` |
| `REQUEST_QUEUE_TIME` | `first_scheduled - arrival` |
| `PROMPT_TOKENS` | Input token count |
| `GENERATION_TOKENS` | Output token count |

**Iteration-Level Metrics (40+ 类)**:
- KV cache: hit rate, utilization, block counts, tokens-per-block
- Inflight batching: context/generation requests, paused, scheduled
- Speculative decoding: draft tokens, accepted tokens, acceptance length
- Memory: GPU/CPU/pinned memory 使用量 (bytes)

### 2.3 Phase 时间分解 (`perf_utils.py`)

四个关键时间戳：
```python
class RequestEventTiming(Enum):
    ARRIVAL_TIME          # 请求到达
    FIRST_SCHEDULED_TIME  # 调度器分配
    FIRST_TOKEN_TIME      # 首 token 生成
    LAST_TOKEN_TIME       # 末 token 生成
    KV_CACHE_TRANSFER_START  # KV cache 传输开始
    KV_CACHE_TRANSFER_END    # KV cache 传输结束
```

Phase 计算：
```
TTFT = first_token - arrival
Prefill = first_token - first_scheduled
Decode = last_token - first_token
Queue = first_scheduled - arrival
TPOT = (last_token - first_token) / (output_length - 1)
```

防御性验证：所有计算前验证时间戳有效性 (`> 0`)。

### 2.4 Benchmarking (`trtllm-bench`)

- **Warmup**: 显式 warmup 阶段，包括请求提交和完成周期；warmup 后重启 backend
- **统计**: P50, P90, P95, P99, MIN, MAX, AVERAGE
- **模式**: Offline batch submission（所有请求尽快提交）
- **Latency benchmark**: 单请求模式 (`--max_batch_size 1`)

### 2.5 CUDA Profiling 工具链

TensorRT-LLM 推荐使用 NVIDIA 外部工具：
- **Nsight Systems (nsys)** — GPU kernel timeline profiling
- **Nsight Compute (ncu)** — 单 kernel 深度分析
- 自身代码不包含 CUDA event timing

**评价**: TensorRT-LLM 的 profiling 最成熟，phase 分解最清晰 (TTFT/TPOT/Prefill/Decode)，Prometheus 集成适合生产环境监控。但其核心 timer 用 `time.time()` 而非 CUDA events，精度有限。

---

## 3. DeepSpeed (Microsoft)

**仓库**: `microsoft/DeepSpeed`
**关键文件**: `deepspeed/profiling/flops_profiler/profiler.py`, `deepspeed/utils/timer.py`, `deepspeed/inference/engine.py`, `deepspeed/monitor/monitor.py`

### 3.1 FlopsProfiler — FLOPs/MACs 计算器

核心能力：在模块级别计算计算量，支持训练和推理。

```python
class FlopsProfiler:
    def start_profile()   # 注册 hooks + monkey-patch 操作
    def stop_profile()    # 移除 patches 和 hooks
    def get_total_flops() / get_total_macs() / get_total_duration() / get_total_params()
    def print_model_profile()      # 详细报告
    def print_model_aggregated_profile()  # Top N 模块
```

**FLOPs 计算策略** — 按操作类型：
- **Linear**: `2 * input_elements * output_features`
- **Conv**: `output_dims * kernel_size * channels * positions`
- **MatMul**: `2 * product_of_input_shapes * last_dim`
- **Norm layers**: `input_elements * 4-5`
- **Einsum**: 通过 numpy optimization path 计算

**两层 Hook 系统**:
1. **Module-level hooks** (`register_module_hooks()`): pre-hook 初始化、post-hook 聚合
2. **Functional patching** (`_patch_functionals()`): monkey-patch `F.linear`, `F.conv` 等 + `tensor.matmul`, `tensor.add`

**配置方式** — 零代码 JSON 配置：
```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true
  }
}
```

### 3.2 SynchronizedWallClockTimer

```python
class SynchronizedWallClockTimer:
    class Timer:
        # 支持 host timing (time.time()) 和 device timing (CUDA events)
        # 通过 config.synchronized 控制是否 sync

class CudaEventTimer:
    # GPU event-based timing wrapper
    # accelerator.elapsed_time() 查询

class ThroughputTimer:
    # samples/sec 级别的吞吐量监控
    # 集成 psutil 的 memory_usage()

class NoopTimer:
    # Null-object pattern, timing 关闭时使用
```

**CUDA 同步处理**：
```python
if self.config.synchronized:
    get_accelerator().synchronize()
```

在 start/stop 前后条件性同步 GPU pipeline。

**统计方法** — `trim_mean()`:
- 排序后剔除两端 10% 数据，计算 trimmed mean
- 消除 GPU 执行的离群值噪声

### 3.3 Inference Engine Profiling

```python
def profile_model_time(self, use_cuda_events=True):
    # 注册 forward pre/post hooks
    # 两种模式: CUDA events 或 host sync

# 数据通过 self._model_times 列表收集
# 没有显式 memory tracking
```

### 3.4 分布式 Profiling

- `model_parallel_size * flops_per_gpu = total_flops`
- Data parallelism 不影响单 GPU metrics
- 通过 `mp_world_size` 缩放
- MoE 层特殊处理：检测 `deepspeed.moe.layer.MoE`

### 3.5 Monitor 系统

多后端监控路由：
- TensorBoard (`TensorBoardMonitor`)
- Weights & Biases (`WandbMonitor`)
- CSV (`csvMonitor`)
- Comet (`CometMonitor`)

Rank 0 only 写入，避免分布式环境下重复日志。

**评价**: DeepSpeed 的 FlopsProfiler 是独一无二的 — 它能计算理论 FLOPs/MACs，这对理解 GPU 利用率 (actual/theoretical) 很有价值。其 `trim_mean()` 统计方法和 dual-backend timer (CPU + CUDA events) 设计值得借鉴。

---

## 4. Triton Inference Server (NVIDIA)

**仓库**: `triton-inference-server/server`, `triton-inference-server/perf_analyzer`
**关键文档**: `docs/user_guide/metrics.md`

### 4.1 Prometheus Metrics 体系

**Inference Request Metrics**:
- Success/failure counters (by finish reason: REJECTED, CANCELED, BACKEND, OTHER)
- Inference count (batch-adjusted) + execution count
- Pending request count per model

**GPU Metrics** (via DCGM):
- Power: usage, limit, energy (joules)
- Utilization: 0.0-1.0 scale
- Memory: total + used (bytes)

**CPU Metrics** (Linux):
- System-wide utilization
- Total + used RAM

**Pinned Memory**:
- Pool total + used (bytes), 跨 model 聚合

### 4.2 Latency 三种度量模式

| Mode | 描述 | 配置 |
|------|------|------|
| **Counters** (默认) | 累计 queue/compute input/inference/output 时间 | 默认启用 |
| **Histograms** | TTFR (Time to First Response) + 可配置 buckets | `--metrics-config histogram_latencies=true` |
| **Summaries** | Quantile-based 滑动窗口分析 + 自定义百分位 | 需要 opt-in |

Endpoint: `http://localhost:8002/metrics` (Prometheus pull 模式)

### 4.3 Latency Breakdown

Server-side 三段分解：
1. **Queue**: 请求在调度队列中等待时间
2. **Compute**: 实际推理时间（含 GPU 数据拷贝）
3. **Overhead**: endpoint 中无法精确归类的时间

Client-side 额外计入 HTTP/gRPC 发送/接收 + marshalling 开销。

### 4.4 perf_analyzer 工具

**吞吐量**: `total_completed_requests / measurement_duration_seconds`

**稳定性检测**: 多次 measurement pass 直到结果稳定

**三种负载模式**:
1. **Concurrency Mode**: 维持固定并发数
2. **Request Rate Mode**: 固定请求发送速率
3. **Custom Interval Mode**: 自定义请求间隔

**Measurement Windows**:
- Time Windows: 固定时间窗口内计数（默认 5000ms）
- Count Windows: 动态增长窗口直到达到目标请求数（默认 50 requests）

**GPU Metrics 聚合**:
- Utilization/Power: 跨采集周期取平均
- Memory: 取最大值
- Total memory: 取第一次采集值

**统计**: P50, P90, P95, P99 + mean

### 4.5 Custom Metrics API

C API 暴露 `TRITONSERVER_MetricFamily*` 和 `TRITONSERVER_Metric*`，backend 可以注册自定义 metrics（TensorRT-LLM backend 用此追踪 KV cache 和 inflight batching）。

**评价**: Triton 的 profiling 是最"生产级"的 — Prometheus 集成、三种 latency 度量模式、perf_analyzer 的稳定性检测。但它是 serving-level profiling，不适合 model-internal profiling（不关心单层 latency）。

---

## 5. vLLM

**仓库**: `vllm-project/vllm`
**关键文件**: `vllm/profiler/layerwise_profile.py`, `vllm/profiler/wrapper.py`, `vllm/profiler/utils.py`

### 5.1 Layerwise Profiler

vLLM 实现了 **层级 profiling**，可以追踪每一层的 CUDA 时间占比：

```python
class LayerwiseProfileResults:
    _build_correlation_map()   # Kineto events -> profiler events 映射
    _build_module_tree()       # 构建模块层次结构
    _build_stats_trees()       # 生成 summary + model 统计树

class _ModuleTreeNode:  # 层级事件表示
class _StatsTreeNode:   # 统计树节点（Generic）
class SummaryStatsEntry: # 汇总统计: cuda_time_us, pct_cuda_time, invocations
class ModelStatsEntry:   # 单操作统计: cpu_time_us, cuda_time_us, stack traces
```

**Timing 机制**:
- **CUDA**: 从 Kineto GPU events 获取 `duration_ns()`，转换为 microseconds
- **CPU**: `node.event.duration_time_ns / 1000`
- 递归累加子节点 CUDA time

**Module 分类**: 基于 `event_module_repr()` 的模块命名约定，不显式区分 attention/MLP。

**统计**: 
- 相同 trace 的多次调用聚合：`cuda_time_us += ...; invocations += 1`
- 百分比计算：`(cuda_time_us / total_cuda_time) * 100`
- 输出格式：Table (带缩进层级) + CSV + JSON

### 5.2 Profiler Wrapper

```python
class WorkerProfiler:  # Abstract base
class TorchProfilerWrapper:  # PyTorch profiler 封装
class CudaProfilerWrapper:   # CUDA profiler 封装
```

**状态机生命周期**:
1. `start()` → 开始 profiling (可延迟)
2. `step()` → 每 iteration 更新状态
3. `stop()` → 停止
4. `shutdown()` → 清理

**配置** (via `ProfilerConfig`):
- `delay_iterations` — 延迟启动
- Warmup + max iterations
- Memory profiling, stack recording, FLOP counting
- Per-rank trace files

**评价**: vLLM 的 layerwise profiler 是最接近我们需求的 — 它能按模块层次追踪 CUDA time 占比。但它是 profiling 工具（生成 trace），不是 benchmarking 工具（不计算 E/P/D 分解）。

---

## 6. SGLang

**仓库**: `sgl-project/sglang`
**关键文件**: `python/sglang/srt/observability/` 目录

### 6.1 Observability 模块全景

```
observability/
├── cpu_monitor.py              # CPU 使用监控
├── func_timer.py               # 函数级别计时装饰器
├── label_transform.py          # 标签转换
├── metrics_collector.py        # Prometheus metrics 收集
├── req_time_stats.py           # 请求级别时间统计 (最重要)
├── request_metrics_exporter.py # metrics 导出
├── scheduler_metrics_mixin.py  # 调度器 metrics
├── startup_func_log_and_timer.py # 启动计时
├── trace.py                    # Distributed tracing
└── utils.py
```

### 6.2 Request Time Stats (核心创新)

SGLang 实现了最精细的 **请求级别 phase tracking**：

**三级 timing 层次**:
- **Level 1 (原子操作)**: `tokenize`, `prefill_forward`, `decode_forward`
- **Level 2 (调度阶段)**: `api_server_dispatch`, `request_process`
- **Level 3 (复合)**: `decode_loop`, `prefill_chunked_forward`

**三种执行模式下的 phase 分解**:
```
Unified:  wait_queue → forward → completion
Prefill disagg: bootstrap_queue → wait_queue → forward → transfer_queue
Decode disagg:  prealloc_queue → transfer_queue → wait_queue → forward
```

**时间基准优化**:
- 使用 "calibrated `perf_counter()` values" 减少系统调用
- `convert_time_cross_thread()` 处理跨进程 monotonic clock drift
- Trace 集成：nanosecond 精度的 span 发射

### 6.3 Function Timer

```python
@time_func_latency(name="optional_name")
async def my_function():
    pass
```

- 基于 `time.monotonic()`，不用 CUDA events
- 支持 async/sync 两种函数
- 结果 feed 到 Prometheus Histogram (50ms ~ 50s bucket range)

### 6.4 Prometheus Metrics 体系

**Gauges**: `num_running_reqs`, `num_queue_reqs`, `token_usage`, `kv_available_tokens`, `utilization`, `cache_hit_rate`

**Counters**: `prompt_tokens_total`, `generation_tokens_total`, `num_requests_total`, `cuda_graph_passes_total`

**Histograms**: `queue_time_seconds`, `per_stage_req_latency_seconds`, `kv_transfer_latency_ms`, `kv_transfer_speed_gb_s`

还追踪 estimated FLOPs 和 memory bandwidth per GPU。

**评价**: SGLang 的 observability 是最全面的。其三级 phase tracking + disaggregation-aware timing 是最先进的设计。特别是 `req_time_stats.py` 中对 prefill/decode disaggregation 的 timing 追踪，直接对应我们的研究方向。

---

## 7. llama.cpp

**仓库**: `ggml-org/llama.cpp`
**关键文件**: `include/llama.h` (API 声明), `common/common.h` (`common_time_meas`)

### 7.1 Performance 数据结构

```c
struct llama_perf_context_data {
    double t_start_ms;    // 绝对开始时间
    double t_load_ms;     // 模型加载耗时
    double t_p_eval_ms;   // prompt 处理时间 (Prefill)
    double t_eval_ms;     // token 生成时间 (Decode)
    int32_t n_p_eval;     // prompt token 数
    int32_t n_eval;       // 生成 token 数
    int32_t n_reused;     // graph reuse 次数
};

struct llama_perf_sampler_data {
    double t_sample_ms;   // sampling 耗时
    int32_t n_sample;     // 采样 token 数
};
```

### 7.2 API 函数

```c
llama_perf_context()        // 获取当前 context 性能数据
llama_perf_context_print()  // 格式化输出
llama_perf_context_reset()  // 清零
llama_perf_sampler()        // 获取 sampler 性能
llama_perf_sampler_print()  // 格式化输出
llama_perf_sampler_reset()  // 清零
llama_memory_breakdown_print()  // Per-device memory 分解
```

### 7.3 RAII 计时器

```cpp
struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();
    const int64_t t_start_us;  // 微秒级开始时间
    int64_t & t_acc;           // 累加器引用
};
```

RAII pattern：构造时记录 start，析构时自动累加到 accumulator。非常简洁。

### 7.4 Derived Metrics

```
prompt_eval_speed = n_p_eval / t_p_eval_ms * 1000  // tokens/sec (prefill)
eval_speed = n_eval / t_eval_ms * 1000              // tokens/sec (decode)
sample_speed = n_sample / t_sample_ms * 1000        // tokens/sec (sampling)
```

**评价**: llama.cpp 的 profiling 是最简洁的 — 直接追踪 prefill (`t_p_eval_ms`) 和 decode (`t_eval_ms`) 两个核心 phase，加上 sampling 时间。RAII timer 设计非常优雅。没有 CUDA events（因为 llama.cpp 使用 ggml 后端）。

---

## 8. MLC LLM

**仓库**: `mlc-ai/mlc-llm`
**关键文件**: `python/mlc_llm/bench/request_record.py`, `python/mlc_llm/bench/api_endpoint.py`

### 8.1 Request-Level Metrics

```python
class Metrics:
    ttft: float              # Time to First Token
    tpot: float              # Time per Output Token
    inter_token_latency: float  # Token 间延迟
    e2e_latency: float       # End-to-end 延迟
    input_tokens: int
    output_tokens: int
```

### 8.2 Timing 机制

- Client-side: `time.monotonic()` 包裹 HTTP 请求
- Server-side: 从 API response 提取 `prefill_tokens_per_s`, `inter_token_latency_s`

### 8.3 统计方法

```python
# _compute_metrics_statistics()
- Percentiles: P25, P50, P75, P90, P95, P99
- Descriptive: mean, std, min, max
- Throughput: requests/sec, tokens/sec/GPU
- Pandas DataFrame 转换
```

**评价**: MLC LLM 的 bench 模块是一个 serving-level benchmark client，类似 perf_analyzer。它区分 client-side 和 server-side metrics，统计方法最全面。

---

## 横向比较

### Timing 机制对比

| System | CUDA Events | CPU Timer | PyTorch Profiler | 精度 |
|--------|:-----------:|:---------:|:----------------:|:----:|
| FastVideo | - | - | Yes | kernel-level (via trace) |
| TensorRT-LLM | - | `time.time()` | - | ~ms |
| DeepSpeed | Yes (可选) | `time.time()` | - | sub-ms (CUDA mode) |
| Triton | N/A (server) | N/A | N/A | request-level |
| vLLM | Yes (Kineto) | Yes | Yes | kernel-level |
| SGLang | - | `time.monotonic()` | - | ~ms |
| llama.cpp | - | `int64_t μs` | N/A | ~μs |
| MLC LLM | - | `time.monotonic()` | - | ~ms |
| **VLLA (ours)** | **Yes** | **`perf_counter()`** | - | **sub-ms** |

### Phase 定义对比

| System | Phases | Granularity |
|--------|--------|-------------|
| TensorRT-LLM | Queue → Prefill → Decode | Per-request |
| SGLang | tokenize → queue → prefill_forward → decode_forward | Per-request, 3-level |
| llama.cpp | load → prompt_eval → eval → sample | Per-session |
| DeepSpeed | forward → backward → step | Per-iteration |
| vLLM | Per-layer CUDA time breakdown | Per-module |
| **VLLA (ours)** | **Encode → Prefill → Decode** | **Per-phase, accumulative** |

### Warmup 策略对比

| System | Approach |
|--------|----------|
| FastVideo | PyTorch profiler schedule (wait/warmup/active) |
| TensorRT-LLM | 显式 warmup 请求 + backend 重启 |
| DeepSpeed | `warm_up=N` 参数跳过前 N iterations |
| Triton perf_analyzer | Stabilization passes 直到结果稳定 |
| vLLM | `delay_iterations` + warmup steps |
| llama.cpp | 无显式 warmup |
| **VLLA (ours)** | **无显式 warmup (待改进)** |

### 统计方法对比

| System | Methods |
|--------|---------|
| TensorRT-LLM | P50/P90/P95/P99, MIN, MAX, AVERAGE |
| DeepSpeed | `trim_mean()` (10% trim), 分布式聚合 |
| Triton | Histograms, Summaries (quantile), 稳定性检测 |
| MLC LLM | P25/P50/P75/P90/P95/P99, mean, std, min, max |
| SGLang | Prometheus Histograms with configurable buckets |
| llama.cpp | Simple tokens/sec |
| **VLLA (ours)** | **单次运行 (待改进)** |

### Memory Tracking 对比

| System | Approach |
|--------|----------|
| TensorRT-LLM | `psutil` USS + `pynvml` GPU memory |
| DeepSpeed | `psutil` + `accelerator.memory_usage()` + parameter counting |
| Triton | DCGM (GPU), `/proc/meminfo` (CPU), pinned memory pool |
| vLLM | PyTorch profiler `profile_memory` |
| llama.cpp | `llama_memory_breakdown_print()` per-device |
| **VLLA (ours)** | **无 (待添加)** |

---

## 对 VLLA Framework 的设计启示

### 已有优势
1. **CUDA Events** — 我们的 `PhaseTimer` 使用 CUDA events，比 TensorRT-LLM (`time.time()`) 和 SGLang (`time.monotonic()`) 更精确
2. **Phase 分解** — Encode/Prefill/Decode 三阶段分解清晰
3. **Immutable patterns** — `self._active = {**self._active, phase: backend}` 遵循不可变原则

### 建议改进

#### P0: 必须添加
1. **Warmup 策略** — 参考 DeepSpeed 的 `warm_up=N` 参数或 TensorRT-LLM 的显式 warmup 阶段
2. **Memory Tracking** — 参考 TensorRT-LLM 的 `psutil` + `pynvml` 双层方案
3. **多次运行统计** — 参考 DeepSpeed 的 `trim_mean()`，至少支持 mean + std + percentiles

#### P1: 应该添加
4. **Per-layer Profiling** — 参考 vLLM 的 `LayerwiseProfileResults`，追踪 VLM 各模块 (vision encoder / LLM attention / MLP) 的 CUDA time 占比
5. **Prometheus Metrics 导出** — 参考 SGLang/TensorRT-LLM 的 Prometheus 集成，支持实验结果自动采集

#### P2: 可以考虑
6. **FLOPs 估算** — 参考 DeepSpeed FlopsProfiler，计算 theoretical vs actual GPU utilization
7. **Disaggregated Phase Tracking** — 参考 SGLang 的三级 timing 层次，支持 prefill/decode disaggregation 的精细 timing
8. **PyTorch Profiler 集成** — 参考 FastVideo/vLLM，在需要 kernel-level 分析时无缝切换到 torch.profiler

---

## 参考源

1. `hao-ai-lab/FastVideo` — `fastvideo/profiler.py`, `fastvideo/pipelines/composed_pipeline_base.py`
2. `NVIDIA/TensorRT-LLM` — `tensorrt_llm/profiler.py`, `tensorrt_llm/metrics/{collector,perf_utils,enums}.py`, `tensorrt_llm/bench/`
3. `microsoft/DeepSpeed` — `deepspeed/profiling/flops_profiler/profiler.py`, `deepspeed/utils/timer.py`, `deepspeed/inference/engine.py`, `deepspeed/monitor/monitor.py`
4. `triton-inference-server/server` — `docs/user_guide/metrics.md`; `triton-inference-server/perf_analyzer`
5. `vllm-project/vllm` — `vllm/profiler/{layerwise_profile,wrapper,utils}.py`
6. `sgl-project/sglang` — `python/sglang/srt/observability/{metrics_collector,func_timer,req_time_stats}.py`
7. `ggml-org/llama.cpp` — `include/llama.h` (`llama_perf_context_data`, `llama_perf_sampler_data`)
8. `mlc-ai/mlc-llm` — `python/mlc_llm/bench/{request_record,api_endpoint}.py`
9. DeepSpeed FlopsProfiler tutorial — https://www.deepspeed.ai/tutorials/flops-profiler/
10. Triton Inference Server metrics docs — NVIDIA official documentation
11. TensorRT-LLM benchmarking docs — NVIDIA official documentation
