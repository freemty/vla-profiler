# SGLang Profiling & Timing 实现深度调研

> 调研时间: 2026-04-15
> 目的: 为 VLM profiling framework 设计提供参考，理解 SGLang 如何在高性能 serving 框架中集成 profiling 能力

## 1. 整体架构概览

SGLang 的 profiling 和 observability 系统分布在以下几个层级：

```
┌─────────────────────────────────────────────────┐
│  HTTP API 层 (/start_profile, /stop_profile)     │
├─────────────────────────────────────────────────┤
│  Engine 层 (engine.py → tokenizer_manager)       │
├─────────────────────────────────────────────────┤
│  Scheduler 层 (scheduler.py + mixins)            │
│  ├── SchedulerProfilerMixin   → torch profiler   │
│  ├── SchedulerMetricsMixin    → Prometheus 指标   │
│  └── SchedulerOutputProcessorMixin → 结果统计     │
├─────────────────────────────────────────────────┤
│  Model Executor 层 (model_runner.py)             │
│  ├── NVTX hooks (hook_manager.py)                │
│  ├── FlashInfer autotune                         │
│  └── Warmup / dummy run                          │
├─────────────────────────────────────────────────┤
│  Observability 模块 (python/sglang/srt/observability/) │
│  ├── req_time_stats.py    → 请求级时间追踪         │
│  ├── metrics_collector.py → Prometheus 采集器      │
│  ├── func_timer.py        → 函数级延迟装饰器       │
│  ├── trace.py             → OpenTelemetry 分布式追踪│
│  ├── cpu_monitor.py       → CPU 利用率监控         │
│  └── startup_func_log_and_timer.py → 启动阶段计时  │
├─────────────────────────────────────────────────┤
│  Benchmark 套件 (bench_serving, bench_one_batch 等)│
└─────────────────────────────────────────────────┘
```

**关键设计理念**: SGLang 将"运行时 profiling"和"离线 benchmark"严格分离。运行时通过 Prometheus metrics + OpenTelemetry trace 提供 observability；离线通过 `bench_*` 工具和 `torch.profiler` 提供 kernel 级分析。

---

## 2. Model Executor Profiling (model_runner.py)

**文件**: `python/sglang/srt/model_executor/model_runner.py`

### 2.1 初始化阶段计时

使用 `time.perf_counter()` 测量关键初始化步骤：

```python
# 分布式初始化
tic = time.perf_counter()
# ... init torch distributed ...
elapsed = time.perf_counter() - tic
logger.info(f"elapsed={elapsed:.2f} s, mem usage={...:.2f} GB")

# 模型加载
tic_total = time.perf_counter()
before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
# ... load model ...
after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
self.weight_load_mem_usage = before_avail_memory - after_avail_memory
```

### 2.2 NCCL/RCCL Warmup 测量

```python
warmup_start = time.perf_counter()
dist.all_reduce(warmup_tensor, group=tp_group_handle)
torch.cuda.synchronize()
warmup_elapsed = time.perf_counter() - warmup_start
logger.info(f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s")
```

### 2.3 FlashInfer Autotune

在非默认 CUDA stream 上执行 autotune，避免 NCCL 冲突：
```python
with autotune():
    self._dummy_run(batch_size=...)
```

### 2.4 NVTX Layer-wise Profiling Hooks

**文件**: `python/sglang/srt/model_executor/hook_manager.py`

通过 `register_forward_hooks()` 函数注册 PyTorch forward hooks：
- 使用 `fnmatch` 通配符匹配 module name
- 动态导入 hook factory，注册到匹配的 module
- 启用条件: `--enable-layerwise-nvtx-marker` server flag
- 需要配合 `--disable-cuda-graph` 使用（CUDA graph 会跳过 hooks）

```python
# model_runner.py 中
self.pyt_hooks = PytHooks()
self.pyt_hooks.register_hooks(self.model, module_prefix="model")
```

### 2.5 show_time_cost 全局计时

`enable_show_time_cost()` 是一个全局开关，激活后在各处添加计时日志。通过 `--show-time-cost` server arg 控制。

**关键发现**: Model executor 层**不在 hot path 中嵌入显式计时**。运行时 forward pass 的 profiling 完全委托给 `torch.profiler` 和 NVTX markers，这是为了避免影响推理性能。

---

## 3. torch.profiler 集成 (SchedulerProfilerMixin)

**文件**: `python/sglang/srt/managers/scheduler_profiler_mixin.py`

这是 SGLang 中最核心的 profiling 实现，设计为 Scheduler 的 Mixin 类。

### 3.1 架构设计

```python
class SchedulerProfilerMixin:
    """Scheduler 的 profiling mixin，通过多继承注入到 Scheduler 中"""
```

支持两种模式:
- **V1 (legacy)**: 直接在 mixin 中管理 `torch.profiler.profile()` 生命周期
- **V2 (新版)**: 通过 `ProfileManager` 类管理，支持更复杂的 stage-based 触发

### 3.2 Profiler 初始化 (init_profile)

```python
def init_profile(
    self,
    output_dir: Optional[str],          # 输出目录，默认 /tmp 或 SGLANG_TORCH_PROFILER_DIR
    start_step: Optional[int],           # 延迟启动（第N步开始）
    num_steps: Optional[int],            # 采集步数
    activities: Optional[List[str]],     # ["CPU", "GPU", "MEM", "CUDA_PROFILER", "RPD"]
    with_stack: Optional[bool],          # 采集 Python stack trace
    record_shapes: Optional[bool],       # 记录 tensor shape
    profile_by_stage: bool,              # 按 prefill/decode 阶段分别采集
    profile_id: str,                     # 唯一标识符
    merge_profiles: bool,                # 是否合并分布式 trace
    profile_prefix: str,                 # 文件名前缀
    profile_stages: Optional[List[str]], # 指定要 profile 的阶段
) -> ProfileReqOutput
```

### 3.3 Profiler 启动 (start_profile)

支持五种 profiling 活动：

| Activity | 实现 | 用途 |
|----------|------|------|
| `CPU` | `torch.profiler.ProfilerActivity.CPU` | CPU 算子分析 |
| `GPU` | `torch.profiler.ProfilerActivity.CUDA` | GPU kernel 分析 |
| `RPD` | `rpdTracerControl` (ROCm) | AMD GPU profiling |
| `MEM` | `torch.cuda.memory._record_memory_history()` | GPU 内存分配追踪 |
| `CUDA_PROFILER` | `torch.cuda.cudart().cudaProfilerStart()` | Nsight 外部 profiler 控制 |

```python
# GPU/CPU profiling
self.torch_profiler = torch.profiler.profile(
    activities=torchprof_activities,
    with_stack=True,
    record_shapes=False,
    on_trace_ready=None,  # NPU 用 tensorboard_trace_handler
)
self.torch_profiler.start()
```

### 3.4 Stage-based Profiling (prefill/decode 分离采集)

**这是一个非常精巧的设计**: `_profile_batch_predicate()` 在每次 forward pass 前被调用，根据 `batch.forward_mode` 自动控制 profiler 的 start/stop：

```python
def _profile_batch_predicate(self, batch):
    if self.profile_by_stage:
        if batch.forward_mode.is_prefill():
            if self.profiler_prefill_ct == 0:
                self.start_profile(batch.forward_mode)
            self.profiler_prefill_ct += 1
            if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                self.stop_profile(stage=ForwardMode.EXTEND)
        elif batch.forward_mode.is_decode():
            if self.profiler_decode_ct == 0:
                self.stop_profile(stage=ForwardMode.EXTEND)  # 先停 prefill
                self.start_profile(batch.forward_mode)       # 再启 decode
            self.profiler_decode_ct += 1
            if self.profiler_decode_ct > self.profiler_target_decode_ct:
                self.stop_profile(stage=ForwardMode.DECODE)
```

这样可以生成独立的 prefill trace 和 decode trace 文件。

### 3.5 Trace 导出与合并

输出文件命名: `{prefix}-{profile_id}-TP-{tp_rank}[-DP-{dp_rank}][-PP-{pp_rank}][-EP-{ep_rank}]-{stage}.trace.json.gz`

分布式 trace 合并:
```python
def _merge_profile_traces(self):
    merger = ProfileMerger(self.torch_profiler_output_dir, self.profile_id)
    merged_path = merger.merge_chrome_traces()
    summary = merger.get_merge_summary()  # {total_events, total_files}
```

### 3.6 V2 ProfileManager (profile_utils.py)

**文件**: `python/sglang/srt/utils/profile_utils.py`

V2 使用更模块化的设计：

```
ProfileManager
  ├── _StageBasedTrigger   → 状态机控制 prefill/decode 采集
  ├── _ProfilerTorch       → torch.profiler.profile
  ├── _ProfilerMemory      → GPU memory history
  ├── _ProfilerCudart      → CUDA profiler API
  └── _ProfilerRPD         → ROCm profiling
```

`_StageBasedTrigger` 是一个状态机，跟踪当前 stage (prefill/decode)，在 stage 转换时触发 start/stop callbacks，当 step count 超过配置阈值时自动停止。

---

## 4. Benchmark 套件

SGLang 提供四级 benchmark 工具，从最接近真实场景到最底层 kernel：

### 4.1 bench_serving (在线服务基准测试)

**文件**: `python/sglang/bench_serving.py`

这是最常用的 benchmark 工具，模拟真实在线服务场景。

**指标定义**:
| 指标 | 计算方式 |
|------|---------|
| TTFT | `timestamp_first_token - start_time` |
| TPOT | `(total_latency - ttft) / (output_tokens - 1)` |
| ITL | `timestamp_current - timestamp_previous` (相邻 token 间隔) |
| E2E | `start_time + total_latency` |
| Throughput | `completed_requests / duration` |
| Concurrency | `sum(e2e_latencies) / benchmark_duration` |

**统计方法**:
```python
metrics = BenchmarkMetrics(
    mean_ttft_ms=np.mean(ttfts) * 1000,
    median_ttft_ms=np.median(ttfts) * 1000,
    std_ttft_ms=np.std(ttfts) * 1000,
    p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
    p95_itl_ms=np.percentile(itls, 95) * 1000,
    p90_e2e_ms=np.percentile(e2e_latencies, 90) * 1000,
    # ... 同样应用于 ITL, TPOT, E2E
)
```

**Warmup 策略**: 默认 1 个 warmup request，使用数据集第一条数据，可选 flush cache。

**请求生成**: Poisson 分布 (`np.random.exponential(1.0 / rate)`) 控制请求间隔，支持 trace timestamp 回放模式。

**并发控制**: `asyncio.Semaphore(max_concurrency)` 限制最大并发数。

**注意**: 没有显式 outlier removal，直接用 numpy 的 mean/median/percentile。

### 4.2 bench_one_batch (单 batch kernel 级 profiling)

**文件**: `python/sglang/bench_one_batch.py`

直接操作 `ModelRunner`，绕过 Scheduler 和 HTTP 层。

**计时方法**:
```python
model_runner.synchronize()                    # GPU sync
tic = time.perf_counter()
next_token_ids, _, batch = model_runner.extend(reqs)  # prefill
model_runner.synchronize()                    # GPU sync
prefill_latency = time.perf_counter() - tic

# Decode: 每个 token 独立计时
for i in range(output_len - 1):
    tic = time.perf_counter()
    next_token_ids, _ = model_runner.decode(next_token_ids, batch)
    latency = time.perf_counter() - tic
    decode_latencies.append(latency)
```

**Warmup**: 1 次 warmup run，使用缩短的 output_len (`min(32, output_len)`)。

**Profiler 集成**: 可选在 prefill 和 decode 阶段分别启用 torch.profiler，生成独立 trace 文件。

**容量检查**: `max_batch_size = max_total_num_tokens // (input_len + output_len)`

### 4.3 bench_offline_throughput

直接使用 `Engine` 类，绕过 HTTP 层但保留 Scheduler。测量最大吞吐量。

### 4.4 bench_one_batch_server

通过 HTTP 对运行中的 server 发送单 batch 请求，测量端到端延迟。

---

## 5. 请求级时间追踪 (req_time_stats.py)

**文件**: `python/sglang/srt/observability/req_time_stats.py`

### 5.1 时间校准

```python
def calibrate_time_diff():
    """校准 monotonic clock 和 wall clock 的差值"""
    global global_diff_realtime_monotonic
    global_diff_realtime_monotonic = time.time() - time.perf_counter()

def convert_time_to_realtime(time_value: float) -> float:
    """将 perf_counter 值转换为 wall clock time"""
    return time_value + global_diff_realtime_monotonic
```

**设计理由**: 内部用 `time.perf_counter()` (monotonic) 做高精度计时，跨进程序列化时转为 wall clock time。

### 5.2 请求阶段定义

三级阶段划分：
- **Level 1 (原子操作)**: TOKENIZE, PREFILL_FORWARD, DECODE_FORWARD
- **Level 2 (调度阶段)**: API_SERVER_DISPATCH, REQUEST_PROCESS
- **Level 3 (组合阶段)**: DECODE_LOOP, PREFILL_CHUNKED_FORWARD

### 5.3 APIServerReqTimeStats

API 层的时间追踪：
```python
class APIServerReqTimeStats:
    created_time          # 请求创建时间
    finished_time         # 请求完成时间
    first_token_time      # 第一个 token 到达时间
    tokenize_finish_time  # tokenize 完成时间
    api_server_dispatch_time
    response_sent_to_client_time

    def get_first_token_latency(self):  # TTFT
        return self.first_token_time - self.created_time

    def get_e2e_latency(self):
        return self.finished_time - self.created_time

    def get_decode_latency(self):
        return self.finished_time - self.first_token_time
```

### 5.4 SchedulerReqTimeStats

Scheduler 层更详细的追踪，支持 PD 分离部署：
```python
class SchedulerReqTimeStats:
    wait_queue_entry_time      # 进入等待队列时间
    forward_entry_time         # 进入 forward batch 时间
    prefill_run_batch_start_time
    prefill_run_batch_end_time
    # PD 分离相关
    bootstrap_queue_entry_time
    transfer_start_time
    prealloc_start_time
    kv_transfer_speed          # KV cache 传输速度
    kv_transfer_total_mb       # 传输量

    def get_queueing_time(self):
        return self.forward_entry_time - self.wait_queue_entry_time
```

---

## 6. Prometheus Metrics (metrics_collector.py)

**文件**: `python/sglang/srt/observability/metrics_collector.py`

### 6.1 SchedulerMetricsCollector

**Gauge 指标** (瞬时值):
| 指标 | 含义 |
|------|------|
| `sglang:num_running_reqs` | 运行中请求数 |
| `sglang:num_queue_reqs` | 队列中请求数 |
| `sglang:token_usage` | Token pool 利用率 |
| `sglang:gen_throughput` | 生成吞吐量 (tokens/sec) |
| `sglang:cache_hit_rate` | Prefix cache 命中率 |
| `sglang:utilization` | 系统利用率 |
| `sglang:kv_available_tokens` | 可用 KV cache slots |
| `sglang:kv_used_tokens` | 已用 KV cache slots |
| `sglang:kv_evictable_tokens` | 可驱逐的 KV cache slots |

**Histogram 指标** (分布):
| 指标 | Buckets |
|------|---------|
| `sglang:queue_time_seconds` | 36 个 bucket (0ms → 1000s) |
| `sglang:kv_transfer_speed_gb_s` | 0.1 → 400 GB/s |
| `sglang:per_stage_req_latency_seconds` | 指数 bucket |

**Counter 指标** (累计):
| 指标 | 含义 |
|------|------|
| `sglang:num_retracted_requests_total` | 回撤请求总数 |
| `sglang:cuda_graph_passes_total` | CUDA graph 执行次数 |
| `sglang:realtime_tokens_total` | 实时 prefill/decode token 数 |

### 6.2 TokenizerMetricsCollector

| 指标 | 类型 | Buckets/说明 |
|------|------|-------------|
| `sglang:time_to_first_token_seconds` | Histogram | 18 buckets (0.1s → 400s) |
| `sglang:inter_token_latency_seconds` | Histogram | 23 buckets (2ms → 8s) |
| `sglang:e2e_request_latency_seconds` | Histogram | 22 buckets (0.1s → 2400s) |
| `sglang:prompt_tokens_total` | Counter | prefill tokens |
| `sglang:generation_tokens_total` | Counter | decode tokens |
| `sglang:cached_tokens_total` | Counter | 按来源分: device/host/storage |

### 6.3 Bucket 生成工具

```python
def exponential_buckets(start, width, length):
    """指数增长 bucket: start * width^i"""

def two_sides_exponential_buckets(middle, base, count):
    """双侧指数 bucket: 以 middle 为中心向两侧扩展"""

def generate_buckets(buckets_rule, default_buckets):
    """支持三种模式: 'default', 'tse' (two-sided exponential), 'custom'"""
```

用户可通过 `--bucket-time-to-first-token`, `--bucket-inter-token-latency`, `--bucket-e2e-request-latency` 自定义 histogram bucket。

### 6.4 条件性指标

根据功能开关动态创建:
- LoRA pool 利用率 (当 `enable_lora=True`)
- HiCache host-tier 指标 (当 hierarchical cache 启用)
- Streaming session 指标
- Priority scheduling 指标 (动态 label 生成)

---

## 7. 函数级计时 (func_timer.py)

**文件**: `python/sglang/srt/observability/func_timer.py`

一个优雅的 decorator，同时支持同步和异步函数:

```python
@time_func_latency(name="my_operation")
async def my_operation():
    ...

@time_func_latency
def sync_operation():
    ...
```

内部实现:
- 使用 `time.monotonic()` 计时
- 输出到 Prometheus Histogram: `sglang:func_latency_seconds`
- 18 个指数 bucket (start=0.05, width=1.5)
- 通过 `enable_func_timer()` 全局开关控制
- 未启用时零开销 (early return)

---

## 8. OpenTelemetry 分布式追踪 (trace.py)

**文件**: `python/sglang/srt/observability/trace.py`

### 8.1 核心组件

| 组件 | 功能 |
|------|------|
| `process_tracing_init()` | 初始化 OTLP 端点 |
| `TraceReqContext` | 请求级追踪上下文，支持序列化 |
| `TraceThreadContext` | 线程级 span 追踪 |
| `TraceSliceContext` | 单操作 span |
| `TraceCustomIdGenerator` | 防跨进程 ID 碰撞 |

### 8.2 API

```python
trace_req_start() / trace_req_finish()  # 请求生命周期
trace_slice_start() / trace_slice_end()  # 操作计时
trace_event()                            # 事件记录
```

### 8.3 Span 属性 (SpanAttributes)

```python
GEN_AI_USAGE_COMPLETION_TOKENS
GEN_AI_REQUEST_TOP_P
GEN_AI_REQUEST_TEMPERATURE
GEN_AI_LATENCY_E2E
GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN
```

---

## 9. 内存 Profiling

### 9.1 KV Cache Memory Pool (memory_pool.py)

**文件**: `python/sglang/srt/mem_cache/memory_pool.py`

核心类:
```python
class ReqToTokenPool:
    def available_size(self) -> int    # 可用 slot 数
    def alloc(self, reqs) -> Optional[List[int]]
    def free(self, req)

class KVCache(abc.ABC):  # 抽象基类
    def get_kv_size_bytes(self) -> Union[int, Tuple[int, int]]  # K/V 大小

class MHATokenToKVPool(KVCache):
    def get_contiguous_buf_infos(self)  # RDMA 注册信息

class MLATokenToKVPool(KVCache):
    def get_kv_size_bytes(self) -> int  # 合并 K/V layout
```

Memory tracking 方式:
- `get_available_gpu_memory(device, gpu_id)` — 初始化时前后对比
- `available_size()` — 运行时 free slot 计数
- `TorchMemorySaverAdapter.region(GPU_MEMORY_TYPE_KV_CACHE)` — 可选 CPU offload

### 9.2 Radix Cache Metrics (radix_cache.py)

**文件**: `python/sglang/srt/mem_cache/radix_cache.py`

```python
class RadixCache:
    def evictable_size(self)     # 可驱逐大小
    def protected_size(self)     # 被锁定的 cache 大小
    def total_size(self)
    def update_eviction_metrics(self, num_evicted, start_time)
    def take_events(self)        # 原子取出事件队列
```

TreeNode 级指标:
- `hit_count`: 访问频率
- `last_access_time`: LRU 排序
- `creation_time`: 节点创建时间
- `lock_ref`: 引用计数保护

对应 Prometheus 指标:
| 指标 | 类型 |
|------|------|
| `sglang:evicted_tokens_total` | Counter |
| `sglang:load_back_tokens_total` | Counter |
| `sglang:eviction_duration_seconds` | Histogram |
| `sglang:load_back_duration_seconds` | Histogram |

### 9.3 GPU Memory Snapshot

通过 `torch.profiler` 的 `MEM` activity:
```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 运行推理 ...
torch.cuda.memory._dump_snapshot("memory.pickle")
```

---

## 10. Scheduler 中的 Prefill/Decode 分离

**文件**: `python/sglang/srt/managers/scheduler.py`

### 10.1 调度逻辑

```python
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    ...
):
    def get_next_batch_to_run(self)          # 主调度入口
    def get_new_batch_prefill(self)          # 获取 prefill batch
    def _get_new_batch_prefill_raw(self)     # 核心 prefill batch 组装
    def update_running_batch(self)           # 更新 decode batch 状态
```

### 10.2 Chunked Prefill

```python
self.chunked_req = None                     # 正在处理的多 chunk 请求
self.chunked_prefill_size = server_args.chunked_prefill_size
self.enable_dynamic_chunking = (
    server_args.enable_dynamic_chunking and pp_size > 1
)
```

`profile_and_init_predictor()` 方法用于动态 chunk 大小预测。

### 10.3 Forward Mode 区分

```python
new_batch_prefill = batch.forward_mode.is_extend()
is_mixed_chunk  # 混合 prefill-decode batch

# ForwardMode 枚举
ForwardMode.EXTEND   → prefill
ForwardMode.DECODE   → decode
ForwardMode.IDLE     → 空闲
```

### 10.4 Overlap Scheduling

```python
self.enable_overlap = not server_args.disable_overlap_schedule
self.result_queue = deque()  # 存储 (batch, result) 对

def event_loop_normal(self)    # 顺序: CPU→GPU→CPU
def event_loop_overlap(self)   # 重叠: 前一轮 GPU 与当前轮 CPU 并行
```

### 10.5 性能估算 (SchedulerMetricsMixin)

```python
def _estimate_prefill_perf(self, num_tokens):
    return (FLOPs, read_bytes, write_bytes)

def _estimate_decode_perf(self, batch, num_tokens):
    return (FLOPs, read_bytes, write_bytes)

# Context managers
def record_forward_metrics(self, batch):  # 包裹 forward pass
def record_bubble_metrics(self, batch):   # 追踪 GPU 空闲时间
```

---

## 11. Nsight Systems 集成

SGLang 支持两种 Nsight 集成方式：

### 11.1 外部 Nsight 包裹

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64
```

### 11.2 CUDA Profiler API 控制

通过 HTTP API 精确控制 Nsight 采集区间:
```bash
# 启动 server 时
python -m sglang.launch_server ... --enable-layerwise-nvtx-marker --disable-cuda-graph

# 运行时通过 Nsight 包裹 + CUDA_PROFILER API
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...

# 然后通过 HTTP 触发采集
curl -X POST http://localhost:30000/start_profile \
  -d '{"activities": ["CUDA_PROFILER"]}'
```

内部实现:
```python
if "CUDA_PROFILER" in activities:
    if self.gpu_id == get_global_server_args().base_gpu_id:
        torch.cuda.cudart().cudaProfilerStart()
```

---

## 12. 对我们 VLM Profiling Framework 的启示

### 12.1 值得借鉴的设计

1. **Hot path 零开销原则**: 运行时 forward pass 不嵌入计时代码，profiling 完全通过 torch.profiler 和 NVTX markers 实现
2. **Stage-based profiling**: `_profile_batch_predicate()` 自动根据 ForwardMode 切分 prefill/decode trace，这正是我们需要的 E/P/D 分离
3. **Mixin 架构**: Profiler、Metrics、OutputProcessor 各自独立为 Mixin，松耦合
4. **时间校准**: `perf_counter()` 内部计时 + `time.time()` 跨进程序列化的双轨设计
5. **Prometheus histogram bucket 可配置**: 用户可自定义 TTFT/ITL/E2E 的 bucket 分布
6. **多级 benchmark 工具**: 从 kernel 级到在线服务级，四种工具覆盖不同需求

### 12.2 我们框架的差异化

1. **我们关注 VLM 特有的 Encode 阶段**: SGLang 没有显式区分 vision encoder 的时间，而这是我们 exp01a 的核心发现
2. **我们需要 per-input profiling**: SGLang 的 profiling 是 server 级的，我们需要单条输入级别的 E/P/D 分解
3. **我们可以接受 hot path 开销**: 因为我们是 offline profiling 工具，不是 serving 系统
4. **CUDA event timing**: 我们应该考虑用 CUDA events 代替 `torch.cuda.synchronize()` + `perf_counter()`，减少 sync 开销

### 12.3 具体建议

- 参考 SGLang 的 `SchedulerProfilerMixin` 架构，将 profiling 逻辑从主流程中解耦为 Mixin
- 参考 `bench_one_batch.py` 的 warmup 策略（1 次 warmup + 缩短 output_len）
- 参考 `req_time_stats.py` 的时间校准方法
- 考虑增加 Prometheus metrics 导出，便于批量实验的监控
- 引入 stage-based profiling 思路，自动区分 encode/prefill/decode

---

## 源码索引

| 文件 | 功能 |
|------|------|
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | **核心**: torch.profiler 集成，stage-based profiling |
| `python/sglang/srt/utils/profile_utils.py` | V2 ProfileManager，_StageBasedTrigger 状态机 |
| `python/sglang/srt/observability/req_time_stats.py` | 请求级时间追踪，TTFT/E2E/queueing 计算 |
| `python/sglang/srt/observability/metrics_collector.py` | Prometheus 指标定义，~50 个指标 |
| `python/sglang/srt/observability/func_timer.py` | 函数级延迟 decorator |
| `python/sglang/srt/observability/trace.py` | OpenTelemetry 分布式追踪 |
| `python/sglang/srt/observability/startup_func_log_and_timer.py` | 启动阶段计时 |
| `python/sglang/srt/observability/cpu_monitor.py` | CPU 利用率后台监控 |
| `python/sglang/srt/model_executor/model_runner.py` | 模型加载/初始化计时，NVTX hooks |
| `python/sglang/srt/model_executor/hook_manager.py` | Forward hook 注册，NVTX markers |
| `python/sglang/srt/managers/scheduler.py` | Scheduler 主类，prefill/decode 调度 |
| `python/sglang/srt/managers/scheduler_metrics_mixin.py` | DeviceTimer, 性能估算, bubble 追踪 |
| `python/sglang/srt/mem_cache/memory_pool.py` | KV cache memory 管理和追踪 |
| `python/sglang/srt/mem_cache/radix_cache.py` | Radix cache 指标，驱逐追踪 |
| `python/sglang/bench_serving.py` | 在线服务 benchmark，TTFT/TPOT/ITL |
| `python/sglang/bench_one_batch.py` | 单 batch kernel 级 profiling |
| `python/sglang/srt/server_args.py` | Profiling 相关 CLI 参数 |
| `docs/developer_guide/benchmark_and_profiling.md` | 官方 profiling 使用文档 |
