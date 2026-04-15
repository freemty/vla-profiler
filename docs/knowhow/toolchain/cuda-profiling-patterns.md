# CUDA Profiling Patterns for VLM Inference

> Correct GPU timing patterns learned from surveying 8 production ML systems (vLLM, SGLang, TensorRT-LLM, DeepSpeed, etc.)

## Problem

GPU operations are asynchronous — naive CPU timing (`time.perf_counter()` without sync) measures kernel launch time, not execution time. Need reliable E/P/D phase timing for VLM profiling.

## Cause

Three fundamentally different timing approaches exist, each measuring something different:
- **CPU timer without sync**: measures kernel launch overhead (~100µs), useless
- **CPU timer with `torch.cuda.synchronize()`**: measures wall-clock including CPU-GPU sync overhead
- **CUDA Events**: measures pure GPU execution time (~0.5µs precision), excludes CPU overhead

## Solution

### Pattern 1: CUDA Event Timing (our PhaseTimer approach)

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... GPU operations ...
end.record()
torch.cuda.synchronize()  # or end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Best for: sub-phase GPU timing, micro-benchmarks, when you need pure GPU time.

### Pattern 2: synchronize + perf_counter (vLLM/SGLang approach)

```python
torch.cuda.synchronize()        # barrier before
start = time.perf_counter()
# ... GPU operations ...
torch.cuda.synchronize()        # barrier after
elapsed = time.perf_counter() - start
```

Best for: end-to-end phase timing, when sync overhead is negligible vs operation time.
This is what vLLM uses for EncoderTimingStats and SGLang uses in bench_one_batch.

### Pattern 3: torch.profiler record_function (cross-validation)

```python
with torch.profiler.profile(activities=[CPU, CUDA]) as prof:
    with torch.profiler.record_function("PHASE_encode"):
        output = encoder(images)
# Extract: prof.key_averages() → cuda_time_total
```

Best for: kernel-level analysis, cross-validating other timing methods.
NOT for benchmarking (profiler overhead distorts timing).

### Warmup Requirements

| Factor | Why warmup needed |
|--------|------------------|
| CUDA context init | First CUDA call: ~100ms-1s |
| cuBLAS/cuDNN handles | First matmul/conv extra overhead |
| torch.compile JIT | First run compiles |
| cudnn.benchmark | First conv searches optimal algo |
| Memory allocator | First alloc requests from CUDA |

Minimum 3 warmup iterations for VLM inference profiling.

### Statistical Rigor

- **Primary metric**: median (robust to outliers)
- **Report**: P10/P90/P99 + CV (coefficient of variation)
- **CV > 5%**: measurement unstable, investigate interference
- **Minimum runs**: 10 (vLLM uses 30, Triton do_bench uses 100ms window)
- **Outlier handling**: DeepSpeed trim_mean (10% trim) is industry best practice

### GPU Clock Stability

```bash
# Lock GPU clock for reproducible benchmarks
nvidia-smi -i 0 -lgc 1500,1500
# ... run benchmarks ...
nvidia-smi -i 0 -rgc  # restore
```

### NVTX Annotation (for Nsight Systems)

```python
# In module hooks:
torch.cuda.nvtx.range_push("ENCODE")
# ... operation ...
torch.cuda.nvtx.range_pop()

# Then: nsys profile --trace=cuda,nvtx python script.py
```

## Key Insight

Production systems (vLLM, SGLang, TensorRT-LLM) all use sync+perf_counter, NOT CUDA Events, for latency benchmarking. CUDA Events are used only for multi-stream synchronization. The difference is typically <1ms for VLM inference phases.

## Notes
- Date: 2026-04-15
- Source: 4-agent parallel survey of 8 ML inference systems
- Full report: `survey/papers/ml-profiling-systems-comprehensive-survey.md`
