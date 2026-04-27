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

## GPU Power-State Warmup Bimodality (2026-04-27)

### Symptom
Per-iteration latency splits into **two clusters** instead of Gaussian-distributed around the mean:
- Runs 1-12 consistently slower (~1.25x)
- Runs 13-20 consistently faster (true steady state)
- CV appears moderate (~10-12%) but distribution is bimodal, not normal

Real example from exp07a (Pi-Zero, 5 warmup + 20 benchmark on RTX 5880 Ada):
```
Action phase all_ms (ms):
  runs 1-12:  [205.5, 204.1, 204.9, 204.9, 206.6, 206.1, 205.6, 210.9, 212.7, 204.1, 205.3, 192.8]
  runs 13-20: [162.0, 162.8, 163.2, 167.6, 166.8, 166.2, 164.2, 165.3]
  gap: ~40ms (25%)
```

### Cause
Default 5 warmup iterations are **too few to ramp the GPU through P-states**. The card starts in a low-power state (P2/P8); initial kernels run at reduced clocks, and after roughly 6-10 iterations of sustained load the scheduler upclocks to P0. This is especially pronounced on RTX workstation cards (5880 Ada, A6000) where power management is more aggressive than datacenter H100/A100.

Naive **mean** over a bimodal distribution is meaningless — it lands between the two modes and drifts with warmup count.

### Solution — three complementary mitigations

**(a) Increase warmup to 15** — cheapest, works for most workloads:
```python
# scripts/profile_fastwam.py, scripts/profile_lingbot_va.py
parser.add_argument("--warmup", type=int, default=15,
    help="Default 15 absorbs GPU power-state warmup bimodality")
```

**(b) Lock GPU to persistent high-power mode** — most reliable, needs root:
```bash
sudo nvidia-smi -pm 1                  # persistent mode (keep driver loaded)
sudo nvidia-smi -i 0 -lgc 2505,2505    # lock graphics clock to max
# ...run benchmarks...
sudo nvidia-smi -i 0 -rgc              # restore
sudo nvidia-smi -pm 0
```
`nvidia-smi -pm 1` alone often cuts the bimodality in half; combined with `-lgc` it flatlines CV to <2%.

**(c) Canonical = stable-window median, not mean**: when you can't rerun, report `median(all_ms[warmup_est:])` as the canonical number and label naive mean as "polluted". exp07a canonical = median of runs 13-20 (200.5ms) vs naive mean of 20 runs (225ms) — 12% lower, matches real-world sustained latency.

### How to detect this in data you already have
Plot `all_ms` iteration index vs latency. If there's a visible step-down (not gradual), you have P-state bimodality, not statistical noise. Cross-check: compute `median(first_half)` vs `median(second_half)`; >15% gap = bimodal.

Rule: **always persist `all_ms` raw iteration array** (`_profiling_stats.compute_phase_stats` does this). Without it you cannot detect or recover from this post-hoc.

### When this matters most
- First-time profiling of a new model — you don't yet know the steady-state latency
- Cross-model comparison on the same GPU — unequal warmup = unfair comparison
- Before committing a "canonical" number to a paper figure — rerun with warmup=15 + `-pm 1`

## Notes
- Date: 2026-04-15 (original), updated 2026-04-27 (bimodality section)
- Source: 4-agent parallel survey of 8 ML inference systems (original); exp07a/exp04b rerun audit (bimodality)
- Full report: `survey/papers/ml-profiling-systems-comprehensive-survey.md`
- Related experiments: exp07a (Pi-Zero bimodal discovery), exp04b rerun (warmup=15 canonical)
