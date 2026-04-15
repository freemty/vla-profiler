# PhaseTimer CPU Backend Bug

> `record_end()` was silently skipped for CPU backend due to `if self._use_cuda:` guard

## Problem

PhaseTimer's `mark_end()` wrapped `backend.record_end()` in `if self._use_cuda:`, so when running in CPU-only mode (no CUDA), end timestamps were never recorded. CPU backend elapsed times would always be 0.

## Cause

Original code in `src/utils/timing.py:129`:

```python
def mark_end(self, phase: str) -> None:
    backend = self._active[phase]
    if self._use_cuda:          # BUG: CPU backend skipped
        backend.record_end()
```

The `backend` variable already holds the correct type (`_CpuTimerBackend` or `_CudaTimerBackend`) via polymorphism. The `if self._use_cuda:` check was redundant and broke CPU mode.

## Solution

Remove the guard — `record_end()` should be called unconditionally:

```python
def mark_end(self, phase: str) -> None:
    backend = self._active[phase]
    backend.record_end()       # FIX: unconditional, polymorphism handles dispatch
```

## How It Was Found

During a systematic profiling systems survey (8 ML systems), the CUDA best practices research agent audited our PhaseTimer code and identified the conditional branch that would prevent CPU timing from working.

## Cross-Validation Approach

To catch similar bugs in the future, we added `timing_validation` task that runs the same inference with both PhaseTimer (CUDA Events) and `torch.profiler` (record_function), then compares the two independent measurements. Deviation > 5% triggers WARN, > 15% FAIL.

```bash
# Runs automatically when timing_validation is in tasks list
bash scripts/launch_exp.sh 0 qwen_vl_7b/profiling
# Output: output/*/timing_validation/validation_report.json
```

## Notes
- Date: 2026-04-15
- Fix commit: timing.py line 129
- Related: profiling stats also enhanced with median/P10/P90/P99/CV
