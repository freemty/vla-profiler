# exp08b/exp08c Rescue — Data Validity Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Invalidate currently-shipped exp08b/exp08c results, fix the 4 showstopper measurement bugs identified in the 2026-04-27 Codex adversarial review, and produce defensible data before any downstream analysis/slides/PR references them.

**Architecture:** Two parallel workstreams. (A) **Harness rescue** — patch `scripts/exp08b_interference_matrix.py` so each iteration is measured as an actually-co-located event pair with identical per-call workloads, and `launch_exp08b.sh` enforces the same power/warmup conditions used in exp07a canonical runs. (B) **Paper trail cleanup** — mark existing results as INVALIDATED, scrub fake `citeturn*` citation tokens from the imported survey, fix fake cross-validation in the contention model. No new science until harness passes a self-consistency gate on a known-good phase pair.

**Tech Stack:** Python 3.11 + PyTorch (CUDA streams + events), threading.Barrier, NitroGenController, HuggingFace transformers, bash (launcher), remote server xdlab23 (8x RTX 5880 Ada).

---

## Showstoppers being fixed (from Codex review 2026-04-27)

1. **A phase fell back to dummy TransformerEncoder** — every A number (and every "stealth disruptor" claim) is on dummy data (`scripts/exp08b_interference_matrix.py:182-223`, `exp/exp08b/run.log:25`).
2. **No per-iteration cross-stream barrier** — threads sync once at start, then free-run. Raw arrays in `results_EA.json` / `results_PA.json` show E drifting 150→68ms and P drifting 105→29ms within one "concurrent" run. Medians are schedule artifacts.
3. **Decode KV cache grows across iterations + combos** — `build_llm_decode_payload` mutates `past_kv` / `token` every call; same `step_fn` is reused across warmup / isolated / concurrent / next-combo. D is not a constant workload.
4. **`exp08c_contention_model.py` "CROSS-VALIDATION" is resubstitution** — reports predictions on the same 12 records it fit. True LOO (verified by Codex): MAE 1.07x, R² **−12.69**, worst point PD/D predicts 11.44x vs observed 2.48x.

Confounds to also fix: launcher misses `nvidia-smi -pm 1` (exp07a canonical requires it); `A`'s isolated baseline drifts 39% across combos (51.2ms in EA/PA → 36.8ms in DA), indicating order effect / state leakage.

---

## File Structure

Files that will be touched (by task):

- **Create** `exp/exp08b/INVALIDATED.md` — front-matter banner file pointing to this plan
- **Modify** `exp/exp08c/FINDINGS.md` — prepend INVALIDATED banner, don't delete body (keep for blame/diff)
- **Modify** `scripts/exp08b_interference_matrix.py` — 4 surgical patches (no fallback for A, per-iter barrier, D payload reset, `--require-real-models` flag)
- **Modify** `scripts/launch_exp08b.sh` — add `nvidia-smi -pm 1` + env check
- **Modify** `scripts/exp08c_contention_model.py` — replace fake CV with real LOO + k-fold, rename misleading function
- **Modify** `survey/papers/vla-wam-efficiency-systems-deep-research.md` — strip `citeturn*` tokens, add "UNVERIFIED CITATIONS" banner at top
- **Create** `tests/test_exp08b_harness.py` — unit tests for per-iteration timing alignment, barrier presence, payload idempotency
- **Create** `scripts/verify_exp08b_baseline.py` — self-consistency gate: A isolated baseline must be within ±5% across 3 combos
- **Modify** `exp/summary.md` — append rescue entry
- **Modify** `.claude/skills/project-skill/SKILL.md` — flip exp08a/08b/08c status to "invalidated, rescue in progress"

---

## Task 1: Freeze blast radius — INVALIDATED banners

**Files:**
- Create: `exp/exp08b/INVALIDATED.md`
- Modify: `exp/exp08c/FINDINGS.md` (prepend banner, do not delete body)
- Modify: `survey/papers/vla-wam-efficiency-systems-deep-research.md` (prepend banner)
- Modify: `.claude/skills/project-skill/SKILL.md` (flip exp08 status lines)

- [ ] **Step 1: Create `exp/exp08b/INVALIDATED.md`**

```markdown
# INVALIDATED — do not cite 2026-04-27

All JSON results in this directory (`results_*.json`, `interference_matrix.json`) are
produced by a broken harness. Do NOT use these numbers in slides, commits, PRs, weekly
reports, or `exp/summary.md` downstream analysis.

## What is broken

1. **A phase used dummy TransformerEncoder, not NitroGen.** `build_nitrogen_action_payload`
   silently fell back to a random-weights TransformerEncoder (`scripts/exp08b_interference_matrix.py:203-223`).
2. **No per-iteration cross-stream barrier.** Concurrent loops sync once at start, then
   drift. Per-iter medians are schedule artifacts (`results_EA.json` shows E 150ms→68ms
   within one run).
3. **Decode KV cache grows across iterations.** D's workload drifts monotonically.
4. **Launcher missed `nvidia-smi -pm 1`** — exp07a canonical requires persistence mode.

## Rescue plan

See `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md`.

## What to cite instead

Nothing from this directory until the rescue plan's harness self-consistency gate
(Task 9) passes.
```

- [ ] **Step 2: Prepend banner to `exp/exp08c/FINDINGS.md`**

Prepend (do NOT delete existing body — kept for diff/blame):

```markdown
> **⚠️ INVALIDATED 2026-04-27.** Upstream data in `exp/exp08b/` is from a broken harness.
> All conclusions below (A as "stealth disruptor", E+A safe, M4 contention model R²=0.94,
> DistServe phenomenon validated, optimal split recommendations) are NOT supported by
> valid data. See `exp/exp08b/INVALIDATED.md` and `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md`.
> Original body preserved below for diff history.

---

```

- [ ] **Step 3: Prepend banner to `survey/papers/vla-wam-efficiency-systems-deep-research.md`**

Prepend:

```markdown
> **⚠️ UNVERIFIED CITATIONS — 2026-04-27.** This report was imported from an external
> deep-research tool and contains fake citation tokens (`citeturn33view0`, etc.) that
> do NOT correspond to real sources. Specific numeric claims in this document (e.g.
> OpenVLA-OFT 4.2Hz→109.7Hz, Fast-WAM 190ms/step, vLLM 78.3k stars, JCT -91.4%)
> MUST be independently verified against primary sources before being cited in
> papers, slides, or project docs. See Task 7 of
> `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md` for the scrub status.

---

```

- [ ] **Step 4: Update `.claude/skills/project-skill/SKILL.md` exp08 status**

Replace lines describing exp08a as "done, PA P-inflation 3.15x" and the exp08b/08c plan
descriptions with:

```
- **exp08a (PA/DA pilot 2026-04-26):** **INVALIDATED 2026-04-27** — same harness bug
  as exp08b (no per-iter barrier, A on dummy model). Numbers (PA 3.15x, DA 3.52x) are
  schedule artifacts. See `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md`.
- **exp08b (6-way matrix):** **INVALIDATED 2026-04-27** — see `exp/exp08b/INVALIDATED.md`.
- **exp08c (contention model):** **INVALIDATED 2026-04-27** — fake CV (resubstitution,
  not LOO). True LOO R² = −12.69.
```

- [ ] **Step 5: Commit**

```bash
git add exp/exp08b/INVALIDATED.md exp/exp08c/FINDINGS.md \
  survey/papers/vla-wam-efficiency-systems-deep-research.md \
  .claude/skills/project-skill/SKILL.md
git commit -m "docs(exp08): mark exp08a/08b/08c results INVALIDATED pending rescue"
```

---

## Task 2: Abort instead of silent-fallback when NitroGen unavailable

**Files:**
- Modify: `scripts/exp08b_interference_matrix.py:182-223` (function `build_nitrogen_action_payload`)
- Test: `tests/test_exp08b_harness.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_exp08b_harness.py`:

```python
"""Unit tests for exp08b harness defensive behavior.

Does NOT require CUDA — tests contract-level invariants.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import exp08b_interference_matrix as hm


def test_nitrogen_payload_aborts_without_real_model(monkeypatch):
    """When --require-real-models is set and NitroGen load fails, abort."""
    import importlib

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated: NitroGen import failed")

    # Force the NitroGenController import to fail
    monkeypatch.setitem(sys.modules, "src.controllers.nitrogen_controller", None)

    with pytest.raises(RuntimeError, match="NitroGen"):
        hm.build_nitrogen_action_payload(gpu=0, k=10, require_real=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exp08b_harness.py::test_nitrogen_payload_aborts_without_real_model -v`
Expected: FAIL — current code swallows the error and returns dummy payload.

- [ ] **Step 3: Modify `build_nitrogen_action_payload` signature + behavior**

Change `scripts/exp08b_interference_matrix.py` around line 182:

```python
def build_nitrogen_action_payload(
    gpu: int,
    k: int = 10,
    require_real: bool = True,
) -> Callable[[], None]:
    """A phase: NitroGen DiT k-step denoising loop.

    If `require_real` is True (default), failing to load the real NitroGen
    controller raises RuntimeError. The previous silent fallback to a random-weights
    TransformerEncoder produced invalid `A` measurements throughout exp08a/08b.
    """
    import torch

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    try:
        _repo_root = Path(__file__).resolve().parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from src.controllers.nitrogen_controller import NitroGenController
        ctrl = NitroGenController(
            controller_config={"device": device, "dtype": "bfloat16", "k": k,
                               "model_name": "", "action_dim": 7, "proprio_dim": 7},
        )
        ctrl.init_pipeline()

        def step_fn():
            ctrl.infer_action()
        return step_fn
    except Exception as e:
        if require_real:
            raise RuntimeError(
                f"NitroGen controller unavailable ({e}); refusing dummy fallback. "
                f"Pass --allow-dummy-action to explicitly opt in (smoke tests only)."
            ) from e
        print(f"[warn] NitroGen unavailable ({e}); using dummy DiT (smoke test only).")

    hidden, heads, layers = 1024, 16, 24
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, dtype=dtype,
        ),
        num_layers=layers,
    ).to(device)
    model.eval()
    x = torch.randn(1, 64, hidden, device=device, dtype=dtype)

    def step_fn():
        with torch.no_grad():
            h = x
            for _ in range(k):
                h = model(h)

    return step_fn
```

- [ ] **Step 4: Wire through CLI flag**

In the argparse block (near `args.k`), add:

```python
parser.add_argument(
    "--allow-dummy-action",
    action="store_true",
    help="Allow NitroGen→TransformerEncoder dummy fallback. Smoke tests only.",
)
```

In the builder call (line 481 area):

```python
phase_fns["A"] = build_nitrogen_action_payload(
    args.gpu, k=args.k, require_real=not args.allow_dummy_action,
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_exp08b_harness.py::test_nitrogen_payload_aborts_without_real_model -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/exp08b_interference_matrix.py tests/test_exp08b_harness.py
git commit -m "fix(exp08b): abort on NitroGen load failure instead of silent dummy fallback

Previous behavior silently swapped in a random-weights TransformerEncoder when
NitroGenController import failed. This poisoned every A-phase measurement in
exp08a/08b (see Codex review 2026-04-27). New default require_real=True raises,
with --allow-dummy-action opt-in for smoke tests only."
```

---

## Task 3: Per-iteration cross-stream barrier for true co-location

**Files:**
- Modify: `scripts/exp08b_interference_matrix.py:240-328` (functions `run_loop`, `run_concurrent`)
- Test: `tests/test_exp08b_harness.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_exp08b_harness.py`:

```python
def test_run_loop_uses_per_iter_barrier(monkeypatch):
    """Concurrent runner must synchronize threads at each iteration, not just at start.

    We instrument the barrier and verify it is waited on `n_iter` times per thread,
    not just once.
    """
    import threading
    from unittest.mock import MagicMock

    call_count = {"wait": 0}
    barrier = MagicMock(wraps=threading.Barrier(1))
    original_wait = barrier.wait

    def counted_wait(*args, **kwargs):
        call_count["wait"] += 1
        return original_wait(*args, **kwargs)

    barrier.wait = counted_wait

    step_fn = lambda: None
    times_out = []
    hm.run_loop(
        name="X", step_fn=step_fn, n_iter=5, warmup=0,
        stream=None, times_out=times_out, barrier=barrier, use_cuda=False,
    )

    # Must wait once per iteration (+ optionally once at start). For 5 iters, expect >= 5.
    assert call_count["wait"] >= 5, (
        f"Expected per-iter barrier.wait(), got {call_count['wait']} for 5 iterations. "
        f"This is the 'no per-iter barrier' bug Codex flagged."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exp08b_harness.py::test_run_loop_uses_per_iter_barrier -v`
Expected: FAIL with `assert 1 >= 5` (current code barriers only once pre-loop).

- [ ] **Step 3: Modify `run_loop` to add per-iter barrier**

Replace lines 240-277 in `scripts/exp08b_interference_matrix.py`:

```python
def run_loop(
    name: str,
    step_fn: Callable[[], None],
    n_iter: int,
    warmup: int,
    stream: Any,
    times_out: List[float],
    barrier: Optional[threading.Barrier] = None,
    use_cuda: bool = True,
) -> None:
    """Loop step_fn, record per-iter time via CUDA events.

    The `barrier`, if provided, synchronizes threads BEFORE EACH ITERATION so the
    per-iter timing window covers true co-location, not a schedule-drifted overlap.
    Without this, concurrent loops free-run after a single pre-loop sync — medians
    become schedule artifacts (see Codex review 2026-04-27, results_EA.json drift
    E 150ms→68ms within one run).
    """
    if use_cuda:
        import torch
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                step_fn()
            torch.cuda.synchronize(stream.device)

            for _ in range(n_iter):
                if barrier is not None:
                    # Wait here on host so all streams launch their next iter together.
                    barrier.wait()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                step_fn()
                e1.record(stream)
                e1.synchronize()
                times_out.append(e0.elapsed_time(e1))
    else:
        for _ in range(warmup):
            step_fn()
        for _ in range(n_iter):
            if barrier is not None:
                barrier.wait()
            t0 = time.perf_counter()
            step_fn()
            times_out.append((time.perf_counter() - t0) * 1000.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exp08b_harness.py::test_run_loop_uses_per_iter_barrier -v`
Expected: PASS (wait count >= 5 for 5 iters).

- [ ] **Step 5: Write a second test for the slowest-phase ceiling effect**

Append:

```python
def test_per_iter_barrier_does_not_collapse_measurements():
    """With per-iter barriers and mixed-speed phases, the fast phase's measured
    latency should still reflect its own work, not the slow phase's.

    We run two CPU phases with different sleeps and confirm the faster one's median
    doesn't inflate to match the slower one's.
    """
    phase_fns = {
        "fast": lambda: time.sleep(0.010),  # 10ms
        "slow": lambda: time.sleep(0.040),  # 40ms
    }

    import time
    results = hm.run_concurrent(
        phase_fns=phase_fns, n_iter=10, warmup=2, use_cuda=False,
    )

    fast_median = sorted(results["fast"])[len(results["fast"]) // 2]
    slow_median = sorted(results["slow"])[len(results["slow"]) // 2]

    assert 8 < fast_median < 25, f"fast phase median should be ~10ms, got {fast_median}"
    assert 35 < slow_median < 55, f"slow phase median should be ~40ms, got {slow_median}"
    # Fast < slow, with gap
    assert fast_median < slow_median - 15
```

- [ ] **Step 6: Run both tests**

Run: `pytest tests/test_exp08b_harness.py -v`
Expected: PASS for both barrier tests.

- [ ] **Step 7: Commit**

```bash
git add scripts/exp08b_interference_matrix.py tests/test_exp08b_harness.py
git commit -m "fix(exp08b): per-iteration cross-stream barrier for true co-location

Threads previously synced once pre-loop then free-ran, so concurrent medians were
schedule artifacts (E drifts 150→68ms within one run; P drifts 105→29ms — per
Codex review 2026-04-27). Now barrier.wait() at the head of every iteration so
each measurement window covers a real overlap."
```

---

## Task 4: Reset decode KV cache per iteration

**Files:**
- Modify: `scripts/exp08b_interference_matrix.py:121-153` (function `build_llm_decode_payload`)
- Test: `tests/test_exp08b_harness.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_exp08b_harness.py`:

```python
def test_decode_payload_has_stable_workload(monkeypatch):
    """Decode step_fn must do identical work every call.

    Build a tiny fake decode payload by monkeypatching the model and processor,
    then call step_fn 10 times. The observed KV cache length MUST stay fixed;
    if it grows monotonically, the workload isn't constant.
    """
    # We inspect the payload builder's internal contract: after each step_fn()
    # call, the past_kv tuple length / sequence length must be bounded.
    # We test this by wrapping a small fake.

    import torch

    # Build the payload with a tiny model by monkeypatching the loader
    class _FakeOut:
        def __init__(self, seq):
            self.past_key_values = [torch.zeros(1, 1, seq, 4)]
            self.logits = torch.zeros(1, 1, 10)

    class _FakeModel:
        def __init__(self):
            self._seq = 4
        def __call__(self, *a, **kw):
            # Simulate bounded KV: reset each call when the payload is correct.
            self._seq = 4
            return _FakeOut(self._seq)
        def eval(self): return self

    class _FakeProc:
        def __call__(self, *a, **kw):
            class _B:
                def to(self, device):
                    return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
            return _B()

    monkeypatch.setattr(hm, "_load_decode_model", lambda *a, **kw: (_FakeProc(), _FakeModel()))
    step_fn, state = hm.build_llm_decode_payload_testable(gpu=0)

    initial_seq = state["past_kv"][0].shape[2]
    for _ in range(10):
        step_fn()

    final_seq = state["past_kv"][0].shape[2]
    assert final_seq == initial_seq, (
        f"KV cache grew from seq={initial_seq} to seq={final_seq} across 10 calls. "
        f"Decode workload is not constant — Codex review finding #4."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exp08b_harness.py::test_decode_payload_has_stable_workload -v`
Expected: FAIL (function `build_llm_decode_payload_testable` doesn't exist and KV grows).

- [ ] **Step 3: Refactor decode payload to reset KV each call**

Replace `build_llm_decode_payload` in `scripts/exp08b_interference_matrix.py`:

```python
def _load_decode_model(model_name: str, device: str, dtype):
    """Seam for tests to monkeypatch."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa",
    )
    model.eval()
    return proc, model


def build_llm_decode_payload_testable(
    gpu: int,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
):
    """Return (step_fn, state) for test instrumentation.

    Each `step_fn()` call does a 1-token decode at a FIXED KV cache length. The
    cache is captured once from a prefill and re-bound fresh every step, so work
    per call is constant (unlike the previous version that let the cache grow
    monotonically across iterations and combos — Codex review finding #4).
    """
    import torch

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    proc, model = _load_decode_model(model_name, device, dtype)

    text = "The capital of France is"
    inputs = proc(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)

    # Snapshot the fixed-length KV once. We'll re-use this exact tuple every call
    # — no mutation, no growth.
    import copy
    fixed_kv = prefill_out.past_key_values
    fixed_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    state = {"past_kv": fixed_kv, "token": fixed_token}

    def step_fn():
        with torch.no_grad():
            model(
                input_ids=state["token"],
                past_key_values=state["past_kv"],  # never updated
                use_cache=True,
            )
        # Intentionally do NOT overwrite state — every call is identical work.

    return step_fn, state


def build_llm_decode_payload(
    gpu: int,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> Callable[[], None]:
    """D phase: 1-token decode at fixed KV length."""
    step_fn, _state = build_llm_decode_payload_testable(gpu, model_name)
    return step_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exp08b_harness.py::test_decode_payload_has_stable_workload -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp08b_interference_matrix.py tests/test_exp08b_harness.py
git commit -m "fix(exp08b): decode payload holds KV cache fixed across iterations

Previous step_fn mutated past_key_values each call, so D's workload grew
monotonically across warmup → isolated → concurrent → next-combo (Codex review
finding #4). Now snapshot KV from the prefill once and reuse the exact tuple
every iteration; D is genuinely the same work per call."
```

---

## Task 5: Launcher enforces nvidia-smi -pm 1 + warmup=15

**Files:**
- Modify: `scripts/launch_exp08b.sh`

- [ ] **Step 1: Inspect current launcher**

Run: `cat scripts/launch_exp08b.sh`

Confirm it lacks `nvidia-smi -pm 1` and warmup=15 enforcement.

- [ ] **Step 2: Update launcher**

Modify `scripts/launch_exp08b.sh`:

```bash
#!/usr/bin/env bash
# exp08b — EPDA interference matrix launcher (xdlab23)
# Enforces exp07a canonical conditions: persistence mode + warmup=15.

set -euo pipefail

GPU="${1:-0}"
WARMUP="${WARMUP:-15}"
ITERATIONS="${ITERATIONS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/exp08b}"

if [[ "$WARMUP" -lt 15 ]]; then
  echo "ERROR: WARMUP must be >= 15 (exp07a canonical). Got $WARMUP." >&2
  exit 2
fi

# Persistence mode — mandatory (exp07a bimodal-pollution fix).
if ! nvidia-smi -pm 1 >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi -pm 1 failed. Persistence mode required for stable power state." >&2
  echo "       Without it, GPU clock ramps across warmup and contaminates measurements." >&2
  echo "       Run as root or via sudo -n to enable." >&2
  exit 3
fi

# Lock clocks if the driver supports it (RTX 5880 Ada does).
nvidia-smi -i "$GPU" --lock-gpu-clocks=tdp,tdp >/dev/null 2>&1 || \
  echo "[warn] could not lock GPU clocks; continuing (persistence mode alone is usually enough)"

python scripts/exp08b_interference_matrix.py \
  --gpu "$GPU" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS" \
  --combos EP ED EA PD PA DA \
  --output-dir "$OUTPUT_DIR" \
  "$@"
```

- [ ] **Step 3: Sanity-check the script compiles**

Run: `bash -n scripts/launch_exp08b.sh`
Expected: no output (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_exp08b.sh
git commit -m "fix(exp08b): launcher enforces persistence mode + warmup>=15

Previous launcher forgot nvidia-smi -pm 1 (required by exp07a canonical per
SKILL.md). Without persistence mode the GPU clock ramps across warmup, causing
the bimodal pollution we already debugged in exp07a. Exit 3 on failure, not warn."
```

---

## Task 6: Real leave-one-out CV in exp08c_contention_model.py

**Files:**
- Modify: `scripts/exp08c_contention_model.py` — rename `_cross_validate` to `_resubstitution`, add `_leave_one_out` + `_kfold` + clear report.
- Test: `tests/test_exp08c_contention_model.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_exp08c_contention_model.py`:

```python
"""Tests for exp08c contention model statistical honesty."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import exp08c_contention_model as m


@pytest.fixture
def sample_records():
    data_path = Path(__file__).resolve().parent.parent / "exp/exp08c/model_pairs.json"
    if not data_path.exists():
        pytest.skip(f"{data_path} missing — run exp08b first")
    return json.load(open(data_path))["data"]


def test_leave_one_out_exists_and_is_not_resubstitution(sample_records):
    """LOO must hold each record out, not evaluate on training set."""
    assert hasattr(m, "leave_one_out"), "leave_one_out function must exist"

    loo = m.leave_one_out(sample_records, m.fit_m4_asymmetric, m.predict_inflation)

    # LOO must report per-fold error (one per record), not a single resubstitution
    # number.
    assert len(loo["errors"]) == len(sample_records)
    assert "mae" in loo
    assert "r2" in loo

    # Sanity: observed-vs-predicted pairs must be LOO, so they should generally
    # show MORE error than resubstitution. We just assert both exist and differ.
    resub = m.resubstitution(sample_records, m.fit_m4_asymmetric, m.predict_inflation)
    assert resub["mae"] <= loo["mae"] + 1e-6, (
        "Resubstitution MAE should be <= LOO MAE (training error <= test error). "
        "If this fails, the 'LOO' function is actually resubstitution."
    )


def test_report_flags_negative_r2(sample_records):
    """If LOO R² < 0, the reported summary must say so — not hide it."""
    loo = m.leave_one_out(sample_records, m.fit_m4_asymmetric, m.predict_inflation)
    report = m.format_report(loo, kind="loo")
    if loo["r2"] < 0:
        assert "R² < 0" in report or "negative" in report.lower() or "invalid" in report.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exp08c_contention_model.py -v`
Expected: FAIL — `leave_one_out` / `resubstitution` / `format_report` don't exist yet.

- [ ] **Step 3: Add real CV functions to `exp08c_contention_model.py`**

Append near the bottom of `scripts/exp08c_contention_model.py`, BEFORE `main()`:

```python
# --------------------------------------------------------------------------- #
# Honest cross-validation                                                     #
# --------------------------------------------------------------------------- #

def _eval(records, preds):
    import statistics
    obs = [r["inflation"] for r in records]
    errors = [abs(o - p) for o, p in zip(obs, preds)]
    mae = sum(errors) / len(errors)
    mean_obs = sum(obs) / len(obs)
    ss_res = sum((o - p) ** 2 for o, p in zip(obs, preds))
    ss_tot = sum((o - mean_obs) ** 2 for o in obs)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {"mae": mae, "r2": r2, "errors": errors, "obs": obs, "preds": preds}


def resubstitution(records, fit_fn, predict_fn):
    """Evaluate on the training set — strictly worse than LOO as a quality gate.

    Reported because previous versions of this script called this 'cross-validation'
    by mistake (Codex review 2026-04-27 finding #7).
    """
    model = fit_fn(records)
    preds = [predict_fn(model, r["combo"])[r["phase"]] for r in records]
    return _eval(records, preds)


def leave_one_out(records, fit_fn, predict_fn):
    """True leave-one-out CV.

    For each record i: fit on records[:i] + records[i+1:], predict record i,
    accumulate absolute error and squared residual.
    """
    preds = []
    for i in range(len(records)):
        train = records[:i] + records[i + 1:]
        model = fit_fn(train)
        pred = predict_fn(model, records[i]["combo"])[records[i]["phase"]]
        preds.append(pred)
    return _eval(records, preds)


def format_report(stats, kind: str = "loo") -> str:
    label = {"loo": "LEAVE-ONE-OUT", "resub": "RESUBSTITUTION (training-set only)"}[kind]
    lines = [f"=== {label} ==="]
    lines.append(f"MAE:  {stats['mae']:.3f}x")
    lines.append(f"R²:   {stats['r2']:.3f}")
    if stats["r2"] < 0:
        lines.append("  ⚠  R² < 0 — model predicts WORSE than the dataset mean. NOT predictive.")
    lines.append("  per-record abs errors: " + ", ".join(f"{e:.2f}" for e in stats["errors"]))
    return "\n".join(lines)
```

In `main()`, replace any call to the old `_cross_validate` / resubstitution-masquerading code with:

```python
print(format_report(resubstitution(records, fit_m4_asymmetric, predict_inflation), "resub"))
print()
print(format_report(leave_one_out(records, fit_m4_asymmetric, predict_inflation), "loo"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exp08c_contention_model.py -v`
Expected: PASS.

- [ ] **Step 5: Re-run the script on the existing INVALIDATED data just to surface the true numbers**

Run: `python scripts/exp08c_contention_model.py`
Expected: report shows resubstitution R²≈0.94 AND LOO R²≈-12.69 (or whatever current data gives).
This is fine to commit — we're not using the numbers, just exposing the gap.

- [ ] **Step 6: Commit**

```bash
git add scripts/exp08c_contention_model.py tests/test_exp08c_contention_model.py
git commit -m "fix(exp08c): real leave-one-out CV + resubstitution report

Previous '_cross_validate' scored predictions on the training set (Codex review
finding #7). With 12 observations and ~7 M4 parameters, resub R²=0.94 was
meaningless; true LOO R² is -12.69 (worse than predicting the mean). Now report
both labels explicitly and flag negative R² in the summary."
```

---

## Task 7: Scrub fake `citeturn*` tokens from the imported survey

**Files:**
- Modify: `survey/papers/vla-wam-efficiency-systems-deep-research.md` (and `CLAUDE.md` index line)

- [ ] **Step 1: Count the damage**

Run: `grep -oE "turn[0-9]+(view|search)[0-9]+" survey/papers/vla-wam-efficiency-systems-deep-research.md | wc -l`
Record the count (Codex saw 77 earlier).

- [ ] **Step 2: Strip the tokens in-place**

Run:

```bash
python3 - <<'PY'
import re
from pathlib import Path
p = Path("survey/papers/vla-wam-efficiency-systems-deep-research.md")
text = p.read_text()
# Remove "citeturn33view0turn7view0" style runs entirely (keeps surrounding prose).
# These tokens are NOT markdown links; they are raw tool artifacts.
new = re.sub(r"(?:cite)?turn\d+(?:view|search)\d+", "[UNVERIFIED]", text)
# Collapse consecutive [UNVERIFIED] markers
new = re.sub(r"(?:\[UNVERIFIED\]\s*){2,}", "[UNVERIFIED] ", new)
p.write_text(new)
print("done")
PY
```

- [ ] **Step 3: Sanity-check no tokens remain**

Run: `grep -E "turn[0-9]+(view|search)[0-9]+" survey/papers/vla-wam-efficiency-systems-deep-research.md || echo "clean"`
Expected: `clean`

- [ ] **Step 4: Downgrade CLAUDE.md index line**

Edit `CLAUDE.md` — the line describing this survey — prepend `⚠️ (UNVERIFIED CITATIONS)`:

```
- **VLA/WAM Efficiency Systems Deep Research (2026)** ⚠️ UNVERIFIED CITATIONS: `survey/papers/vla-wam-efficiency-systems-deep-research.md` — ...[existing text]...
```

- [ ] **Step 5: Commit**

```bash
git add survey/papers/vla-wam-efficiency-systems-deep-research.md CLAUDE.md
git commit -m "docs(survey): strip fake citeturn* tokens from deep-research import

Imported deep-research report contained 77 fake citation tokens from the
originating tool (not real sources). Replaced with [UNVERIFIED]. Numeric claims
in this file still require primary-source verification before citation — see
task 7 of 2026-04-27-exp08bc-rescue plan."
```

---

## Task 8: Build self-consistency gate — A baseline must be stable

**Files:**
- Create: `scripts/verify_exp08b_baseline.py`

- [ ] **Step 1: Write the verification script**

Create `scripts/verify_exp08b_baseline.py`:

```python
#!/usr/bin/env python
"""exp08b self-consistency gate.

Runs the same phase (A only) in 3 different combos' isolated sections and asserts
the isolated median is within ±5% across combos. The 2026-04-27 broken harness
showed A isolated 51.2ms in EA/PA but 36.8ms in DA — a 39% order effect that
invalidates pairwise comparisons. A passing gate is a prerequisite for any exp08
data being trusted.

Usage:
    python scripts/verify_exp08b_baseline.py --gpu 0 --iterations 20

Exits 0 if the gate passes, 1 otherwise. Prints per-combo baseline + spread.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_combo(gpu: int, combo: str, iterations: int, out_path: Path) -> dict:
    subprocess.run(
        [
            "python",
            "scripts/exp08b_interference_matrix.py",
            "--gpu", str(gpu),
            "--combos", combo,
            "--iterations", str(iterations),
            "--warmup", "15",
            "--output", str(out_path),
        ],
        check=True,
    )
    return json.load(open(out_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Max relative spread of A's isolated median across combos")
    args = parser.parse_args()

    scratch = Path("exp/exp08b/_verify")
    scratch.mkdir(parents=True, exist_ok=True)

    combos = ["EA", "PA", "DA"]
    baselines: dict[str, float] = {}
    for c in combos:
        out = scratch / f"result_{c}.json"
        data = run_combo(args.gpu, c, args.iterations, out)
        baselines[c] = statistics.median(data["isolated"]["A"]["all_ms"])
        print(f"{c}: A isolated median = {baselines[c]:.2f} ms")

    vals = list(baselines.values())
    spread = (max(vals) - min(vals)) / statistics.mean(vals)
    print(f"\nA-baseline relative spread: {spread * 100:.1f}% (tol {args.tol * 100:.1f}%)")

    if spread > args.tol:
        print("\n❌ GATE FAIL — A's isolated baseline drifts across combos.")
        print("   Fix order effects / state leakage before trusting exp08b numbers.")
        return 1
    print("\n✅ GATE PASS — A's isolated baseline is stable across combos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/verify_exp08b_baseline.py`

- [ ] **Step 3: Syntax-check**

Run: `python -c "import ast; ast.parse(open('scripts/verify_exp08b_baseline.py').read())"`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_exp08b_baseline.py
git commit -m "feat(exp08b): self-consistency gate — A isolated baseline must be stable

Codex review 2026-04-27 noted A's isolated median drifted 39% across combos
(51.2ms in EA/PA, 36.8ms in DA). That's a 7-8σ red flag invalidating pairwise
contention ratios. This gate runs the same phase in 3 combos and exits non-zero
if the spread exceeds 5%. Must pass before any exp08 rerun results are trusted."
```

---

## Task 9: Rerun exp08b on xdlab23 with fixed harness + gate

**Files:**
- No file edits, runbook-style steps. Produces new JSONs in `exp/exp08b/` (overwriting INVALIDATED data) **only if the gate passes**.

- [ ] **Step 1: Sync code to xdlab23**

Run: `bash scripts/sync_to_remote.sh`
Expected: bundle pushed, ssh confirmation.

- [ ] **Step 2: Dry-run on local smoke payload first (CPU)**

Run locally:

```bash
python scripts/exp08b_interference_matrix.py --use-cpu --combos EP PA \
  --iterations 20 --warmup 5 --output-dir /tmp/exp08b_smoke
```

Expected: run completes, JSONs appear in `/tmp/exp08b_smoke`. No crash.

- [ ] **Step 3: SSH to xdlab23 and run the gate**

Run:

```bash
ssh xdlab23_yang "cd /data1/ybyang/vlla && \
  conda activate vit-probe && \
  sudo -n nvidia-smi -pm 1 && \
  python scripts/verify_exp08b_baseline.py --gpu 0 --iterations 20"
```

Expected: `✅ GATE PASS`. If FAIL, stop here and investigate order-effect (model caching, thermal, different seed between combos) before proceeding.

- [ ] **Step 4: Run the full 6-combo matrix (gate passed)**

Run:

```bash
ssh xdlab23_yang "cd /data1/ybyang/vlla && \
  conda activate vit-probe && \
  bash scripts/launch_exp08b.sh 0 2>&1 | tee exp/exp08b/run.log"
```

Expected: 6 JSONs in `exp/exp08b/` + `interference_matrix.json` + `run.log`. No silent dummy-fallback warning (Task 2 aborts instead).

- [ ] **Step 5: Download results**

Run: `bash scripts/download-results.sh`
Expected: files synced to local `exp/exp08b/`.

- [ ] **Step 6: Quick sanity on raw distributions**

Run:

```bash
python3 - <<'PY'
import json, statistics, glob
for f in sorted(glob.glob("exp/exp08b/results_*.json")):
    d = json.load(open(f))
    combo = f.split("_")[-1].replace(".json", "")
    print(f"\n{combo}:")
    for kind in ("isolated", "colocated"):
        for ph, s in d[kind].items():
            arr = s["all_ms"]
            first = statistics.median(arr[:10])
            last = statistics.median(arr[-10:])
            drift = (first - last) / first * 100 if first else 0
            flag = "⚠️ DRIFT" if abs(drift) > 10 else "ok"
            print(f"  {kind:9s} {ph}: first10={first:6.2f} last10={last:6.2f} drift={drift:+5.1f}% {flag}")
PY
```

Expected: per-run drift < 10% in both isolated and colocated sections. If any `⚠️ DRIFT` appears, **do not** re-generate `FINDINGS.md` — escalate.

- [ ] **Step 7: Commit (only if gate + drift check both pass)**

```bash
git add exp/exp08b/results_*.json exp/exp08b/interference_matrix.json exp/exp08b/run.log
git rm exp/exp08b/INVALIDATED.md  # rescue complete for exp08b data
git commit -m "data(exp08b): fresh interference matrix with fixed harness (2026-04-27 rescue)

Gate passed (A isolated spread < 5% across EA/PA/DA). Per-iter drift < 10% on
all combos. Supersedes the 2026-04-26 INVALIDATED data."
```

---

## Task 10: Rewrite FINDINGS.md from fresh data + refit contention model

**Files:**
- Modify: `exp/exp08c/FINDINGS.md` (rewrite body, keep commit history as audit trail)
- Modify: `exp/exp08c/model_pairs.json` (regenerated from new results)
- Modify: `exp/summary.md`
- Modify: `.claude/skills/project-skill/SKILL.md`

- [ ] **Step 1: Regenerate model_pairs.json from fresh results**

Run:

```bash
python scripts/exp08c_contention_model.py --build-pairs \
  --results-dir exp/exp08b --output exp/exp08c/model_pairs.json
```

(If `--build-pairs` doesn't exist, add a small helper that reads each `results_*.json`, computes `inflation = median(colocated[phase]) / median(isolated[phase])`, writes `{"data":[{"combo":..., "phase":..., "inflation":..., "isolated_ms":..., "colocated_ms":...}]}`.)

- [ ] **Step 2: Rerun the contention model**

Run: `python scripts/exp08c_contention_model.py > exp/exp08c/report.txt`

Read `exp/exp08c/report.txt`. Record:
- Resubstitution R²
- LOO R²
- Whether LOO R² < 0

- [ ] **Step 3: Write honest FINDINGS.md**

Overwrite `exp/exp08c/FINDINGS.md`:

```markdown
# exp08c — Contention Mechanism Findings (v2, 2026-04-27 rescue)

## Status

Rescue run of exp08c after 2026-04-27 Codex adversarial review invalidated v1.
Underlying exp08b data now comes from:
- Per-iteration cross-stream barrier (Task 3)
- Decode KV cache frozen (Task 4)
- `nvidia-smi -pm 1` enforced at launcher (Task 5)
- NitroGen real load required, dummy fallback disabled (Task 2)

Self-consistency gate (A isolated spread < 5% across EA/PA/DA combos) PASSED.

## Raw inflation table (medians)

[fill in from exp/exp08b/interference_matrix.json after Task 9]

## Model fit

| Model | Resubstitution R² | LOO R² | Verdict |
|-------|-------------------|--------|---------|
| M4 rank-1 (aggressiveness × vulnerability) | [fill] | [fill] | [fill] |

If LOO R² < 0.3: M4 is NOT predictive. Do not extrapolate to triples.
If 0.3 ≤ LOO R² < 0.7: M4 captures average trend; individual predictions ±50%.
If LOO R² ≥ 0.7: defensible phenomenological model for this 4-phase, 1-GPU, 1-model setup ONLY.

## What we can claim

[populate only from numbers that survive both the consistency gate AND LOO ≥ 0.3]

## What we CANNOT claim from this data

- Triple/quad co-location behavior (no data; M4 is not a mechanism model)
- "DistServe-style validation" — too small + too confounded for that bar
- Cross-GPU / cross-model generalization
- Absolute SLO numbers for any production setup

## Next

If LOO R² < 0.3: abandon M4 shape, consider per-pair table lookups instead of a
parametric model. If ≥ 0.3: design exp08d for independent validation (new model
sizes, new payload lengths, held-out combos).
```

- [ ] **Step 4: Append to `exp/summary.md`**

Add a row under `exp08c`:

```
| exp08c (v2) | 2026-04-27 rescue | [status from report.txt] | [LOO R²] | See FINDINGS.md |
```

- [ ] **Step 5: Update SKILL.md project-skill**

Replace the "INVALIDATED" entries added in Task 1 with the new state:

```
- **exp08b v2 (2026-04-27 rescue):** 6-combo interference matrix with fixed harness.
  Gate passed. See exp/exp08b/interference_matrix.json.
- **exp08c v2 (2026-04-27 rescue):** Contention model refit. LOO R² = [fill]. See
  exp/exp08c/FINDINGS.md for what can and cannot be claimed.
```

- [ ] **Step 6: Commit**

```bash
git add exp/exp08c/FINDINGS.md exp/exp08c/model_pairs.json exp/exp08c/report.txt \
  exp/summary.md .claude/skills/project-skill/SKILL.md
git commit -m "docs(exp08c): rewrite FINDINGS.md from post-rescue data with honest CV

Resubstitution + LOO R² both reported. Mechanism claims constrained to what
LOO supports. Triples/quads claims removed. Original v1 body still accessible
in git history."
```

---

## Self-Review

- **Spec coverage:** 4 Codex showstoppers → Tasks 2, 3, 4, 6. Confounds (launcher power state, A baseline drift) → Tasks 5, 8. Paper trail (banners, fake citations) → Tasks 1, 7. Fresh data + honest writeup → Tasks 9, 10. Every Codex finding has at least one task.
- **Placeholder scan:** FINDINGS.md template in Task 10 contains `[fill]` placeholders — these are INTENTIONAL; they get populated from live data at execution time. Not placeholders for task logic. All code in Tasks 2–6 is complete inline.
- **Type consistency:** `build_llm_decode_payload_testable` returns `(step_fn, state)` in Task 4; Task 4 test uses `step_fn, state = ...` — match. `leave_one_out` / `resubstitution` / `format_report` signatures in Task 6 test match Task 6 implementation.
- **Ordering:** Task 1 (banners) first so any parallel session can't accidentally cite bad data while the fix is in-flight. Task 9 (rerun) depends on Tasks 2–5; Task 10 depends on Task 9.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-exp08bc-rescue.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Good fit here because Tasks 2–6 are independent surgical patches with contract tests; parallelism possible.

**2. Inline Execution** — execute in this session, batch with checkpoints. Simpler but sequential.

Which approach?
