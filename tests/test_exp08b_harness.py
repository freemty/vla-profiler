"""Unit tests for exp08b harness defensive behavior.

Does NOT require CUDA — tests contract-level invariants.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
import threading
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import exp08b_interference_matrix as hm


def test_nitrogen_payload_aborts_without_real_model(monkeypatch):
    """When require_real=True and NitroGen load fails, abort."""
    monkeypatch.setitem(sys.modules, "src.controllers.nitrogen_controller", None)

    with pytest.raises(RuntimeError, match="NitroGen"):
        hm.build_nitrogen_action_payload(gpu=0, k=10, require_real=True)


def test_run_loop_uses_per_iter_barrier():
    """Concurrent runner must synchronize threads at each iteration, not just at start."""
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

    assert call_count["wait"] >= 5, (
        f"Expected per-iter barrier.wait(), got {call_count['wait']} for 5 iterations."
    )


def test_per_iter_barrier_does_not_collapse_measurements():
    """With per-iter barriers and mixed-speed phases, the fast phase's measured
    latency should still reflect its own work, not the slow phase's."""
    phase_fns = {
        "fast": lambda: time.sleep(0.010),
        "slow": lambda: time.sleep(0.040),
    }

    results = hm.run_concurrent(
        phase_fns=phase_fns, n_iter=10, warmup=2, use_cuda=False,
    )

    fast_median = sorted(results["fast"])[len(results["fast"]) // 2]
    slow_median = sorted(results["slow"])[len(results["slow"]) // 2]

    assert 8 < fast_median < 25, f"fast phase median should be ~10ms, got {fast_median}"
    assert 35 < slow_median < 55, f"slow phase median should be ~40ms, got {slow_median}"
    assert fast_median < slow_median - 15


def test_decode_payload_has_stable_workload(monkeypatch):
    """Decode step_fn must do identical work every call — KV cache must not grow."""
    import torch

    class _FakeOut:
        def __init__(self, seq):
            self.past_key_values = [torch.zeros(1, 1, seq, 4)]
            self.logits = torch.zeros(1, 1, 10)

    class _FakeModel:
        def __call__(self, *a, **kw):
            return _FakeOut(4)
        def eval(self):
            return self

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
        f"KV cache grew from seq={initial_seq} to seq={final_seq} across 10 calls."
    )
