"""
CUDA event-based phase timing with CPU fallback.

Provides PhaseTimer for measuring inference phase durations
(encode, prefill, decode, action) with sub-millisecond precision on GPU
and perf_counter fallback on CPU.

Stream support (exp08 infrastructure):
  mark_start/mark_end accept an optional `stream` argument (CUDA stream or
  None for current stream). When timing concurrent phases on separate
  streams, events are recorded on their respective streams and synchronized
  independently, enabling interference measurement without cross-stream
  serialization artifacts.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class _CpuTimerBackend:
    """CPU timing backend using time.perf_counter."""

    __slots__ = ("_start_ns", "_end_ns")

    def __init__(self) -> None:
        self._start_ns: float = 0.0
        self._end_ns: float = 0.0

    def record_start(self, stream: Any = None) -> None:  # noqa: ARG002
        self._start_ns = time.perf_counter()

    def record_end(self, stream: Any = None) -> None:  # noqa: ARG002
        self._end_ns = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._end_ns - self._start_ns) * 1000.0


class _CudaTimerBackend:
    """CUDA event-based timing backend with optional stream targeting.

    When `stream` is passed to record_start/record_end, the event is
    recorded on that stream (otherwise current stream). This is essential
    for measuring per-stream phase latency under concurrent execution
    (exp08a interference matrix).
    """

    __slots__ = ("_start_event", "_end_event", "_synchronized", "_stream")

    def __init__(self) -> None:
        import torch

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._synchronized = False
        self._stream: Any = None

    def record_start(self, stream: Any = None) -> None:
        self._stream = stream
        if stream is not None:
            self._start_event.record(stream)
        else:
            self._start_event.record()
        self._synchronized = False

    def record_end(self, stream: Any = None) -> None:
        # Allow end stream override (rare); default to the start stream.
        target = stream if stream is not None else self._stream
        if target is not None:
            self._end_event.record(target)
        else:
            self._end_event.record()
        self._synchronized = False

    def elapsed_ms(self) -> float:
        if not self._synchronized:
            self._end_event.synchronize()
            self._synchronized = True
        return self._start_event.elapsed_time(self._end_event)


class PhaseTimer:
    """
    Track timing for named inference phases.

    Uses CUDA events when available, falls back to CPU perf_counter.
    Tracks decode step count for per-token timing calculations.

    Stream-aware timing (exp08 interference probe):
        Pass `stream=` to mark_start/mark_end to record the event on a
        specific CUDA stream. This lets a single PhaseTimer instance track
        multiple concurrent phases without cross-stream serialization.

    Usage::

        timer = PhaseTimer()
        timer.mark_start("encode")
        # ... run encoder ...
        timer.mark_end("encode")
        print(timer.elapsed_ms("encode"))  # milliseconds

        # Concurrent phases on separate streams:
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        timer.mark_start("P", stream=s1)
        timer.mark_start("A", stream=s2)
        with torch.cuda.stream(s1): run_prefill()
        with torch.cuda.stream(s2): run_action()
        timer.mark_end("P", stream=s1)
        timer.mark_end("A", stream=s2)
    """

    def __init__(self, *, force_cpu: bool = False) -> None:
        self._use_cuda = _cuda_available() and not force_cpu
        self._completed: Dict[str, Any] = {}
        self._active: Dict[str, Any] = {}
        self._decode_step_count: int = 0

        backend_name = "CUDA" if self._use_cuda else "CPU"
        logger.info("PhaseTimer initialized with %s backend", backend_name)

    @property
    def decode_step_count(self) -> int:
        """Number of completed decode steps."""
        return self._decode_step_count

    def _make_backend(self) -> Any:
        """Create a new timer backend instance."""
        if self._use_cuda:
            return _CudaTimerBackend()
        return _CpuTimerBackend()

    def mark_start(self, phase: str, stream: Any = None) -> None:
        """Record the start of a named phase.

        Args:
            phase: Phase name (e.g., "encode", "prefill", "decode", "action").
            stream: Optional torch.cuda.Stream. If None, uses current stream.
                CPU backend ignores this argument.
        """
        backend = self._make_backend()
        backend.record_start(stream=stream)
        self._active = {**self._active, phase: backend}

    def mark_end(self, phase: str, stream: Any = None) -> None:
        """
        Record the end of a named phase.

        For phases that repeat (e.g., "decode" runs N times), elapsed times
        are accumulated. Increments decode_step_count when phase is "decode".

        Args:
            phase: Phase name to close. Must match a prior mark_start.
            stream: Optional stream override for the end event. Defaults to
                the stream used at mark_start.
        """
        if phase not in self._active:
            raise KeyError(
                f"Cannot end phase '{phase}': mark_start was not called"
            )

        backend = self._active[phase]
        backend.record_end(stream=stream)

        # Accumulate: store list of completed backends per phase
        existing = self._completed.get(phase, [])
        self._completed = {**self._completed, phase: [*existing, backend]}
        self._active = {k: v for k, v in self._active.items() if k != phase}

        if phase == "decode":
            self._decode_step_count += 1

    def elapsed_ms(self, phase: str) -> float:
        """
        Return total elapsed time in milliseconds for a phase.

        If the phase was marked multiple times (e.g., decode), returns
        the sum of all intervals. Raises KeyError if phase was never completed.
        """
        if phase not in self._completed:
            raise KeyError(
                f"Phase '{phase}' not found in completed timings. "
                f"Available: {list(self._completed.keys())}"
            )
        backends = self._completed[phase]
        return sum(b.elapsed_ms() for b in backends)

    def summary(self) -> Dict[str, float]:
        """Return a dict of all completed phase names to total elapsed ms."""
        return {phase: self.elapsed_ms(phase) for phase in self._completed}

    def reset(self) -> None:
        """Clear all timing state and reset decode step count."""
        self._completed = {}
        self._active = {}
        self._decode_step_count = 0
