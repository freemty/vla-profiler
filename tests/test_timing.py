"""Tests for PhaseTimer — CUDA event-based timing with CPU fallback."""

import time

import pytest

from src.utils.timing import PhaseTimer


class TestPhaseTimerMarkAndElapsed:
    """Test basic mark_start / mark_end / elapsed_ms cycle."""

    def test_mark_and_elapsed_returns_positive_ms(self):
        timer = PhaseTimer()
        timer.mark_start("encode")
        time.sleep(0.01)  # ~10ms
        timer.mark_end("encode")
        elapsed = timer.elapsed_ms("encode")
        assert elapsed > 0.0

    def test_elapsed_before_mark_end_raises(self):
        timer = PhaseTimer()
        timer.mark_start("prefill")
        with pytest.raises(KeyError):
            timer.elapsed_ms("prefill")

    def test_unknown_phase_raises_key_error(self):
        timer = PhaseTimer()
        with pytest.raises(KeyError):
            timer.elapsed_ms("nonexistent")


class TestPhaseTimerMultiplePhases:
    """Test independent tracking of multiple phases."""

    def test_two_phases_tracked_independently(self):
        timer = PhaseTimer()

        timer.mark_start("encode")
        time.sleep(0.01)
        timer.mark_end("encode")

        timer.mark_start("decode")
        time.sleep(0.02)
        timer.mark_end("decode")

        encode_ms = timer.elapsed_ms("encode")
        decode_ms = timer.elapsed_ms("decode")
        assert encode_ms > 0.0
        assert decode_ms > 0.0
        # Decode slept longer, so should generally be larger
        # (not strictly guaranteed on CPU, but with these margins it's fine)
        assert decode_ms > encode_ms * 0.5


class TestPhaseTimerDecodeAccumulation:
    """Test that decode_step_count increments on each mark_end('decode')."""

    def test_decode_step_count_starts_at_zero(self):
        timer = PhaseTimer()
        assert timer.decode_step_count == 0

    def test_decode_step_count_increments(self):
        timer = PhaseTimer()

        for i in range(5):
            timer.mark_start("decode")
            timer.mark_end("decode")

        assert timer.decode_step_count == 5

    def test_non_decode_phase_does_not_increment(self):
        timer = PhaseTimer()
        timer.mark_start("encode")
        timer.mark_end("encode")
        assert timer.decode_step_count == 0


class TestPhaseTimerReset:
    """Test that reset clears all state."""

    def test_reset_clears_phases(self):
        timer = PhaseTimer()
        timer.mark_start("encode")
        timer.mark_end("encode")
        timer.mark_start("decode")
        timer.mark_end("decode")

        timer.reset()

        with pytest.raises(KeyError):
            timer.elapsed_ms("encode")
        assert timer.decode_step_count == 0

    def test_reset_allows_reuse(self):
        timer = PhaseTimer()
        timer.mark_start("encode")
        timer.mark_end("encode")
        timer.reset()

        timer.mark_start("encode")
        time.sleep(0.005)
        timer.mark_end("encode")
        assert timer.elapsed_ms("encode") > 0.0


class TestPhaseTimerSummary:
    """Test that summary returns all completed phases."""

    def test_summary_returns_all_phases(self):
        timer = PhaseTimer()

        timer.mark_start("encode")
        timer.mark_end("encode")
        timer.mark_start("prefill")
        timer.mark_end("prefill")
        timer.mark_start("decode")
        timer.mark_end("decode")

        result = timer.summary()
        assert "encode" in result
        assert "prefill" in result
        assert "decode" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_summary_empty_when_no_phases(self):
        timer = PhaseTimer()
        assert timer.summary() == {}
