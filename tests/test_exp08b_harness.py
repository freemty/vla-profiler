"""Unit tests for exp08b harness defensive behavior.

Does NOT require CUDA — tests contract-level invariants.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import exp08b_interference_matrix as hm


def test_nitrogen_payload_aborts_without_real_model(monkeypatch):
    """When require_real=True and NitroGen load fails, abort."""
    monkeypatch.setitem(sys.modules, "src.controllers.nitrogen_controller", None)

    with pytest.raises(RuntimeError, match="NitroGen"):
        hm.build_nitrogen_action_payload(gpu=0, k=10, require_real=True)
