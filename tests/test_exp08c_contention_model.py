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

    assert len(loo["errors"]) == len(sample_records)
    assert "mae" in loo
    assert "r2" in loo

    resub = m.resubstitution(sample_records, m.fit_m4_asymmetric, m.predict_inflation)
    assert resub["mae"] <= loo["mae"] + 1e-6, (
        "Resubstitution MAE should be <= LOO MAE (training error <= test error)."
    )


def test_report_flags_negative_r2(sample_records):
    """If LOO R² < 0, the reported summary must say so."""
    loo = m.leave_one_out(sample_records, m.fit_m4_asymmetric, m.predict_inflation)
    report = m.format_report(loo, kind="loo")
    if loo["r2"] < 0:
        assert "R² < 0" in report or "negative" in report.lower() or "invalid" in report.lower()
