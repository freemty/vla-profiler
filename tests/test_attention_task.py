"""
Tests for attention analysis core functions.

Covers: _compute_attention_scores (including GQA), _gini_coefficient,
_top_k_concentration, _attention_entropy, _find_qk_keys, and the three
task functions with mock controllers.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


# ---- Utility function tests ----

class TestComputeAttentionScores:
    """Tests for _compute_attention_scores."""

    def test_4d_input_returns_softmax_probs(self):
        from src.tasks.attention_task import _compute_attention_scores
        # [batch=1, heads=2, seq=4, head_dim=8]
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        attn = _compute_attention_scores(q, k)
        assert attn.shape == (1, 2, 4, 4)
        # Softmax rows should sum to 1
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_3d_input_reshapes_correctly(self):
        from src.tasks.attention_task import _compute_attention_scores
        # [batch=1, seq=6, hidden=16] with head_dim=8 -> 2 heads
        q = torch.randn(1, 6, 16)
        k = torch.randn(1, 6, 16)
        attn = _compute_attention_scores(q, k, head_dim=8)
        assert attn.shape == (1, 2, 6, 6)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_gqa_mismatched_heads(self):
        from src.tasks.attention_task import _compute_attention_scores
        # Q: 4 heads, K: 2 heads (GQA with repeat_factor=2)
        q = torch.randn(1, 6, 32)  # 4 heads * 8 dim = 32
        k = torch.randn(1, 6, 16)  # 2 heads * 8 dim = 16
        attn = _compute_attention_scores(q, k, head_dim=8)
        assert attn.shape == (1, 4, 6, 6)  # Q head count
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_gqa_4d_input(self):
        from src.tasks.attention_task import _compute_attention_scores
        # 4D with mismatched heads
        q = torch.randn(1, 8, 4, 16)   # 8 Q heads
        k = torch.randn(1, 2, 4, 16)   # 2 K heads (GQA factor=4)
        attn = _compute_attention_scores(q, k)
        assert attn.shape == (1, 8, 4, 4)

    def test_head_dim_not_divisible_raises(self):
        from src.tasks.attention_task import _compute_attention_scores
        q = torch.randn(1, 4, 100)  # 100 not divisible by 128
        k = torch.randn(1, 4, 100)
        with pytest.raises(ValueError, match="not divisible"):
            _compute_attention_scores(q, k, head_dim=128)

    def test_unexpected_dim_raises(self):
        from src.tasks.attention_task import _compute_attention_scores
        q = torch.randn(4, 8)  # 2D
        k = torch.randn(4, 8)
        with pytest.raises(ValueError, match="Unexpected"):
            _compute_attention_scores(q, k)

    def test_identity_attention(self):
        from src.tasks.attention_task import _compute_attention_scores
        # Q == K should produce high diagonal attention
        x = torch.eye(4).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        attn = _compute_attention_scores(x, x)
        # Diagonal should be highest in each row
        for row in range(4):
            assert attn[0, 0, row, row] == attn[0, 0, row].max()


class TestGiniCoefficient:
    """Tests for _gini_coefficient."""

    def test_uniform_distribution_near_zero(self):
        from src.tasks.attention_task import _gini_coefficient
        # Uniform: all equal values -> Gini ≈ 0
        values = torch.ones(100)
        gini = _gini_coefficient(values)
        assert abs(gini) < 0.02  # near zero

    def test_extreme_inequality_near_one(self):
        from src.tasks.attention_task import _gini_coefficient
        # All mass on one element -> Gini ≈ 1
        values = torch.zeros(100)
        values[0] = 1.0
        gini = _gini_coefficient(values)
        assert gini > 0.95

    def test_empty_returns_zero(self):
        from src.tasks.attention_task import _gini_coefficient
        assert _gini_coefficient(torch.tensor([])) == 0.0

    def test_all_zeros_returns_zero(self):
        from src.tasks.attention_task import _gini_coefficient
        assert _gini_coefficient(torch.zeros(10)) == 0.0

    def test_two_element_known_value(self):
        from src.tasks.attention_task import _gini_coefficient
        # [0, 1] -> Gini = 0.5
        gini = _gini_coefficient(torch.tensor([0.0, 1.0]))
        assert abs(gini - 0.5) < 0.01


class TestTopKConcentration:
    """Tests for _top_k_concentration."""

    def test_uniform_returns_k_over_n(self):
        from src.tasks.attention_task import _top_k_concentration
        values = torch.ones(10)
        conc = _top_k_concentration(values, k=3)
        assert abs(conc - 0.3) < 0.01  # 3/10

    def test_all_mass_on_one_returns_one(self):
        from src.tasks.attention_task import _top_k_concentration
        values = torch.zeros(10)
        values[5] = 1.0
        conc = _top_k_concentration(values, k=1)
        assert abs(conc - 1.0) < 0.01

    def test_empty_returns_zero(self):
        from src.tasks.attention_task import _top_k_concentration
        assert _top_k_concentration(torch.tensor([]), k=5) == 0.0

    def test_k_larger_than_n(self):
        from src.tasks.attention_task import _top_k_concentration
        values = torch.tensor([0.3, 0.7])
        conc = _top_k_concentration(values, k=10)
        assert abs(conc - 1.0) < 0.01  # all elements


class TestAttentionEntropy:
    """Tests for _attention_entropy."""

    def test_uniform_distribution_max_entropy(self):
        from src.tasks.attention_task import _attention_entropy
        # Uniform over N elements -> entropy = log(N)
        n = 8
        values = torch.ones(n) / n
        entropy = _attention_entropy(values)
        expected = math.log(n)
        assert abs(entropy - expected) < 0.01

    def test_delta_distribution_zero_entropy(self):
        from src.tasks.attention_task import _attention_entropy
        # All mass on one element -> entropy = 0
        values = torch.zeros(10)
        values[0] = 1.0
        entropy = _attention_entropy(values)
        assert abs(entropy) < 0.01

    def test_empty_returns_zero(self):
        from src.tasks.attention_task import _attention_entropy
        assert _attention_entropy(torch.tensor([])) == 0.0


class TestFindQKKeys:
    """Tests for _find_qk_keys."""

    def test_finds_matching_pairs(self):
        from src.tasks.attention_task import _find_qk_keys
        store = {
            "0_q_states": [torch.zeros(1)],
            "0_k_states": [torch.zeros(1)],
            "7_q_states": [torch.zeros(1)],
            "7_k_states": [torch.zeros(1)],
        }
        pairs = _find_qk_keys(store)
        assert len(pairs) == 2
        assert pairs[0] == (0, "0_q_states", "0_k_states")
        assert pairs[1] == (7, "7_q_states", "7_k_states")

    def test_unpaired_keys_excluded(self):
        from src.tasks.attention_task import _find_qk_keys
        store = {
            "0_q_states": [torch.zeros(1)],
            # no 0_k_states
            "7_q_states": [torch.zeros(1)],
            "7_k_states": [torch.zeros(1)],
        }
        pairs = _find_qk_keys(store)
        assert len(pairs) == 1
        assert pairs[0][0] == 7

    def test_empty_store(self):
        from src.tasks.attention_task import _find_qk_keys
        assert _find_qk_keys({}) == []

    def test_sorted_by_layer_idx(self):
        from src.tasks.attention_task import _find_qk_keys
        store = {
            "14_q_states": [], "14_k_states": [],
            "0_q_states": [], "0_k_states": [],
            "7_q_states": [], "7_k_states": [],
        }
        pairs = _find_qk_keys(store)
        layer_indices = [p[0] for p in pairs]
        assert layer_indices == [0, 7, 14]


# ---- Task function tests with mock controller ----

class MockController:
    """Minimal mock controller for task testing."""

    def __init__(self, global_store=None):
        self.global_store = global_store or {}
        self.logger = type("L", (), {
            "info": lambda self, *a, **kw: None,
            "warning": lambda self, *a, **kw: None,
        })()


class TestTaskVisualTextAttn:
    """Tests for task_visual_text_attn with mock data."""

    def test_produces_output_json(self, tmp_path):
        from src.tasks.attention_task import task_visual_text_attn
        # Create mock QK data: [batch=1, seq=10, hidden=16], head_dim=8 -> 2 heads
        q = torch.randn(1, 10, 16)
        k = torch.randn(1, 10, 16)
        ctrl = MockController({
            "0_q_states": [q],
            "0_k_states": [k],
        })
        result = task_visual_text_attn(ctrl, str(tmp_path), {"top_k": 3, "head_dim": 8})
        assert "layer_0" in result
        assert "text_to_visual_gini" in result["layer_0"]
        # Check output file
        output_file = tmp_path / "visual_text_attention" / "sparsity_stats.json"
        assert output_file.exists()

    def test_no_qk_returns_empty(self, tmp_path):
        from src.tasks.attention_task import task_visual_text_attn
        ctrl = MockController({})
        result = task_visual_text_attn(ctrl, str(tmp_path), {})
        assert result == {}


class TestTaskSinkDetection:
    """Tests for task_sink_detection with mock data."""

    def test_finds_sink_tokens(self, tmp_path):
        from src.tasks.attention_task import task_sink_detection
        q = torch.randn(1, 10, 16)
        k = torch.randn(1, 10, 16)
        ctrl = MockController({
            "0_q_states": [q],
            "0_k_states": [k],
        })
        result = task_sink_detection(ctrl, str(tmp_path), {"sink_k": 3, "head_dim": 8})
        assert "layer_0" in result
        sinks = result["layer_0"]["sink_tokens"]
        assert len(sinks) == 3
        # Each sink has required fields
        for s in sinks:
            assert "position" in s
            assert "type" in s
            assert "attention_received" in s


class TestTaskPerLayerStats:
    """Tests for task_per_layer_stats with mock data."""

    def test_computes_entropy_and_stats(self, tmp_path):
        from src.tasks.attention_task import task_per_layer_stats
        q = torch.randn(1, 8, 16)
        k = torch.randn(1, 8, 16)
        ctrl = MockController({
            "5_q_states": [q],
            "5_k_states": [k],
        })
        result = task_per_layer_stats(ctrl, str(tmp_path), {"head_dim": 8})
        assert "layer_5" in result
        stats = result["layer_5"]
        assert "mean_entropy" in stats
        assert "mean_attention" in stats
        assert stats["mean_entropy"] > 0  # entropy should be positive
        assert 0 < stats["mean_attention"] < 1  # softmax values
