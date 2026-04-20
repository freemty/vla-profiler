"""Tests for demo_reproduce_task."""

import os
import json
import tempfile

import torch
import pytest

from src.tasks.demo_reproduce_task import (
    _check_vlm_text,
    _check_openvla_action,
    _check_vla_action,
    task_demo_reproduce,
)


class TestVLMTextChecks:
    def test_valid_output(self):
        result = ["A woman is standing on a beach with her dog at sunset."]
        expected = {"min_output_tokens": 5, "keywords": ["woman", "beach", "dog"]}
        checks = _check_vlm_text(result, expected)
        assert all(c["passed"] for c in checks)

    def test_empty_output(self):
        result = [""]
        expected = {"min_output_tokens": 5}
        checks = _check_vlm_text(result, expected)
        assert not checks[0]["passed"]

    def test_too_short(self):
        result = ["Hello"]
        expected = {"min_output_tokens": 10}
        checks = _check_vlm_text(result, expected)
        non_empty = checks[0]
        min_len = checks[1]
        assert non_empty["passed"]
        assert not min_len["passed"]

    def test_keyword_miss(self):
        result = ["A cat is sleeping on a couch."]
        expected = {"min_output_tokens": 3, "keywords": ["dog", "beach"]}
        checks = _check_vlm_text(result, expected)
        keyword_check = next(c for c in checks if c["name"] == "keyword_presence")
        assert not keyword_check["passed"]

    def test_dict_input(self):
        result = {"output_text": ["A woman walking her dog."]}
        expected = {"min_output_tokens": 3}
        checks = _check_vlm_text(result, expected)
        assert checks[0]["passed"]


class TestOpenVLAChecks:
    def test_valid_7_tokens(self):
        result = {"action_token_ids": [128, 64, 32, 200, 100, 50, 255]}
        expected = {"action_token_count": 7}
        checks = _check_openvla_action(result, expected)
        assert all(c["passed"] for c in checks)

    def test_wrong_count(self):
        result = {"action_token_ids": [128, 64, 32]}
        expected = {"action_token_count": 7}
        checks = _check_openvla_action(result, expected)
        assert not checks[0]["passed"]

    def test_out_of_range(self):
        result = {"action_token_ids": [128, 64, 32, 200, 100, 50, 300]}
        expected = {"action_token_count": 7}
        checks = _check_openvla_action(result, expected)
        range_check = next(c for c in checks if c["name"] == "action_token_range")
        assert not range_check["passed"]


class TestVLAActionChecks:
    def test_valid_shape(self):
        actions = torch.randn(1, 4, 7)
        result = {"actions": actions, "action_shape": [1, 4, 7], "denoise_steps": 10}
        expected = {"action_shape": [1, 4, 7], "denoise_steps": 10, "action_clip_value": 5.0}
        checks = _check_vla_action(result, expected)
        assert all(c["passed"] for c in checks)

    def test_wrong_shape(self):
        result = {"action_shape": [1, 8, 7], "denoise_steps": 10}
        expected = {"action_shape": [1, 4, 7]}
        checks = _check_vla_action(result, expected)
        shape_check = next(c for c in checks if c["name"] == "action_shape")
        assert not shape_check["passed"]

    def test_nan_detection(self):
        actions = torch.tensor([[[float("nan"), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]])
        result = {"actions": actions, "action_shape": [1, 1, 7]}
        expected = {"action_ndim": 3}
        checks = _check_vla_action(result, expected)
        nan_check = next(c for c in checks if c["name"] == "action_no_nan_inf")
        assert not nan_check["passed"]

    def test_clip_violation(self):
        actions = torch.ones(1, 4, 7) * 2.0
        result = {"actions": actions, "action_shape": [1, 4, 7]}
        expected = {"action_clip_value": 1.0}
        checks = _check_vla_action(result, expected)
        clip_check = next(c for c in checks if c["name"] == "action_value_range")
        assert not clip_check["passed"]

    def test_string_shape_error(self):
        result = {"action_shape": "unknown"}
        expected = {}
        checks = _check_vla_action(result, expected)
        assert not checks[0]["passed"]

    def test_denoise_steps_mismatch(self):
        result = {"action_shape": [1, 4, 7], "denoise_steps": 5}
        expected = {"action_ndim": 3, "denoise_steps": 10}
        checks = _check_vla_action(result, expected)
        denoise_check = next(c for c in checks if c["name"] == "denoise_steps")
        assert not denoise_check["passed"]


class TestTaskFunction:
    def test_no_result(self):
        class FakeController:
            pass

        ctrl = FakeController()
        with tempfile.TemporaryDirectory() as tmpdir:
            report = task_demo_reproduce(ctrl, tmpdir, {"model_type": "vla"})
            assert report["status"] == "ERROR"
            assert os.path.exists(os.path.join(tmpdir, "demo_reproduce", "demo_report.json"))

    def test_vlm_pass(self):
        class FakeController:
            _last_result = ["A woman walks her dog on a sandy beach at sunset."]

        ctrl = FakeController()
        with tempfile.TemporaryDirectory() as tmpdir:
            report = task_demo_reproduce(ctrl, tmpdir, {
                "model_type": "vlm",
                "expected": {"min_output_tokens": 5, "keywords": ["woman", "dog"]},
            })
            assert report["status"] == "PASS"

    def test_vla_pass(self):
        class FakeController:
            _last_result = {
                "actions": torch.randn(1, 4, 7).clamp(-0.9, 0.9),
                "action_shape": [1, 4, 7],
                "denoise_steps": 10,
            }

        ctrl = FakeController()
        with tempfile.TemporaryDirectory() as tmpdir:
            report = task_demo_reproduce(ctrl, tmpdir, {
                "model_type": "vla",
                "expected": {
                    "action_shape": [1, 4, 7],
                    "denoise_steps": 10,
                    "action_clip_value": 1.0,
                },
            })
            assert report["status"] == "PASS"

            report_path = os.path.join(tmpdir, "demo_reproduce", "demo_report.json")
            with open(report_path) as f:
                saved = json.load(f)
            assert saved["status"] == "PASS"
