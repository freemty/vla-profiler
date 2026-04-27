"""Standalone demo_reproduce for Fast-WAM and LingBot-VA.

These WAM controllers run via standalone scripts (not Hydra framework),
so they can't use src/tasks/demo_reproduce_task.py directly. This script
replicates the VLA action-validation logic from that task, runs a single
forward pass, and emits a demo_report.json of the same schema.

Usage (on xdlab23):
  # Fast-WAM (random-init, proves forward+action generation works)
  conda activate vit-probe  # or whichever env runs profile_fastwam.py
  cd /data1/ybyang/FastWAM
  python /data1/ybyang/vlla/scripts/wam_demo_reproduce.py fastwam \\
      --gpu 0 \\
      --output /data1/ybyang/vlla/output/FastWAM/official_demo/demo_reproduce/demo_report.json

  # LingBot-VA (random-init, Full WAM)
  cd /data1/ybyang/lingbot-va
  python /data1/ybyang/vlla/scripts/wam_demo_reproduce.py lingbot_va \\
      --gpu 0 \\
      --output /data1/ybyang/vlla/output/lingbot-va/official_demo/demo_reproduce/demo_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


def _check_vla_action(actions: torch.Tensor, expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """VLA action-tensor validation — mirrors src/tasks/demo_reproduce_task.py."""
    checks = []

    # shape
    action_shape = list(actions.shape)
    expected_shape = expected.get("action_shape")
    if expected_shape:
        checks.append({
            "name": "action_shape",
            "passed": action_shape == list(expected_shape),
            "actual": action_shape,
            "expected": expected_shape,
        })

    expected_ndim = expected.get("action_ndim")
    if expected_ndim:
        checks.append({
            "name": "action_ndim",
            "passed": actions.ndim == expected_ndim,
            "actual": actions.ndim,
            "expected": expected_ndim,
        })

    # NaN / Inf
    has_nan = torch.isnan(actions).any().item()
    has_inf = torch.isinf(actions).any().item()
    checks.append({
        "name": "action_no_nan_inf",
        "passed": not has_nan and not has_inf,
        "actual": f"nan={has_nan}, inf={has_inf}",
        "expected": "no NaN or Inf",
    })

    # clip range
    clip_value = expected.get("action_clip_value")
    if clip_value:
        in_range = (actions.abs() <= clip_value + 1e-3).all().item()
        checks.append({
            "name": "action_value_range",
            "passed": in_range,
            "actual": f"max_abs={actions.abs().max().item():.4f}",
            "expected": f"<= {clip_value}",
        })

    return checks


def run_fastwam(gpu: int) -> Dict[str, Any]:
    """Run single Fast-WAM forward pass and check action tensor."""
    sys.path.insert(0, "/data1/ybyang/FastWAM")  # for import
    from fastwam.wan_action_dit import ActionDiT  # noqa: F401

    # Replicate profile_fastwam.py::build_model_random minimal path
    import profile_fastwam as pfw  # type: ignore

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    # Build model + dummy inputs (same code path as profile_fastwam)
    model = pfw.build_model_random(device, dtype, action_dim=7, proprio_dim=7)

    model.eval()
    with torch.no_grad():
        # Simplified: model.run_inference(...) returns action tensor
        # We just invoke the same thing profile_fastwam's measurement block calls.
        # Look at profile_fastwam main() for the forward call we need:
        # typically something like model(frame, proprio, text_emb) → action
        # Fall back to inspecting model's expected interface
        raise NotImplementedError(
            "Please inspect profile_fastwam.py to extract the exact forward call"
        )


# The above is a placeholder. Simpler approach: piggyback on profile_fastwam's
# own output (action_chunk). We'll just patch profile_fastwam to save action.


def run_from_profile_log(script_name: str, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Simpler alternative: no re-run. Derive demo_report directly from
    profile_fastwam.py / profile_lingbot_va.py side-effects on action
    tensor shape/dtype which they already print.

    Returns a synthetic demo_report based on the standalone-script's
    architecture metadata.
    """
    raise NotImplementedError("use run_fastwam_inline instead")


def run_fastwam_inline(gpu: int) -> torch.Tensor:
    """Fast-WAM: invoke profile_fastwam.build_model_random + one forward via infer_action."""
    sys.path.insert(0, "/data1/ybyang/FastWAM")
    sys.path.insert(0, "/data1/ybyang/FastWAM/src")
    sys.path.insert(0, "/data1/ybyang/vlla/scripts")
    import profile_fastwam as pfw  # type: ignore

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    model = pfw.build_model_random(device, dtype, action_dim=7, proprio_dim=7)
    model.eval()

    # Inputs matching profile_fastwam: 2-cam 224x224 horizontal concat
    height, width = 224, 448
    dummy_image = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    dummy_proprio = torch.randn(1, 7, device=device, dtype=dtype)
    dummy_context = torch.randn(1, 20, 4096, device=device, dtype=dtype)
    dummy_context_mask = torch.ones(1, 20, device=device, dtype=torch.bool)

    out = model.infer_action(
        prompt=None,
        input_image=dummy_image,
        action_horizon=10,
        proprio=dummy_proprio,
        context=dummy_context,
        context_mask=dummy_context_mask,
        num_inference_steps=10,
    )

    actions = out["action"]
    return actions


def run_lingbot_va_inline(gpu: int) -> torch.Tensor:
    """LingBot-VA: build random transformer + run action denoise loop (same as profile script)."""
    sys.path.insert(0, "/data1/ybyang/lingbot-va")
    sys.path.insert(0, "/data1/ybyang/vlla/scripts")

    from einops import rearrange
    import torch.nn.functional as F

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    from wan_va.modules.model import WanTransformer3DModel
    from wan_va.utils import FlowMatchScheduler, get_mesh_id

    transformer_config = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "in_channels": 48,
        "out_channels": 48,
        "action_dim": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 14336,
        "num_layers": 30,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 1024,
        "attn_mode": "torch",
    }
    transformer = WanTransformer3DModel(**transformer_config).to(device=device, dtype=dtype)
    transformer.eval()

    action_dim = 30
    action_per_frame = 4
    frame_chunk_size = 4
    action_steps = 10

    prompt_embeds = torch.randn(1, 512, 4096, device=device, dtype=dtype)
    action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)

    with torch.no_grad():
        actions = torch.randn(
            1, action_dim, frame_chunk_size, action_per_frame, 1,
            device=device, dtype=dtype,
        )
        action_scheduler.set_timesteps(action_steps)
        action_timesteps = action_scheduler.timesteps
        action_timesteps = F.pad(action_timesteps, (0, 1), mode='constant', value=0)

        for step_i, t in enumerate(action_timesteps[:-1]):
            grid_id = get_mesh_id(
                actions.shape[-3], actions.shape[-2], actions.shape[-1],
                1, 1, 0, action=True,
            ).to(device)
            action_timestep_vec = torch.ones(
                [1, actions.shape[2]], dtype=torch.float32, device=device,
            ) * t
            input_dict = {
                "noisy_latents": actions,
                "timesteps": action_timestep_vec,
                "grid_id": grid_id[None],
                "text_emb": prompt_embeds.clone(),
            }
            action_pred = transformer(input_dict, update_cache=0, cache_name="pos", action_mode=True)
            action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
            actions = action_scheduler.step(action_pred, t, actions, return_dict=False)

    # Flatten to [B, horizon, action_dim] for validation
    actions_flat = rearrange(actions, "b c f n 1 -> b (f n) c")
    return actions_flat


EXPECTED = {
    "fastwam": {
        "model_type": "vla",
        "action_ndim": 2,  # [horizon, action_dim] — infer_action squeezes batch
    },
    "lingbot_va": {
        "model_type": "vla",
        "action_ndim": 3,  # [B, horizon, action_dim]
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["fastwam", "lingbot_va"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)

    print(f"\n=== WAM demo_reproduce: {args.model} on GPU {args.gpu} ===")
    try:
        if args.model == "fastwam":
            actions = run_fastwam_inline(args.gpu)
        else:
            actions = run_lingbot_va_inline(args.gpu)
    except Exception as e:
        report = {"status": "ERROR", "model_type": "vla", "message": str(e)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.output, "w"), indent=2)
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"Actions shape: {list(actions.shape)}, dtype: {actions.dtype}")

    expected = EXPECTED[args.model]
    checks = _check_vla_action(actions, expected)

    num_passed = sum(1 for c in checks if c["passed"])
    status = "PASS" if num_passed == len(checks) else "FAIL"

    report = {
        "status": status,
        "model_type": expected["model_type"],
        "num_checks": len(checks),
        "num_passed": num_passed,
        "checks": checks,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)

    print("\n" + "=" * 50)
    print(f"Result: {status} ({num_passed}/{len(checks)})")
    for c in checks:
        mark = "✓" if c["passed"] else "✗"
        print(f"  {mark} {c['name']}: {c.get('actual','')} (expected {c.get('expected','')})")
    print(f"\nSaved to: {args.output}")

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
