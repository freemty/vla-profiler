#!/usr/bin/env python3
"""
exp07c: Pi-Zero LIBERO-4 eval via vendored open-pi-zero.

Bypasses openpi's flax/JAX server-client architecture by loading the model
directly through our PiZeroController's init_pipeline() path. Runs closed-loop
eval on 4 LIBERO suites using the same env loop as exp03b.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_exp07c_libero.py --episodes 20
  CUDA_VISIBLE_DEVICES=0 python scripts/run_exp07c_libero.py --episodes 2 --suite libero_spatial
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLA_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(VLLA_ROOT))

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "/data1/ybyang/huggingface"


def load_model(ckpt_path, device="cuda:0", denoise_steps=10):
    from easydict import EasyDict as edict
    from src.controllers.pizero_controller import PiZeroController

    cfg = edict(
        model_name=ckpt_path,
        device=device,
        denoise_steps=denoise_steps,
        use_bf16=True,
    )
    pipeline = PiZeroController.init_pipeline(cfg)
    return pipeline


def make_obs(env_obs, pipeline):
    """Convert LIBERO observation to Pi-Zero input tensors."""
    device = pipeline.device
    dtype = pipeline.dtype
    model = pipeline.model
    model_cfg = pipeline.config

    img = env_obs["agentview_image"]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    image_size = model_cfg.vision.config.image_size
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((image_size, image_size))
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    pixel_values = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    max_image_text_tokens = model_cfg.max_image_text_tokens
    num_image_tokens = model_cfg.vision.config.num_image_tokens

    input_ids = torch.full(
        (1, max_image_text_tokens),
        model_cfg.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, :num_image_tokens] = model_cfg.image_token_index

    attention_mask = torch.zeros(
        1, max_image_text_tokens, dtype=torch.long, device=device,
    )
    attention_mask[:, :num_image_tokens] = 1

    ee_pos = env_obs.get("robot0_eef_pos", np.zeros(3))
    ee_quat = env_obs.get("robot0_eef_quat", np.zeros(4))
    gripper = env_obs.get("robot0_gripper_qpos", np.zeros(2))
    raw_proprio = np.concatenate([ee_pos, ee_quat, gripper])

    proprio_dim = model_cfg.proprio_dim
    proprio = np.zeros(proprio_dim, dtype=np.float32)
    proprio[:min(len(raw_proprio), proprio_dim)] = raw_proprio[:proprio_dim]

    cond_steps = model_cfg.cond_steps
    proprios = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0)
    proprios = proprios.expand(1, cond_steps, -1).to(device=device, dtype=dtype)

    return pixel_values, input_ids, attention_mask, proprios


def infer_action(model, pixel_values, input_ids, attention_mask, proprios):
    """Run Pi-Zero inference (mirrors PiZeroController.model_inference)."""
    dtype = pixel_values.dtype
    device = pixel_values.device

    causal_mask, vlm_pos_ids, proprio_pos_ids, action_pos_ids = (
        model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
    )
    image_text_proprio_mask, action_mask = (
        model.split_full_mask_into_submasks(causal_mask)
    )
    image_text_proprio_mask = image_text_proprio_mask.to(device)
    action_mask = action_mask.to(device)
    vlm_pos_ids = vlm_pos_ids.to(device)
    proprio_pos_ids = proprio_pos_ids.to(device)
    action_pos_ids = action_pos_ids.to(device)

    bsz = pixel_values.size(0)

    inputs_embeds = model._forward_siglip_and_text_embedding(
        input_ids, pixel_values
    )
    proprio_embeds = model.proprio_encoder(proprios)

    kv_caches = model.joint_model.build_mixture_caches()
    _, kv_caches = model.joint_model(
        attention_mask=image_text_proprio_mask,
        position_ids_all={"vlm": vlm_pos_ids, "proprio": proprio_pos_ids},
        embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
        kv_caches=kv_caches,
        return_caches=True,
    )

    action = torch.randn(
        (bsz, model.horizon_steps, model.action_dim),
        device=device, dtype=dtype,
    )
    delta_t = 1.0 / model.num_inference_steps
    t = torch.zeros(bsz, device=device, dtype=dtype)

    for _ in range(model.num_inference_steps):
        time_cond = model.time_embedding(t)
        if model.action_expert_adaptive_mode:
            action_embeds = model.action_encoder(action)
        else:
            action_embeds = model.action_encoder(action, time_cond)

        action_embeds = model.joint_model(
            attention_mask=action_mask,
            position_ids_all={"action": action_pos_ids},
            embeds_all={"action": action_embeds},
            time_cond=time_cond,
            kv_caches=kv_caches,
            cache_mode="append_non_active",
        )["action"]

        action_vel = model.action_decoder(action_embeds)
        action = action + delta_t * action_vel
        t = t + delta_t

    if model.final_action_clip_value is not None:
        action = torch.clamp(
            action,
            -model.final_action_clip_value,
            model.final_action_clip_value,
        )

    return action


def run_suite(pipeline, suite_name, num_episodes, out_dir):
    import libero.libero.benchmark as benchmark_mod

    model = pipeline.model
    device = pipeline.device

    bench = benchmark_mod.get_benchmark(suite_name)()
    task_names = bench.get_task_names()
    num_tasks = len(task_names)

    results = {}
    total_success = 0
    total_episodes = 0

    for task_id in range(num_tasks):
        task = bench.get_task(task_id)
        task_name = task_names[task_id]
        env = bench.get_env(task_id)

        successes = 0
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0
            max_steps = 600

            while not done and step < max_steps:
                pixel_values, input_ids, attention_mask, proprios = make_obs(obs, pipeline)

                with torch.no_grad():
                    actions = infer_action(model, pixel_values, input_ids, attention_mask, proprios)

                action_np = actions[0].float().cpu().numpy()
                horizon = pipeline.config.horizon_steps

                for a_idx in range(min(horizon, len(action_np))):
                    if done:
                        break
                    act = action_np[a_idx][:7]
                    obs, reward, done, info = env.step(act)
                    step += 1
                    if done or info.get("success", False):
                        done = True
                        if info.get("success", False):
                            successes += 1

            total_episodes += 1

        success_rate = successes / num_episodes
        results[task_name] = {
            "success_rate": success_rate,
            "successes": successes,
            "episodes": num_episodes,
        }
        total_success += successes
        print(f"  [{task_id+1}/{num_tasks}] {task_name}: {success_rate:.1%} ({successes}/{num_episodes})")

    avg = total_success / total_episodes if total_episodes > 0 else 0
    results["_average"] = {
        "success_rate": avg,
        "total_success": total_success,
        "total_episodes": total_episodes,
    }
    print(f"  {suite_name} average: {avg:.1%}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{suite_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/data1/ybyang/huggingface/models--allenzren--open-pi-zero/open_pi_zero_bridge.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--out", default="/data1/ybyang/vlla/exp/exp07c")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    suites = [args.suite] if args.suite else [
        "libero_spatial", "libero_object", "libero_goal", "libero_10",
    ]

    print(f"Loading Pi-Zero from {args.ckpt}...")
    pipeline = load_model(args.ckpt, args.device, args.denoise_steps)
    print("Model loaded.")

    all_results = {}
    for suite in suites:
        print(f"\n=== {suite} ({args.episodes} episodes/task) ===")
        avg = run_suite(pipeline, suite, args.episodes, args.out)
        all_results[suite] = avg

    print(f"\n=== Final ===")
    for s, a in all_results.items():
        print(f"  {s}: {a:.1%}")
    if len(all_results) > 1:
        print(f"  overall: {np.mean(list(all_results.values())):.1%}")

    with open(os.path.join(args.out, "libero_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
