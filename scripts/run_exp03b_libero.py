#!/usr/bin/env python3
"""
exp03b: LingBot-VLA-4B LIBERO-4 eval.

Loads LingBotVlaPolicy, runs closed-loop eval on 4 LIBERO suites.
Each suite has 10 tasks × N episodes.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_exp03b_libero.py --episodes 20
  CUDA_VISIBLE_DEVICES=0 python scripts/run_exp03b_libero.py --episodes 2 --suite libero_spatial
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data1/ybyang/lingbot-vla")
sys.path.insert(0, "/data1/ybyang/vlla")

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "/data1/ybyang/huggingface"


def load_policy(ckpt_path, qwen_path, device="cuda:0"):
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy

    try:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
    except ImportError:
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config

    from safetensors import safe_open
    import dataclasses

    with open(os.path.join(ckpt_path, "config.json")) as f:
        config_dict = json.load(f)

    valid_fields = {f.name for f in dataclasses.fields(PI0Config)}
    pi0_kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}
    pi0_config = PI0Config(**pi0_kwargs)
    for k, v in config_dict.items():
        if not hasattr(pi0_config, k):
            setattr(pi0_config, k, v)
    pi0_config.tokenizer_path = qwen_path
    pi0_config.attention_implementation = "eager"

    defaults = {
        "enable_expert_vision": False, "expert_vision_type": None,
        "train_expert_only": False, "loss_type": "l2", "align_params": {},
        "adanorm_time": False, "split_gate_liner": False,
        "nosplit_gate_liner": False, "separate_time_proj": False,
        "old_adanorm": False, "final_norm_adanorm": False, "norm_qkv": False,
        "action_dim": config_dict.get("max_action_dim", 75),
        "vlm_repo_id": None, "expert_vision_path": None,
        "incremental_training": False, "depth_incremental_training": False,
        "post_training": False,
    }
    for k, v in defaults.items():
        if not hasattr(pi0_config, k):
            setattr(pi0_config, k, v)

    policy = LingbotVlaPolicy(config=pi0_config, tokenizer_path=qwen_path, eval=True)

    from glob import glob
    all_safetensors = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))
    merged = {}
    for fpath in all_safetensors:
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged[key] = f.get_tensor(key)

    missing, unexpected = policy.load_state_dict(merged, strict=False)
    print(f"Loaded {len(merged)} tensors ({len(missing)} missing, {len(unexpected)} unexpected)")

    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    return policy, config_dict


def make_obs(env_obs, policy, config, device):
    from PIL import Image

    img = env_obs["agentview_image"]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    patch_size = 14
    temporal = 2
    img_size = config.get("resize_imgs_with_padding", [224, 224])
    grid_h = img_size[0] // patch_size
    grid_w = img_size[1] // patch_size
    num_patches = grid_h * grid_w
    patch_dim = 3 * temporal * patch_size * patch_size

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((img_size[1], img_size[0]))
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_patches = img_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    img_patches = img_patches.contiguous().view(1, 3, num_patches, patch_size * patch_size)
    img_patches = img_patches.permute(0, 2, 1, 3).reshape(1, num_patches, 3 * patch_size * patch_size)

    pad_to = patch_dim
    if img_patches.shape[-1] < pad_to:
        img_patches = torch.nn.functional.pad(img_patches, (0, pad_to - img_patches.shape[-1]))

    images = img_patches.unsqueeze(1).to(device=device, dtype=torch.bfloat16)
    img_masks = torch.ones(1, 1, dtype=torch.bool, device=device)

    state_dim = config.get("max_state_dim", 75)
    ee_pos = env_obs.get("robot0_eef_pos", np.zeros(3))
    ee_quat = env_obs.get("robot0_eef_quat", np.zeros(4))
    gripper = env_obs.get("robot0_gripper_qpos", np.zeros(2))
    raw_state = np.concatenate([ee_pos, ee_quat, gripper])
    state = np.zeros(state_dim, dtype=np.float32)
    state[:len(raw_state)] = raw_state
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    return images, img_masks, state_tensor


def run_suite(policy, config, suite_name, num_episodes, device, out_dir):
    import libero.libero.benchmark as benchmark

    bench = benchmark.get_benchmark(suite_name)()
    task_names = bench.get_task_names()
    num_tasks = len(task_names)

    tokenizer = policy.language_tokenizer
    max_len = config.get("tokenizer_max_length", 72)

    results = {}
    total_success = 0
    total_episodes = 0

    for task_id in range(num_tasks):
        task = bench.get_task(task_id)
        task_name = task_names[task_id]
        env = bench.get_env(task_id)

        instruction = task.language
        lang = tokenizer(instruction, return_tensors="pt", padding="max_length",
                         max_length=max_len, truncation=True)
        lang_tokens = lang["input_ids"].to(device)
        lang_masks = lang["attention_mask"].to(device)

        successes = 0
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0
            max_steps = 600

            while not done and step < max_steps:
                images, img_masks, state = make_obs(obs, policy, config, device)

                with torch.no_grad():
                    actions = policy.model.sample_actions(
                        images=images,
                        img_masks=img_masks,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=state,
                    )

                action_np = actions[0].float().cpu().numpy()
                action_chunk = config.get("n_action_steps", 50)

                for a_idx in range(min(action_chunk, len(action_np))):
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
        results[task_name] = {"success_rate": success_rate, "successes": successes, "episodes": num_episodes}
        total_success += successes
        print(f"  [{task_id+1}/{num_tasks}] {task_name}: {success_rate:.1%} ({successes}/{num_episodes})")

    avg = total_success / total_episodes if total_episodes > 0 else 0
    results["_average"] = {"success_rate": avg, "total_success": total_success, "total_episodes": total_episodes}
    print(f"  {suite_name} average: {avg:.1%}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{suite_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/data1/ybyang/huggingface/robbyant/lingbot-vla-4b")
    parser.add_argument("--qwen", default="/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--out", default="/data1/ybyang/vlla/exp/exp03b")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    suites = [args.suite] if args.suite else ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    print(f"Loading LingBot-VLA from {args.ckpt}...")
    policy, config = load_policy(args.ckpt, args.qwen, args.device)
    print("Model loaded.")

    all_results = {}
    for suite in suites:
        print(f"\n=== {suite} ({args.episodes} episodes/task) ===")
        avg = run_suite(policy, config, suite, args.episodes, args.device, args.out)
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
