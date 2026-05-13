#!/usr/bin/env python3
"""
Cosmos Policy LIBERO-4 eval wrapper.

Thin wrapper around the vendor's run_libero_eval.py that provides sane defaults
matching the paper's 98.5% config. Runs via draccus (the vendor's CLI framework).

Two modes:
  1. Direct vendor invocation (default) — calls run_libero_eval.py as a module
  2. Standalone wrapper (--standalone) — uses our own env loop with get_action()

Usage (vendor mode):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cosmos_libero.py --suite libero_spatial --episodes 20
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cosmos_libero.py --all --episodes 20

Usage (standalone mode for debugging):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cosmos_libero.py --standalone --suite libero_spatial --episodes 2
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VLLA_ROOT = SCRIPT_DIR.parent
COSMOS_REPO = VLLA_ROOT / "vendor" / "cosmos-policy"

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "/data1/ybyang/huggingface"

CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
CONFIG = "cosmos_predict2_2b_480p_libero__inference_only"
CONFIG_FILE = "cosmos_policy/config/config.py"
DATASET_STATS = f"{CKPT}/libero_dataset_statistics.json"
T5_EMBEDDINGS = f"{CKPT}/libero_t5_embeddings.pkl"

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def run_vendor_mode(suite, episodes, out_dir, seed=195):
    """Run via the vendor's draccus-based eval script."""
    cmd = [
        sys.executable, "-m",
        "cosmos_policy.experiments.robot.libero.run_libero_eval",
        "--config", CONFIG,
        "--ckpt_path", CKPT,
        "--config_file", CONFIG_FILE,
        "--use_wrist_image", "True",
        "--use_proprio", "True",
        "--normalize_proprio", "True",
        "--unnormalize_actions", "True",
        "--dataset_stats_path", DATASET_STATS,
        "--t5_text_embeddings_path", T5_EMBEDDINGS,
        "--trained_with_image_aug", "True",
        "--chunk_size", "16",
        "--num_open_loop_steps", "16",
        "--task_suite_name", suite,
        "--num_trials_per_task", str(episodes),
        "--local_log_dir", out_dir,
        "--randomize_seed", "False",
        "--data_collection", "False",
        "--seed", str(seed),
        "--use_variance_scale", "False",
        "--deterministic", "True",
        "--ar_future_prediction", "False",
        "--ar_value_prediction", "False",
        "--use_jpeg_compression", "True",
        "--flip_images", "True",
        "--num_denoising_steps_action", "5",
        "--num_denoising_steps_future_state", "1",
        "--num_denoising_steps_value", "1",
        "--run_id_note", f"vlla-eval--{suite}--{episodes}ep--seed{seed}",
        "--use_wandb", "False",
    ]

    print(f"[cosmos] Running vendor eval: {suite} ({episodes} ep, seed={seed})")
    env = {**os.environ, "PYTHONPATH": str(COSMOS_REPO)}

    result = subprocess.run(cmd, cwd=str(COSMOS_REPO), env=env)
    return result.returncode


def run_standalone_mode(suite, episodes, out_dir, seed=195):
    """Standalone eval using get_action() + our own LIBERO env loop."""
    sys.path.insert(0, str(COSMOS_REPO))
    sys.path.insert(0, str(VLLA_ROOT))

    # Patch hf_hub_download for firewalled servers
    from scripts.exp09a_cosmos_policy_profiling import _patch_hf_downloads
    _patch_hf_downloads()

    import torch
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
    )
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
    from cosmos_policy.utils.utils import set_seed_everywhere
    from dataclasses import dataclass

    @dataclass
    class EvalCfg:
        suite: str = "libero"
        model_family: str = "cosmos"
        config: str = CONFIG
        ckpt_path: str = CKPT
        config_file: str = CONFIG_FILE
        dataset_stats_path: str = DATASET_STATS
        t5_text_embeddings_path: str = T5_EMBEDDINGS
        use_third_person_image: bool = True
        num_third_person_images: int = 1
        use_wrist_image: bool = True
        num_wrist_images: int = 1
        use_proprio: bool = True
        flip_images: bool = True
        use_variance_scale: bool = False
        use_jpeg_compression: bool = True
        trained_with_image_aug: bool = True
        normalize_proprio: bool = True
        unnormalize_actions: bool = True
        chunk_size: int = 16
        num_open_loop_steps: int = 16
        num_denoising_steps_action: int = 5
        num_denoising_steps_future_state: int = 1
        num_denoising_steps_value: int = 1
        ar_future_prediction: bool = False
        ar_value_prediction: bool = False
        ar_qvalue_prediction: bool = False
        planning_model_config_name: str = ""
        planning_model_ckpt_path: str = ""
        seed: int = 195
        randomize_seed: bool = False
        deterministic: bool = True
        task_suite_name: str = "libero_spatial"

    cfg = EvalCfg(seed=seed, task_suite_name=suite)

    set_seed_everywhere(cfg.seed)

    print(f"[cosmos-standalone] Loading model...")
    model = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    resize_size = get_image_resize_size(cfg)

    cfg.unnorm_key = suite
    if suite not in model.norm_stats and f"{suite}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{suite}_no_noops"

    import libero.libero.benchmark as benchmark_mod
    bench = benchmark_mod.get_benchmark(suite)()
    task_names = bench.get_task_names()

    results = {}
    total_success = 0
    total_episodes = 0
    max_steps = TASK_MAX_STEPS.get(suite, 520)

    for task_id in range(len(task_names)):
        task = bench.get_task(task_id)
        task_name = task_names[task_id]
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        initial_states = bench.get_task_init_states(task_id)

        successes = 0
        for ep in range(episodes):
            if cfg.deterministic:
                set_seed_everywhere(cfg.seed)

            env.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])

            from collections import deque
            action_queue = deque(maxlen=cfg.num_open_loop_steps)

            NUM_STEPS_WAIT = 10
            t = 0
            ep_success = False

            while t < max_steps + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                if len(action_queue) == 0:
                    img = get_libero_image(obs, cfg.flip_images)
                    wrist_img = get_libero_wrist_image(obs, cfg.flip_images)
                    observation = {
                        "primary_image": img,
                        "wrist_image": wrist_img,
                        "proprio": np.concatenate((
                            obs["robot0_gripper_qpos"],
                            obs["robot0_eef_pos"],
                            obs["robot0_eef_quat"],
                        )),
                    }

                    with torch.no_grad():
                        action_return = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True,
                        )
                    action_queue.extend(action_return["actions"])

                action = action_queue.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    ep_success = True
                    break
                t += 1

            if ep_success:
                successes += 1
            total_episodes += 1

        success_rate = successes / episodes
        results[task_name] = {
            "success_rate": success_rate,
            "successes": successes,
            "episodes": episodes,
        }
        total_success += successes
        print(f"  [{task_id+1}/{len(task_names)}] {task_name}: {success_rate:.1%} ({successes}/{episodes})")

    avg = total_success / total_episodes if total_episodes > 0 else 0
    results["_average"] = {
        "success_rate": avg,
        "total_success": total_success,
        "total_episodes": total_episodes,
    }
    print(f"  {suite} average: {avg:.1%}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{suite}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--out", default="/data1/ybyang/vlla/exp/cosmos_libero")
    parser.add_argument("--standalone", action="store_true",
                        help="Use standalone env loop instead of vendor draccus CLI")
    args = parser.parse_args()

    if args.all:
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    elif args.suite:
        suites = [args.suite]
    else:
        print("Specify --suite or --all")
        sys.exit(1)

    all_results = {}
    for suite in suites:
        print(f"\n=== {suite} ({args.episodes} episodes/task) ===")
        if args.standalone:
            avg = run_standalone_mode(suite, args.episodes, args.out, args.seed)
        else:
            rc = run_vendor_mode(suite, args.episodes, args.out, args.seed)
            avg = None
            if rc != 0:
                print(f"  {suite} vendor eval exited with code {rc}")
        all_results[suite] = avg

    print(f"\n=== Final ===")
    for s, a in all_results.items():
        label = f"{a:.1%}" if a is not None else "see vendor logs"
        print(f"  {s}: {label}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "libero_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
