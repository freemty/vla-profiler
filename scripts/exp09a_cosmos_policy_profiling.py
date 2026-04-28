#!/usr/bin/env python
"""exp09a — Cosmos Policy direct-mode latency profiling.

Runs inside the cosmos-policy environment (Docker or uv venv).
Measures the three inference phases:
  E: VAE encode (image → latent)
  D: DiT denoise loop (EDM solver, default 5 steps)
  X: Extract action from latent sequence

Fills the gap that the paper (arXiv:2601.16163) never reported:
direct-mode single-chunk inference latency.

Usage (inside cosmos-policy Docker on xdlab23):
    python /data1/ybyang/vlla/scripts/exp09a_cosmos_policy_profiling.py \
        --gpu 0 --warmup 15 --iterations 20 --denoise-steps 5

Output: exp/exp09a/results_profiling.json
"""

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLA_ROOT = SCRIPT_DIR.parent

COSMOS_REPO = VLLA_ROOT / "vendor" / "cosmos-policy"
if str(COSMOS_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS_REPO))


def make_synthetic_observation(image_size: int = 224):
    """Create a fake LIBERO observation for profiling (no sim dependency)."""
    return {
        "primary_image": np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "proprio": np.random.randn(14).astype(np.float32),
    }


def build_config(denoise_steps: int = 5):
    """Build a minimal PolicyEvalConfig for LIBERO direct inference."""
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=denoise_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def profile_single_call(model, cfg, dataset_stats, obs, text_embedding, denoise_steps):
    """Run one inference call with CUDA event timing around each phase.

    Returns dict with phase timings in ms.
    """
    from cosmos_policy.experiments.robot.cosmos_utils import (
        COSMOS_IMAGE_SIZE,
        COSMOS_TEMPORAL_COMPRESSION_FACTOR,
        extract_action_chunk_from_latent_sequence,
        prepare_images_for_model,
        rescale_proprio,
    )
    from cosmos_policy.constants import ACTION_DIM
    from cosmos_policy.utils.utils import duplicate_array

    device = next(model.parameters()).device

    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)
    d_start = torch.cuda.Event(enable_timing=True)
    d_end = torch.cuda.Event(enable_timing=True)
    x_start = torch.cuda.Event(enable_timing=True)
    x_end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        # --- Phase E: Prepare + VAE encode ---
        e_start.record()

        all_camera_images = [obs["wrist_image"], obs["primary_image"]]
        all_camera_images = prepare_images_for_model(all_camera_images, cfg)

        proprio = obs["proprio"]
        if cfg.normalize_proprio:
            proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

        primary_image = all_camera_images[1]
        blank_image = np.zeros_like(primary_image)

        image_sequence = []
        idx = 0

        image_sequence.append(np.expand_dims(np.zeros_like(blank_image), axis=0))
        idx += 1

        blank_dup = duplicate_array(blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.append(blank_dup)
        current_proprio_idx = idx
        idx += 1

        wrist_dup = duplicate_array(all_camera_images[0], total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.append(wrist_dup)
        idx += 1

        primary_dup = duplicate_array(primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.append(primary_dup)
        idx += 1

        image_sequence.append(blank_dup.copy())
        action_idx = idx
        idx += 1

        image_sequence.append(blank_dup.copy())
        future_proprio_idx = idx
        idx += 1

        image_sequence.append(wrist_dup.copy())
        idx += 1

        image_sequence.append(primary_dup.copy())
        idx += 1

        image_sequence.append(blank_dup.copy())
        value_idx = idx
        idx += 1

        raw_seq = np.concatenate(image_sequence, axis=0)
        raw_seq = np.expand_dims(raw_seq, axis=0)
        raw_seq = np.transpose(raw_seq, (0, 4, 1, 2, 3))
        raw_seq = torch.from_numpy(raw_seq).to(dtype=torch.uint8, device=device)

        proprio_tensor = torch.from_numpy(proprio).reshape(1, -1).to(dtype=torch.bfloat16, device=device)

        data_batch = {
            "dataset_name": "video_data",
            "video": raw_seq,
            "t5_text_embeddings": text_embedding.to(dtype=torch.bfloat16, device=device),
            "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
            "padding_mask": torch.zeros((1, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16, device=device),
            "num_conditional_frames": model.config.min_num_conditional_frames,
            "proprio": proprio_tensor,
            "current_proprio_latent_idx": torch.tensor([current_proprio_idx], dtype=torch.int64, device=device),
            "current_wrist_image_latent_idx": torch.tensor([2], dtype=torch.int64, device=device),
            "current_wrist_image2_latent_idx": torch.tensor([-1], dtype=torch.int64, device=device),
            "current_image_latent_idx": torch.tensor([3], dtype=torch.int64, device=device),
            "current_image2_latent_idx": torch.tensor([-1], dtype=torch.int64, device=device),
            "action_latent_idx": torch.tensor([action_idx], dtype=torch.int64, device=device),
            "future_proprio_latent_idx": torch.tensor([future_proprio_idx], dtype=torch.int64, device=device),
            "future_wrist_image_latent_idx": torch.tensor([6], dtype=torch.int64, device=device),
            "future_wrist_image2_latent_idx": torch.tensor([-1], dtype=torch.int64, device=device),
            "future_image_latent_idx": torch.tensor([7], dtype=torch.int64, device=device),
            "future_image2_latent_idx": torch.tensor([-1], dtype=torch.int64, device=device),
            "value_latent_idx": torch.tensor([value_idx], dtype=torch.int64, device=device),
        }

        e_end.record()

        # --- Phase D: DiT denoise (generate_samples_from_batch) ---
        d_start.record()

        generated_latent, orig_clean = model.generate_samples_from_batch(
            data_batch,
            n_sample=1,
            num_steps=denoise_steps,
            seed=42,
            is_negative_prompt=False,
            use_variance_scale=False,
            return_orig_clean_latent_frames=True,
        )

        d_end.record()

        # --- Phase X: Extract action ---
        x_start.record()

        action_indices = torch.full((1,), action_idx, dtype=torch.int64, device=device)
        actions = extract_action_chunk_from_latent_sequence(
            generated_latent, action_shape=(cfg.chunk_size, ACTION_DIM), action_indices=action_indices
        ).to(torch.float32).cpu().numpy()

        x_end.record()

    torch.cuda.synchronize()

    return {
        "encode_ms": e_start.elapsed_time(e_end),
        "denoise_ms": d_start.elapsed_time(d_end),
        "extract_ms": x_start.elapsed_time(x_end),
        "total_ms": e_start.elapsed_time(x_end),
    }


def main():
    parser = argparse.ArgumentParser(description="exp09a — Cosmos Policy profiling")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    print(f"[config] GPU={args.gpu}, warmup={args.warmup}, iter={args.iterations}, steps={args.denoise_steps}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        get_t5_embedding_from_cache,
        load_dataset_stats,
    )

    cfg = build_config(args.denoise_steps)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, cosmos_config = get_model(cfg)

    task_desc = "put both the alphabet soup and the tomato sauce in the basket"
    text_embedding = get_t5_embedding_from_cache(task_desc)
    if text_embedding.dim() == 2:
        text_embedding = text_embedding.unsqueeze(0)

    obs = make_synthetic_observation()

    print(f"\n[warmup] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        profile_single_call(model, cfg, dataset_stats, obs, text_embedding, args.denoise_steps)
        if i == 0:
            torch.cuda.synchronize()
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  Peak VRAM after first call: {mem_mb:.0f} MB")

    print(f"\n[benchmark] Running {args.iterations} timed iterations...")
    results = {"encode_ms": [], "denoise_ms": [], "extract_ms": [], "total_ms": []}
    for i in range(args.iterations):
        r = profile_single_call(model, cfg, dataset_stats, obs, text_embedding, args.denoise_steps)
        for k in results:
            results[k].append(r[k])
        if (i + 1) % 5 == 0:
            med = statistics.median(results["total_ms"])
            print(f"  iter {i+1}/{args.iterations}: total median so far = {med:.1f} ms")

    summary = {}
    for phase in ["encode", "denoise", "extract", "total"]:
        key = f"{phase}_ms"
        vals = results[key]
        summary[phase] = {
            "median_ms": round(statistics.median(vals), 2),
            "mean_ms": round(statistics.mean(vals), 2),
            "std_ms": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
            "min_ms": round(min(vals), 2),
            "max_ms": round(max(vals), 2),
            "all_ms": [round(v, 2) for v in vals],
        }

    total_median = summary["total"]["median_ms"]
    hz = 1000.0 / total_median if total_median > 0 else float("inf")

    output = {
        "experiment": "exp09a",
        "model": "Cosmos-Policy-LIBERO-Predict2-2B",
        "model_params": "2B",
        "denoise_steps": args.denoise_steps,
        "chunk_size": cfg.chunk_size,
        "image_size": 224,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "phases": summary,
        "total_median_ms": total_median,
        "hz": round(hz, 2),
        "per_step_denoise_ms": round(summary["denoise"]["median_ms"] / args.denoise_steps, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }

    print(f"\n{'='*60}")
    print(f"COSMOS POLICY PROFILING RESULTS (direct mode, {args.denoise_steps} steps)")
    print(f"{'='*60}")
    print(f"  Encode (VAE):     {summary['encode']['median_ms']:>8.1f} ms")
    print(f"  Denoise (DiT):    {summary['denoise']['median_ms']:>8.1f} ms  ({args.denoise_steps} steps, {output['per_step_denoise_ms']:.1f} ms/step)")
    print(f"  Extract (action): {summary['extract']['median_ms']:>8.1f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  Total:            {total_median:>8.1f} ms  ({hz:.1f} Hz)")
    print(f"  Peak VRAM:        {output['peak_vram_mb']} MB")

    out_path = args.output or str(VLLA_ROOT / "exp" / "exp09a" / "results_profiling.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
