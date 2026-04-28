#!/usr/bin/env python
"""exp09a — Cosmos Policy direct-mode latency profiling.

Runs inside the cosmos-policy environment (venv or Docker).
Two measurement modes:
  1. End-to-end: CUDA events around the official get_action() call
  2. Phase breakdown: torch.profiler trace for VAE/DiT/extract decomposition

Fills the gap that arXiv:2601.16163 never reported:
direct-mode single-chunk inference latency.

Usage:
    # Canonical 5-step profiling
    python scripts/exp09a_cosmos_policy_profiling.py --gpu 0

    # Step sweep
    for N in 1 2 5 10 20; do
        python scripts/exp09a_cosmos_policy_profiling.py \
            --gpu 0 --denoise-steps $N \
            --output exp/exp09a/results_steps_${N}.json
    done

    # With torch.profiler trace
    python scripts/exp09a_cosmos_policy_profiling.py --gpu 0 --trace
"""

import argparse
import json
import os
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
    """Create a fake LIBERO observation for profiling (no sim required)."""
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


def profile_e2e(get_action_fn, model, cfg, dataset_stats, obs, text_emb, denoise_steps):
    """One end-to-end call with CUDA event timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    get_action_fn(
        cfg, model, dataset_stats, obs,
        text_emb,
        num_denoising_steps_action=denoise_steps,
        generate_future_state_and_value_in_parallel=True,
    )

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description="exp09a — Cosmos Policy profiling")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace", action="store_true",
                        help="Save torch.profiler trace to exp/exp09a/trace.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    print(f"[config] GPU={args.gpu}, warmup={args.warmup}, iter={args.iterations}, "
          f"steps={args.denoise_steps}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        get_t5_embedding_from_cache,
        load_dataset_stats,
    )

    cfg = build_config(args.denoise_steps)

    print("[init] Loading dataset stats...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print("[init] Loading T5 text embeddings...")
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    print("[init] Loading Cosmos-Predict2-2B model...")
    t0 = time.time()
    model, _ = get_model(cfg)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {param_count:.2f}B")

    task_desc = "put both the alphabet soup and the tomato sauce in the basket"
    text_emb = task_desc
    obs = make_synthetic_observation()

    # Warmup
    print(f"\n[warmup] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        profile_e2e(get_action, model, cfg, dataset_stats, obs, text_emb, args.denoise_steps)
        if i == 0:
            torch.cuda.synchronize()
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  Peak VRAM after first call: {mem_mb:.0f} MB")
        if (i + 1) % 5 == 0:
            print(f"  warmup {i+1}/{args.warmup}")

    # Benchmark
    print(f"\n[benchmark] Running {args.iterations} timed iterations...")
    all_ms = []
    for i in range(args.iterations):
        ms = profile_e2e(get_action, model, cfg, dataset_stats, obs, text_emb, args.denoise_steps)
        all_ms.append(ms)
        if (i + 1) % 5 == 0:
            med = statistics.median(all_ms)
            print(f"  iter {i+1}/{args.iterations}: median so far = {med:.1f} ms")

    # Optional torch.profiler trace
    if args.trace:
        print("\n[trace] Running profiler trace (1 iteration)...")
        trace_dir = VLLA_ROOT / "exp" / "exp09a"
        trace_dir.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            get_action(
                cfg, model, dataset_stats, obs,
                text_emb,
                num_denoising_steps_action=args.denoise_steps,
                generate_future_state_and_value_in_parallel=True,
            )
        trace_path = trace_dir / f"trace_steps_{args.denoise_steps}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"  Trace saved to {trace_path}")

        print("\n  Top 20 CUDA operations:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Results
    median_ms = statistics.median(all_ms)
    hz = 1000.0 / median_ms if median_ms > 0 else float("inf")
    chunk_duration_s = cfg.chunk_size / 25.0
    effective_hz = 1.0 / chunk_duration_s if median_ms < chunk_duration_s * 1000 else hz

    output = {
        "experiment": "exp09a",
        "model": "Cosmos-Policy-LIBERO-Predict2-2B",
        "model_params_B": round(param_count, 2),
        "denoise_steps": args.denoise_steps,
        "chunk_size": cfg.chunk_size,
        "chunk_duration_s": chunk_duration_s,
        "image_size": 224,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "e2e": {
            "median_ms": round(median_ms, 2),
            "mean_ms": round(statistics.mean(all_ms), 2),
            "std_ms": round(statistics.stdev(all_ms), 2) if len(all_ms) > 1 else 0,
            "min_ms": round(min(all_ms), 2),
            "max_ms": round(max(all_ms), 2),
            "all_ms": [round(v, 2) for v in all_ms],
        },
        "policy_query_hz": round(hz, 2),
        "effective_control_hz": round(effective_hz, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }

    print(f"\n{'='*60}")
    print(f"COSMOS POLICY PROFILING — direct mode, {args.denoise_steps} steps")
    print(f"{'='*60}")
    print(f"  End-to-end median:  {median_ms:>8.1f} ms")
    print(f"  Policy query Hz:    {hz:>8.1f}")
    print(f"  Chunk = {cfg.chunk_size} steps @ 25Hz = {chunk_duration_s:.2f}s")
    print(f"  Effective control:  {effective_hz:>8.1f} Hz (if query < chunk)")
    print(f"  Peak VRAM:          {output['peak_vram_mb']} MB")
    print(f"  Parameters:         {param_count:.2f}B")

    out_path = args.output or str(VLLA_ROOT / "exp" / "exp09a" / "results_profiling.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
