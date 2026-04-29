#!/usr/bin/env python
"""exp09a — Cosmos Policy direct-mode latency profiling.

Runs inside the cosmos-policy environment (venv or Docker).
Wraps the official get_action() with CUDA events for e2e timing.
Optional --trace flag for torch.profiler phase breakdown.

Does NOT require LIBERO sim — uses synthetic observations.

Usage:
    python scripts/exp09a_cosmos_policy_profiling.py --gpu 0
    python scripts/exp09a_cosmos_policy_profiling.py --gpu 0 --denoise-steps 1 --trace
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VLLA_ROOT = SCRIPT_DIR.parent

COSMOS_REPO = VLLA_ROOT / "vendor" / "cosmos-policy"
if str(COSMOS_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS_REPO))


def _patch_hf_downloads():
    """Monkey-patch hf_hub_download to resolve from local cache without network.

    cosmos-policy's make_config_v2() recursively imports all experiment configs,
    which call get_checkpoint_path("hf://nvidia/..."). On firewalled servers this
    fails even with files in cache (missing HF metadata). This patch intercepts
    hf_hub_download and returns the local blob path directly if it exists.
    """
    import huggingface_hub.file_download as hf_dl

    _original = hf_dl.hf_hub_download

    def _patched(repo_id, filename=None, *args, **kwargs):
        if filename is None:
            filename = args[0] if args else kwargs.get("filename", "")
        search_dirs = []
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            search_dirs.append(os.path.join(hf_home, "hub"))
        search_dirs.append(os.path.expanduser("~/.cache/huggingface/hub"))
        for cache_dir in search_dirs:
            repo_folder = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
            snapshots_dir = os.path.join(repo_folder, "snapshots")
            if os.path.isdir(snapshots_dir):
                snaps = sorted(
                    os.listdir(snapshots_dir),
                    key=lambda s: os.path.getmtime(os.path.join(snapshots_dir, s)),
                    reverse=True,
                )
                for snap in snaps:
                    candidate = os.path.join(snapshots_dir, snap, filename)
                    if os.path.exists(candidate):
                        print(f"[hf-patch] {repo_id}/{filename} → {candidate}")
                        return candidate
        import warnings
        dummy = f"/tmp/_hf_dummy_{repo_id.replace('/', '_')}_{filename.replace('/', '_')}"
        Path(dummy).parent.mkdir(parents=True, exist_ok=True)
        Path(dummy).touch()
        warnings.warn(
            f"[hf-patch] MISS: {repo_id}/{filename} — created empty dummy. "
            "If this file is read during inference (not just config registration), "
            "results will be invalid."
        )
        return dummy

    hf_dl.hf_hub_download = _patched




@dataclass
class ProfileConfig:
    """Minimal config mirroring PolicyEvalConfig fields needed by get_action().
    Avoids importing run_libero_eval.py which requires the full LIBERO sim."""
    suite: str = "libero"
    model_family: str = "cosmos"
    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
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
    seed: int = 7
    randomize_seed: bool = False
    deterministic: bool = True


def make_synthetic_observation(image_size: int = 224):
    """Fake LIBERO observation for profiling (no sim required)."""
    return {
        "primary_image": np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "proprio": np.random.randn(9).astype(np.float32),
    }


def profile_e2e(get_action_fn, model, cfg, dataset_stats, obs, text_emb,
                denoise_steps, parallel_gen=True):
    """One e2e call with CUDA event timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    get_action_fn(
        cfg, model, dataset_stats, obs,
        text_emb,
        num_denoising_steps_action=denoise_steps,
        generate_future_state_and_value_in_parallel=parallel_gen,
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
                        help="Save torch.profiler trace for phase breakdown")
    parser.add_argument("--local-ckpt", type=str, default=None,
                        help="Local path to checkpoint snapshot dir (bypasses HF download)")
    parser.add_argument("--no-parallel-gen", action="store_true",
                        help="Skip future state + value generation (pure action-only timing)")
    args = parser.parse_args()

    _patch_hf_downloads()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    parallel_gen = not args.no_parallel_gen
    print(f"[config] GPU={args.gpu}, warmup={args.warmup}, iter={args.iterations}, "
          f"steps={args.denoise_steps}, parallel_gen={parallel_gen}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    cfg = ProfileConfig(num_denoising_steps_action=args.denoise_steps)

    if args.local_ckpt:
        local_snap = Path(args.local_ckpt)
        # If dir, find .pt file inside (mimicking download_hf_checkpoint logic)
        if local_snap.is_dir():
            pt_files = list(local_snap.glob("*.pt"))
            if pt_files:
                cfg.ckpt_path = str(pt_files[0])
            else:
                cfg.ckpt_path = str(local_snap)
        else:
            cfg.ckpt_path = str(local_snap)
        snap_dir = local_snap if local_snap.is_dir() else local_snap.parent
        cfg.dataset_stats_path = str(snap_dir / "libero_dataset_statistics.json")
        cfg.t5_text_embeddings_path = str(snap_dir / "libero_t5_embeddings.pkl")

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
    obs = make_synthetic_observation()

    # Warmup
    print(f"\n[warmup] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        profile_e2e(get_action, model, cfg, dataset_stats, obs, task_desc, args.denoise_steps, parallel_gen)
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
        ms = profile_e2e(get_action, model, cfg, dataset_stats, obs, task_desc, args.denoise_steps, parallel_gen)
        all_ms.append(ms)
        if (i + 1) % 5 == 0:
            med = statistics.median(all_ms)
            print(f"  iter {i+1}/{args.iterations}: median so far = {med:.1f} ms")

    # Optional profiler trace
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
        ) as prof:
            get_action(
                cfg, model, dataset_stats, obs,
                task_desc,
                num_denoising_steps_action=args.denoise_steps,
                generate_future_state_and_value_in_parallel=parallel_gen,
            )
        trace_path = trace_dir / f"trace_steps_{args.denoise_steps}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"  Trace saved to {trace_path}")
        print("\n  Top 20 CUDA operations:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Results
    median_ms = statistics.median(all_ms)
    hz = 1000.0 / median_ms if median_ms > 0 else float("inf")
    chunk_s = cfg.chunk_size / 25.0

    output = {
        "experiment": "exp09a",
        "model": "Cosmos-Policy-LIBERO-Predict2-2B",
        "model_params_B": round(param_count, 2),
        "denoise_steps": args.denoise_steps,
        "chunk_size": cfg.chunk_size,
        "chunk_duration_s": chunk_s,
        "image_size": 224,
        "parallel_gen": parallel_gen,
        "gpu_name": torch.cuda.get_device_name(0),
        "device": f"cuda:{args.gpu}",
        "dtype": "bfloat16",
        "mode": "synthetic",
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
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }

    print(f"\n{'='*60}")
    print(f"COSMOS POLICY PROFILING — direct mode, {args.denoise_steps} steps")
    print(f"{'='*60}")
    print(f"  End-to-end median:  {median_ms:>8.1f} ms")
    print(f"  Policy query Hz:    {hz:>8.1f}")
    print(f"  Chunk = {cfg.chunk_size} steps @ 25Hz = {chunk_s:.2f}s")
    print(f"  Peak VRAM:          {output['peak_vram_mb']} MB")
    print(f"  Parameters:         {param_count:.2f}B")

    out_path = args.output or str(VLLA_ROOT / "exp" / "exp09a" / "results_profiling.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
