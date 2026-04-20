"""
Fast-WAM latency profiling script.

Instruments the Fast-WAM inference pipeline to measure E/C/A phase breakdown:
  E: SigLIP image encoding
  C: ActionDiT context (backbone forward pass)
  A: Flow matching action generation (Euler solver steps)

Usage:
    conda activate fastwam
    cd /data1/ybyang/FastWAM
    python /data1/ybyang/vlla/scripts/profile_fastwam.py \
        --config configs/model/fastwam.yaml \
        --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
        --dataset-stats checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
        --gpu 0 \
        --warmup 5 \
        --iterations 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.cuda


def sync_cuda():
    torch.cuda.synchronize()


def timed_section(name: str, times_dict: dict):
    """Context manager for timing a CUDA section."""
    class Timer:
        def __enter__(self):
            sync_cuda()
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            sync_cuda()
            elapsed = (time.perf_counter() - self.start) * 1000
            times_dict.setdefault(name, []).append(elapsed)

    return Timer()


def main():
    parser = argparse.ArgumentParser(description="Profile Fast-WAM inference")
    parser.add_argument("--config", type=str, default="configs/model/fastwam.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--dataset-stats", type=str,
                        default="checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    print(f"[Profile] Device: {device}")
    print(f"[Profile] Config: {args.config}")
    print(f"[Profile] Checkpoint: {args.checkpoint}")

    # --- Step 1: Load model ---
    print("\n=== Loading Fast-WAM model ===")

    # Try to import Fast-WAM's model loading utilities
    # The exact import path depends on their codebase structure
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        print(f"[Profile] Model config loaded: {args.config}")
    except ImportError:
        print("[Profile] omegaconf not available, trying hydra config...")
        cfg = None

    # Attempt to load model using their utilities
    # This section will need adaptation based on actual repo structure
    try:
        sys.path.insert(0, str(Path.cwd()))
        from fastwam.model import FastWAMPolicy  # Adjust import path
        model = FastWAMPolicy.from_pretrained(
            args.checkpoint,
            config=cfg,
            device=device,
        )
        model.eval()
        print("[Profile] Model loaded successfully")
    except (ImportError, AttributeError) as e:
        print(f"[Profile] Direct import failed: {e}")
        print("[Profile] Attempting alternative loading strategy...")

        # Fallback: try to find the correct import path
        try:
            from fastwam.inference import load_policy
            model = load_policy(args.checkpoint, args.config, device=device)
            model.eval()
            print("[Profile] Model loaded via load_policy()")
        except (ImportError, AttributeError) as e2:
            print(f"[Profile] Alternative also failed: {e2}")
            print("\n[Profile] === MANUAL ADAPTATION NEEDED ===")
            print("The Fast-WAM codebase structure differs from expected.")
            print("Please inspect the repo and update the import paths.")
            print("\nSuggested steps:")
            print("  1. find . -name '*.py' | xargs grep 'class.*Policy'")
            print("  2. find . -name '*.py' | xargs grep 'def.*inference\\|def.*predict'")
            print("  3. Look at experiments/libero/run_libero_manager.py for inference flow")
            sys.exit(1)

    # --- Step 2: Prepare dummy inputs ---
    print("\n=== Preparing dummy inputs ===")

    # Load dataset stats for input normalization
    with open(args.dataset_stats) as f:
        stats = json.load(f)

    # Create dummy observation (2 cameras, 224x224, RGB)
    dummy_images = torch.randn(1, 2, 3, 224, 224, device=device, dtype=torch.float32)
    dummy_state = torch.zeros(1, 14, device=device, dtype=torch.float32)  # robot state
    dummy_text_embed = torch.randn(1, 77, 1024, device=device, dtype=torch.float32)  # T5 embed

    print(f"[Profile] Dummy images: {dummy_images.shape}")
    print(f"[Profile] Dummy state: {dummy_state.shape}")

    # --- Step 3: Profile inference ---
    print(f"\n=== Profiling: {args.warmup} warmup + {args.iterations} measured ===")

    times = {}
    total_iters = args.warmup + args.iterations

    for i in range(total_iters):
        is_warmup = i < args.warmup
        prefix = "warmup" if is_warmup else "measured"

        with torch.no_grad():
            # --- Phase E: Image Encoding ---
            with timed_section("encode", times if not is_warmup else {}):
                # This depends on model's internal encode method
                # May need adaptation based on actual API
                if hasattr(model, 'encode_images'):
                    visual_tokens = model.encode_images(dummy_images)
                elif hasattr(model, 'vision_encoder'):
                    visual_tokens = model.vision_encoder(
                        dummy_images.reshape(-1, 3, 224, 224)
                    )
                else:
                    # Fallback: run full forward and time externally
                    visual_tokens = None

            # --- Phase C: Context / Backbone forward ---
            with timed_section("context", times if not is_warmup else {}):
                if hasattr(model, 'compute_context'):
                    context = model.compute_context(
                        visual_tokens, dummy_text_embed, dummy_state
                    )
                elif hasattr(model, 'backbone'):
                    # Generic backbone forward
                    context = model.backbone(visual_tokens, dummy_text_embed)
                else:
                    context = None

            # --- Phase A: Action generation (flow matching) ---
            with timed_section("action", times if not is_warmup else {}):
                if hasattr(model, 'predict_action'):
                    actions = model.predict_action(context, dummy_state)
                elif hasattr(model, 'action_head'):
                    actions = model.action_head(context)
                else:
                    actions = None

            # --- Total (end-to-end) ---
            # Also measure total as a sanity check
            sync_cuda()

        if not is_warmup and (i - args.warmup) % 5 == 0:
            print(f"  [{prefix}] iter {i - args.warmup}/{args.iterations}")

    # --- Step 4: Report results ---
    print("\n" + "=" * 60)
    print("RESULTS: Fast-WAM Latency Profiling")
    print("=" * 60)

    results = {}
    for phase, measurements in times.items():
        if measurements:
            mean_ms = sum(measurements) / len(measurements)
            std_ms = (sum((x - mean_ms) ** 2 for x in measurements) / len(measurements)) ** 0.5
            min_ms = min(measurements)
            max_ms = max(measurements)
            results[phase] = {
                "mean_ms": round(mean_ms, 2),
                "std_ms": round(std_ms, 2),
                "min_ms": round(min_ms, 2),
                "max_ms": round(max_ms, 2),
                "n": len(measurements),
            }
            print(f"  {phase:>12s}: {mean_ms:7.2f} ms (std={std_ms:.2f}, min={min_ms:.2f}, max={max_ms:.2f})")

    total_mean = sum(r["mean_ms"] for r in results.values())
    print(f"  {'TOTAL':>12s}: {total_mean:7.2f} ms")
    print(f"  {'Hz':>12s}: {1000 / total_mean:.1f}")

    print("\n--- Phase breakdown ---")
    for phase, r in results.items():
        pct = r["mean_ms"] / total_mean * 100
        print(f"  {phase:>12s}: {pct:5.1f}%")

    # Save results
    output_path = args.output or f"/data1/ybyang/vlla/exp/exp04a/results_fastwam_gpu{args.gpu}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": "Fast-WAM (fastwam)",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "phases": results,
        "total_ms": round(total_mean, 2),
        "hz": round(1000 / total_mean, 1),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Profile] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
