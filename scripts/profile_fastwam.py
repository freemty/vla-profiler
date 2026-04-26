"""
Fast-WAM latency profiling script.

Instruments the Fast-WAM inference pipeline to measure E/C/A phase breakdown:
  E: VAE image encoding (first-frame latent extraction)
  C: Text encode + Video expert prefill + MoT KV cache (context formation)
  A: Flow matching action denoising (Euler solver, 20 steps)

Two modes:
  --mode full    : Load real checkpoint (requires Wan2.2 base model downloaded)
  --mode random  : Random init (same architecture, valid for timing only)

Usage:
    conda activate fastwam
    cd /data1/ybyang/FastWAM
    python /data1/ybyang/vlla/scripts/profile_fastwam.py \
        --mode random \
        --gpu 0 --warmup 5 --iterations 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.cuda

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profiling_stats import compute_phase_stats, print_phase_summary  # noqa: E402


def sync_cuda():
    torch.cuda.synchronize()


def timed_section(name: str, times_dict: dict):
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


def build_model_random(device: str, dtype: torch.dtype, action_dim: int = 7, proprio_dim: int = 7):
    """Build FastWAM with random weights. Same architecture, valid for latency measurement."""
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
    from fastwam.models.wan22.mot import MoT

    print("[Profile] Building model with random weights (timing-only mode)...")

    # Video expert (WanVideoDiT) — same config as Wan2.2-TI2V-5B
    video_expert = WanVideoDiT(
        has_image_input=False,
        patch_size=[1, 2, 2],
        in_dim=48,
        hidden_dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        eps=1e-6,
        seperated_timestep=True,
        require_clip_embedding=False,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=False,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
        action_dim=action_dim,
        action_group_causal_mask_mode="group_diagonal",
    ).to(device=device, dtype=dtype)

    # Action expert (ActionDiT)
    action_expert = ActionDiT(
        action_dim=action_dim,
        hidden_dim=1024,
        ffn_dim=4096,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)

    # MoT (Mixture of Transformers — routes tokens between video/action experts)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    ).to(device=device, dtype=dtype)

    # VAE (for image encoding)
    vae = WanVideoVAE38().to(device=device, dtype=dtype)

    model = FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_dim=4096,
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=dtype,
    )
    model.eval()
    return model


def build_model_full(checkpoint: str, device: str, dtype: torch.dtype):
    """Load full model from pretrained checkpoint."""
    from fastwam.models.wan22.fastwam import FastWAM

    print("[Profile] Loading full model from pretrained...")
    model = FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len=128,
        load_text_encoder=True,
        proprio_dim=7,
        redirect_common_files=True,
        video_dit_config={
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": False,
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": 7,
            "action_group_causal_mask_mode": "group_diagonal",
        },
        action_dit_config={
            "action_dim": 7,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "text_dim": 4096,
            "freq_dim": 256,
            "eps": 1e-6,
            "use_gradient_checkpointing": False,
        },
        action_dit_pretrained_path="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
        skip_dit_load_from_pretrain=False,
        mot_checkpoint_mixed_attn=False,
        video_train_shift=5.0,
        video_infer_shift=5.0,
        video_num_train_timesteps=1000,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=1000,
    )

    print("[Profile] Loading fine-tuned checkpoint...")
    model.load_checkpoint(checkpoint)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Profile Fast-WAM inference")
    parser.add_argument("--mode", choices=["full", "random"], default="random",
                        help="full=real weights, random=random init (valid for timing)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=15,
                        help="Default 15 to absorb GPU power-state warmup bimodality (see exp07a audit 2026-04-26)")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16

    print(f"[Profile] Device: {device}")
    print(f"[Profile] GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"[Profile] Mode: {args.mode}")

    sys.path.insert(0, str(Path.cwd()))

    if args.mode == "random":
        model = build_model_random(device=device, dtype=dtype)
    else:
        model = build_model_full(checkpoint=args.checkpoint, device=device, dtype=dtype)

    print(f"[Profile] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # --- Prepare dummy inputs (LIBERO 2-cam 224x224 horizontal concat) ---
    print("\n=== Preparing dummy inputs ===")
    height, width = 224, 448
    dummy_image = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    dummy_proprio = torch.randn(1, 7, device=device, dtype=dtype)

    # Pre-computed text context (simulates T5 output).
    # Text encoding is a one-time cost outside the control loop, so we
    # exclude it from per-step profiling and use synthetic embeddings.
    context_len = 20
    dummy_context = torch.randn(1, context_len, 4096, device=device, dtype=dtype)
    dummy_context_mask = torch.ones(1, context_len, device=device, dtype=torch.bool)

    print(f"[Profile] Input image: {dummy_image.shape} (2-cam horizontal concat)")
    print(f"[Profile] Action horizon: {args.action_horizon}")
    print(f"[Profile] Flow matching steps: {args.num_inference_steps}")

    # --- Profile phases ---
    print(f"\n=== Profiling: {args.warmup} warmup + {args.iterations} measured ===")

    times = {}
    total_times = []
    total_iters = args.warmup + args.iterations

    for i in range(total_iters):
        is_warmup = i < args.warmup
        record = times if not is_warmup else {}

        with torch.no_grad():
            sync_cuda()
            t_start = time.perf_counter()

            # Phase E: VAE encode (image -> first-frame latents)
            with timed_section("encode", record):
                first_frame_latents = model._encode_input_image_latents_tensor(
                    input_image=dummy_image, tiled=False
                )
                fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

            # Phase C: Context formation (text/proprio encode + video prefill + KV cache)
            with timed_section("context", record):
                # Append proprio to context
                context, context_mask = model._append_proprio_to_context(
                    context=dummy_context,
                    context_mask=dummy_context_mask,
                    proprio=dummy_proprio,
                )

                timestep_video = torch.zeros(
                    (first_frame_latents.shape[0],),
                    dtype=first_frame_latents.dtype,
                    device=device,
                )

                latents_action = torch.randn(
                    (1, args.action_horizon, model.action_expert.action_dim),
                    device=device, dtype=dtype,
                )

                video_pre = model.video_expert.pre_dit(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                video_seq_len = int(video_pre["tokens"].shape[1])

                attention_mask = model._build_mot_attention_mask(
                    video_seq_len=video_seq_len,
                    action_seq_len=latents_action.shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                )

                video_kv_cache = model.mot.prefill_video_cache(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                )

            # Phase A: Flow matching action denoising (Euler solver, N steps)
            with timed_section("action", record):
                infer_timesteps, infer_deltas = model.infer_action_scheduler.build_inference_schedule(
                    num_inference_steps=args.num_inference_steps,
                    device=device,
                    dtype=latents_action.dtype,
                )

                for step_t, step_delta in zip(infer_timesteps, infer_deltas):
                    timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
                    pred_action = model._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                    )
                    latents_action = model.infer_action_scheduler.step(
                        pred_action, step_delta, latents_action
                    )

            sync_cuda()
            t_end = time.perf_counter()
            if not is_warmup:
                total_times.append((t_end - t_start) * 1000)

        if not is_warmup and (i - args.warmup) % 5 == 0:
            print(f"  [measured] iter {i - args.warmup}/{args.iterations}")

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS: Fast-WAM Latency Profiling")
    print(f"  Mode: {args.mode} | Steps: {args.num_inference_steps} | Horizon: {args.action_horizon}")
    print("=" * 60)

    results = {}
    for phase, measurements in times.items():
        if measurements:
            stats = compute_phase_stats(measurements)
            results[phase] = stats
            print_phase_summary(phase, stats, label_width=12)

    total_stats = compute_phase_stats(total_times)
    total_mean = total_stats["mean_ms"]
    total_median = total_stats["median_ms"]
    total_std = total_stats["std_ms"]
    print(f"  {'TOTAL(e2e)':>12s}: mean={total_mean:7.2f}ms median={total_median:.2f}ms "
          f"p10/p90={total_stats['p10_ms']:.2f}/{total_stats['p90_ms']:.2f} cv={total_stats['cv_pct']:.1f}%")
    print(f"  {'Hz':>12s}: {1000 / total_median:.1f} (from median)")

    phase_total = sum(r["mean_ms"] for r in results.values())
    print(f"\n--- Phase breakdown (sum={phase_total:.1f}ms) ---")
    for phase, r in results.items():
        pct = r["mean_ms"] / phase_total * 100
        print(f"  {phase:>12s}: {pct:5.1f}%  ({r['mean_ms']:.1f}ms)")

    # Save
    output_path = args.output or f"/data1/ybyang/vlla/exp/exp04a/results_fastwam_gpu{args.gpu}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": "Fast-WAM",
        "paradigm": "skip-imagination WAM",
        "mode": args.mode,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": args.action_horizon,
        "input_shape": f"1x3x{height}x{width}",
        "phases": results,
        "total_e2e": total_stats,
        "total_e2e_ms": round(total_mean, 2),  # legacy alias
        "total_e2e_std_ms": round(total_std, 2),  # legacy alias
        "hz": round(1000 / total_median, 2),  # from median (more robust)
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Profile] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
