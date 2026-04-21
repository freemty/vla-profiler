"""
LingBot-VA latency profiling script.

Instruments the LingBot-VA (full WAM) inference pipeline:
  E: VAE encode (observation images -> latents)
  V: Video denoise (DiT forward × N_video steps) — "imagination"
  A: Action denoise (DiT forward × N_action steps) — action generation

This is a FULL WAM: video is generated first, then action is predicted.
Compare with Fast-WAM (skip-imagination) which skips the V phase.

Usage:
    conda activate lingbot-va
    cd /data1/ybyang/lingbot-va
    python /data1/ybyang/vlla/scripts/profile_lingbot_va.py \
        --mode random --gpu 0 --warmup 5 --iterations 20

Architecture:
    - Transformer: WanTransformer3DModel (5B, from Wan2.2-TI2V-5B)
    - VAE: AutoencoderKLWan (Wan2.2 VAE, ~120M)
    - Text encoder: UMT5EncoderModel (~4.7B)
    - Video + Action share the SAME transformer (action_mode flag switches behavior)
    - Video: 20 denoise steps (default), CFG scale 5
    - Action: 50 denoise steps (default), CFG scale 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.cuda
import torch.nn.functional as F
from einops import rearrange


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


def main():
    parser = argparse.ArgumentParser(description="Profile LingBot-VA inference")
    parser.add_argument("--mode", choices=["full", "random"], default="random",
                        help="full=real weights (needs Wan2.2 downloaded), random=random init (valid for timing)")
    parser.add_argument("--model-path", type=str,
                        default="/data1/ybyang/huggingface/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--frame-chunk-size", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=30)
    parser.add_argument("--action-per-frame", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16

    print(f"[Profile] Device: {device}")
    print(f"[Profile] GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"[Profile] Mode: {args.mode}")

    sys.path.insert(0, str(Path.cwd()))

    from wan_va.modules.model import WanTransformer3DModel
    from wan_va.modules.utils import WanVAEStreamingWrapper
    from wan_va.utils import FlowMatchScheduler, get_mesh_id

    if args.mode == "random":
        print("[Profile] Building model with random weights (timing-only mode)...")

        # Transformer — LingBot-VA's custom WanTransformer3DModel
        # (NOT diffusers' version; 24 heads, 30 layers, includes action_dim)
        transformer_config = {
            "patch_size": [1, 2, 2],
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 48,
            "out_channels": 48,
            "action_dim": args.action_dim,
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

        # VAE — use diffusers AutoencoderKLWan with random weights
        # z_dim=48 → matches in_channels=48 of transformer (48 latent channels)
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan(
            z_dim=48,
            in_channels=3,
            out_channels=3,
        ).to(device=device, dtype=dtype)
        vae.eval()

    else:
        print("[Profile] Loading full model from pretrained...")
        from wan_va.modules.utils import load_transformer, load_vae
        transformer = load_transformer(
            f"{args.model_path}/transformer",
            torch_dtype=dtype,
            torch_device=device,
            attn_mode="torch",
        )
        transformer.eval()
        vae = load_vae(f"{args.model_path}/vae", torch_dtype=dtype, torch_device=device)
        vae.eval()

    param_count = sum(p.numel() for p in transformer.parameters()) / 1e6
    vae_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"[Profile] Transformer: {param_count:.0f}M params")
    print(f"[Profile] VAE: {vae_params:.0f}M params")

    # --- Prepare inputs ---
    print("\n=== Preparing dummy inputs ===")

    frame_chunk_size = args.frame_chunk_size
    patch_size = (1, 2, 2)
    latent_height = args.height // 16
    latent_width = args.width // 16 * args.num_cameras  # multi-cam concat

    # Dummy observation image (2 cameras, 128x128, 4 frames)
    dummy_video = torch.randn(
        args.num_cameras, 3, frame_chunk_size, args.height, args.width,
        device=device, dtype=dtype
    )

    # Pre-computed text embeddings (T5 output, 512 tokens × 4096 dim)
    prompt_embeds = torch.randn(1, 512, 4096, device=device, dtype=dtype)

    # Scheduler
    video_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)

    print(f"[Profile] Resolution: {args.height}x{args.width} × {args.num_cameras} cameras")
    print(f"[Profile] Latent: {latent_height}x{latent_width} × {frame_chunk_size} frames")
    print(f"[Profile] Video denoise steps: {args.video_steps}")
    print(f"[Profile] Action denoise steps: {args.action_steps}")

    # --- Profile ---
    print(f"\n=== Profiling: {args.warmup} warmup + {args.iterations} measured ===")

    times = {}
    total_times = []
    streaming_vae = WanVAEStreamingWrapper(vae)
    total_iters = args.warmup + args.iterations

    for i in range(total_iters):
        is_warmup = i < args.warmup
        record = times if not is_warmup else {}
        streaming_vae.clear_cache()

        with torch.no_grad():
            sync_cuda()
            t_start = time.perf_counter()

            # Phase E: VAE encode (observation → latents)
            with timed_section("encode", record):
                video_input = dummy_video / 255.0 * 2.0 - 1.0
                enc_out = streaming_vae.encode_chunk(video_input)
                # In random-init mode, VAE output shape may not match transformer's
                # expected latent size (VAE spatial factor=8, but we need H//16).
                # Use random init_latent with correct shape for subsequent phases.
                init_latent = torch.randn(
                    1, 48, frame_chunk_size, latent_height, latent_width,
                    device=device, dtype=dtype
                )

            # Phase V: Video denoise loop (imagination)
            with timed_section("video_denoise", record):
                latents = torch.randn(
                    1, 48, frame_chunk_size, latent_height, latent_width,
                    device=device, dtype=dtype
                )
                latents[:, :, 0:1] = init_latent[:, :, 0:1]

                video_scheduler.set_timesteps(args.video_steps)
                timesteps = video_scheduler.timesteps
                timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

                for step_i, t in enumerate(timesteps[:-1]):
                    grid_id = get_mesh_id(
                        latents.shape[-3] // patch_size[0],
                        latents.shape[-2] // patch_size[1],
                        latents.shape[-1] // patch_size[2],
                        0, 1, 0
                    ).to(device)

                    timestep_vec = torch.ones([1, latents.shape[2]], dtype=torch.float32, device=device) * t

                    input_dict = {
                        'noisy_latents': latents,
                        'timesteps': timestep_vec,
                        'grid_id': grid_id[None],
                        'text_emb': prompt_embeds.clone(),
                    }

                    noise_pred = transformer(input_dict, update_cache=0, cache_name='pos', action_mode=False)

                    # Reshape prediction back to spatial
                    noise_pred_spatial = noise_pred.view(
                        1, frame_chunk_size,
                        latent_height // patch_size[1],
                        latent_width // patch_size[2],
                        48 * patch_size[0] * patch_size[1] * patch_size[2]
                    )
                    noise_pred_spatial = rearrange(
                        noise_pred_spatial, 'b f h w (c p1 p2) -> b c (f) (h p1) (w p2)',
                        p1=patch_size[1], p2=patch_size[2], c=48
                    )

                    latents = video_scheduler.step(noise_pred_spatial, t, latents, return_dict=False)
                    latents[:, :, 0:1] = init_latent[:, :, 0:1]

            # Phase A: Action denoise loop
            with timed_section("action_denoise", record):
                actions = torch.randn(
                    1, args.action_dim, frame_chunk_size, args.action_per_frame, 1,
                    device=device, dtype=dtype
                )

                action_scheduler.set_timesteps(args.action_steps)
                action_timesteps = action_scheduler.timesteps
                action_timesteps = F.pad(action_timesteps, (0, 1), mode='constant', value=0)

                for step_i, t in enumerate(action_timesteps[:-1]):
                    grid_id = get_mesh_id(
                        actions.shape[-3],
                        actions.shape[-2],
                        actions.shape[-1],
                        1, 1, 0, action=True
                    ).to(device)

                    action_timestep_vec = torch.ones([1, actions.shape[2]], dtype=torch.float32, device=device) * t

                    input_dict = {
                        'noisy_latents': actions,
                        'timesteps': action_timestep_vec,
                        'grid_id': grid_id[None],
                        'text_emb': prompt_embeds.clone(),
                    }

                    action_pred = transformer(input_dict, update_cache=0, cache_name='pos', action_mode=True)

                    action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size)
                    actions = action_scheduler.step(action_pred, t, actions, return_dict=False)

            sync_cuda()
            t_end = time.perf_counter()
            if not is_warmup:
                total_times.append((t_end - t_start) * 1000)

        if not is_warmup and (i - args.warmup) % 5 == 0:
            print(f"  [measured] iter {i - args.warmup}/{args.iterations}")

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS: LingBot-VA Latency Profiling (Full WAM)")
    print(f"  Mode: {args.mode} | V-steps: {args.video_steps} | A-steps: {args.action_steps}")
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
            print(f"  {phase:>16s}: {mean_ms:7.2f} ms (std={std_ms:.2f}, min={min_ms:.2f}, max={max_ms:.2f})")

    total_mean = sum(total_times) / len(total_times)
    total_std = (sum((x - total_mean) ** 2 for x in total_times) / len(total_times)) ** 0.5
    print(f"  {'TOTAL(e2e)':>16s}: {total_mean:7.2f} ms (std={total_std:.2f})")
    print(f"  {'Hz':>16s}: {1000 / total_mean:.1f}")

    phase_total = sum(r["mean_ms"] for r in results.values())
    print(f"\n--- Phase breakdown (sum={phase_total:.1f}ms) ---")
    for phase, r in results.items():
        pct = r["mean_ms"] / phase_total * 100
        print(f"  {phase:>16s}: {pct:5.1f}%  ({r['mean_ms']:.1f}ms)")

    # Save
    output_path = args.output or f"/data1/ybyang/vlla/exp/exp04b/results_lingbot_va_gpu{args.gpu}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": "LingBot-VA",
        "paradigm": "full WAM (video imagination + action)",
        "mode": args.mode,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
        "resolution": f"{args.height}x{args.width}x{args.num_cameras}cam",
        "frame_chunk_size": args.frame_chunk_size,
        "action_dim": args.action_dim,
        "transformer_params_M": round(param_count, 0),
        "phases": results,
        "total_e2e_ms": round(total_mean, 2),
        "total_e2e_std_ms": round(total_std, 2),
        "hz": round(1000 / total_mean, 1),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Profile] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
