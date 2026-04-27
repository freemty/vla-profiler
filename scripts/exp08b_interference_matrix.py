"""exp08b — Full EPDA pair-wise interference matrix

Extends exp08a from 2 pairs (DA/PA) to all 6 pairs + multi-phase combos:
  Pairs:  EP, ED, EA, PD, PA, DA  (6 total)
  Triples: EPD, EPA, EDA, PDA     (4 total)
  Quad:   EPDA                     (1 total)

Each experiment:
  1. Run each phase isolated (baseline)
  2. Run N phases concurrently on separate CUDA streams
  3. Compute inflation = coloc_median / alone_median

Model combo (from spec §3):
  E = Qwen2.5-VL-7B vision encoder (ViT)
  P = Qwen2.5-VL-7B prefill (or 3B for memory)
  D = Qwen2.5-VL-7B decode (with KV cache)
  A = NitroGen DiT 174M (k-step denoising)

Usage:
  # Single pair
  python scripts/exp08b_interference_matrix.py \\
      --combo DA --gpu 0 --warmup 15 --iterations 40 \\
      --output exp/exp08b/results_DA.json

  # Full matrix (all 11 combos)
  python scripts/exp08b_interference_matrix.py \\
      --combo ALL --gpu 0 --warmup 15 --iterations 40 \\
      --output-dir exp/exp08b/

  # Pairs only (6 combos)
  python scripts/exp08b_interference_matrix.py \\
      --combo PAIRS --gpu 0 --warmup 15 --iterations 40 \\
      --output-dir exp/exp08b/
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profiling_stats import compute_phase_stats  # noqa: E402


ALL_PHASES = ["E", "P", "D", "A"]


def _log_vram(gpu: int, label: str = "") -> None:
    """Print current VRAM usage."""
    try:
        import torch
        alloc = torch.cuda.memory_allocated(gpu) / 1e9
        reserved = torch.cuda.memory_reserved(gpu) / 1e9
        print(f"  [VRAM] {label}: allocated={alloc:.1f}GB, reserved={reserved:.1f}GB")
    except Exception:
        pass
PAIR_COMBOS = ["EP", "ED", "EA", "PD", "PA", "DA"]
TRIPLE_COMBOS = ["EPD", "EPA", "EDA", "PDA"]
QUAD_COMBOS = ["EPDA"]
ALL_COMBOS = PAIR_COMBOS + TRIPLE_COMBOS + QUAD_COMBOS


# --------------------------------------------------------------------------- #
# Payload factories                                                           #
# --------------------------------------------------------------------------- #

def build_vision_encode_payload(
    gpu: int,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    image_size: int = 448,
) -> Callable[[], None]:
    """E phase: vision encoder forward pass on a dummy image.

    Extracts the ViT (model.visual) from Qwen2.5-VL and runs it on
    a random pixel_values tensor matching typical VLM input shape.
    """
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    vision_encoder = model.visual

    # Qwen2.5-VL ViT expects pixel_values [B, C, H, W] and grid_thw [B, 3].
    # For 448x448 with patch=14 and temporal_patch=2: T=1, H=32, W=32
    patch_size = 14
    temporal_patch = 2
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    seq_len = h_patches * w_patches  # 32*32 = 1024 for 448x448
    pixel_values = torch.randn(seq_len, 3 * temporal_patch, patch_size, patch_size,
                               device=device, dtype=dtype)
    grid_thw = torch.tensor([[1, h_patches, w_patches]], device=device, dtype=torch.long)

    # Free the LLM part to save VRAM — we only need the ViT
    del model.model
    del model.lm_head
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    def step_fn():
        with torch.no_grad():
            vision_encoder(pixel_values, grid_thw=grid_thw)

    return step_fn


def build_llm_decode_payload(
    gpu: int,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> Callable[[], None]:
    """D phase: single decode token with KV cache."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()

    text = "The capital of France is"
    inputs = proc(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    past_kv = prefill_out.past_key_values
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    state = {"past_kv": past_kv, "token": next_token}

    def step_fn():
        with torch.no_grad():
            out = model(
                input_ids=state["token"],
                past_key_values=state["past_kv"],
                use_cache=True,
            )
        state["past_kv"] = out.past_key_values
        state["token"] = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    return step_fn


def build_llm_prefill_payload(
    gpu: int,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> Callable[[], None]:
    """P phase: prefill on fixed-length prompt (~64 tokens)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()

    text = "Describe what you see in detail. " * 8  # ~64 tokens
    inputs = proc(text=[text], return_tensors="pt").to(device)

    def step_fn():
        with torch.no_grad():
            model(**inputs, use_cache=False)

    return step_fn


def build_nitrogen_action_payload(gpu: int, k: int = 10) -> Callable[[], None]:
    """A phase: NitroGen DiT k-step denoising loop."""
    import torch

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    try:
        _repo_root = Path(__file__).resolve().parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from src.controllers.nitrogen_controller import NitroGenController
        ctrl = NitroGenController(
            controller_config={"device": device, "dtype": "bfloat16", "k": k,
                               "model_name": "", "action_dim": 7, "proprio_dim": 7},
        )
        ctrl.init_pipeline()

        def step_fn():
            ctrl.infer_action()
        return step_fn
    except Exception as e:
        print(f"[warn] NitroGen controller unavailable ({e}); falling back to dummy DiT.")

    hidden, heads, layers = 1024, 16, 24
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            batch_first=True, dtype=dtype,
        ),
        num_layers=layers,
    ).to(device)
    model.eval()
    x = torch.randn(1, 64, hidden, device=device, dtype=dtype)

    def step_fn():
        with torch.no_grad():
            h = x
            for _ in range(k):
                h = model(h)

    return step_fn


def build_smoke_payload(phase: str) -> Callable[[], None]:
    """CPU sanity: time.sleep emulation."""
    sleep_ms = {"E": 250, "P": 38, "D": 18, "A": 165}[phase]

    def step_fn():
        time.sleep(sleep_ms / 1000.0)

    return step_fn


# --------------------------------------------------------------------------- #
# Thread runner                                                               #
# --------------------------------------------------------------------------- #

def run_loop(
    name: str,
    step_fn: Callable[[], None],
    n_iter: int,
    warmup: int,
    stream: Any,
    times_out: List[float],
    barrier: Optional[threading.Barrier] = None,
    use_cuda: bool = True,
) -> None:
    """Loop step_fn, record per-iter time via CUDA events."""
    if use_cuda:
        import torch
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                step_fn()
            torch.cuda.synchronize(stream.device)

            if barrier is not None:
                barrier.wait()

            for _ in range(n_iter):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                step_fn()
                e1.record(stream)
                e1.synchronize()
                times_out.append(e0.elapsed_time(e1))
    else:
        for _ in range(warmup):
            step_fn()
        if barrier is not None:
            barrier.wait()
        for _ in range(n_iter):
            t0 = time.perf_counter()
            step_fn()
            times_out.append((time.perf_counter() - t0) * 1000.0)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_isolated(
    name: str, step_fn: Callable, n_iter: int, warmup: int, use_cuda: bool
) -> List[float]:
    """Run phase alone on its own stream."""
    stream = None
    if use_cuda:
        import torch
        stream = torch.cuda.Stream()
    times: List[float] = []
    run_loop(name, step_fn, n_iter, warmup, stream, times, None, use_cuda)
    return times


def run_concurrent(
    phase_fns: Dict[str, Callable],
    n_iter: int,
    warmup: int,
    use_cuda: bool,
) -> Dict[str, List[float]]:
    """Run N phases concurrently, each on its own CUDA stream + thread."""
    results: Dict[str, List[float]] = {name: [] for name in phase_fns}
    streams: Dict[str, Any] = {}

    if use_cuda:
        import torch
        for name in phase_fns:
            streams[name] = torch.cuda.Stream()

    n_phases = len(phase_fns)
    barrier = threading.Barrier(n_phases)

    threads = []
    for name, fn in phase_fns.items():
        t = threading.Thread(
            target=run_loop,
            args=(name, fn, n_iter, warmup, streams.get(name), results[name], barrier, use_cuda),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


def run_combo(
    combo: str,
    phase_fns: Dict[str, Callable],
    n_iter: int,
    warmup: int,
    use_cuda: bool,
) -> Dict:
    """Run one interference combo: isolated baselines + concurrent."""
    phases = list(combo)

    print(f"\n{'='*60}")
    print(f"COMBO: {combo} ({'+'.join(phases)})")
    print(f"{'='*60}")

    # Isolated baselines
    isolated = {}
    for i, ph in enumerate(phases):
        print(f"  [{i+1}/{len(phases)*2}] {ph} isolated...")
        isolated[ph] = compute_phase_stats(
            run_isolated(ph, phase_fns[ph], n_iter, warmup, use_cuda)
        )

    # Concurrent
    concurrent_fns = {ph: phase_fns[ph] for ph in phases}
    print(f"  [{len(phases)+1}/{len(phases)*2}] {'+'.join(phases)} concurrent...")
    concurrent_raw = run_concurrent(concurrent_fns, n_iter, warmup, use_cuda)
    colocated = {ph: compute_phase_stats(concurrent_raw[ph]) for ph in phases}

    # Inflation
    inflation = {}
    for ph in phases:
        a_med = isolated[ph]["median_ms"]
        c_med = colocated[ph]["median_ms"]
        inflation[ph] = round(c_med / a_med, 3) if a_med > 0 else float("inf")

    result = {
        "combo": combo,
        "phases": phases,
        "isolated": isolated,
        "colocated": colocated,
        "inflation": inflation,
    }

    # Report
    print(f"\n  {'Phase':>6s} | {'Isolated':>10s} | {'Colocated':>10s} | {'Inflation':>10s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for ph in phases:
        a = isolated[ph]["median_ms"]
        c = colocated[ph]["median_ms"]
        inf = inflation[ph]
        print(f"  {ph:>6s} | {a:>8.2f}ms | {c:>8.2f}ms | {inf:>8.3f}x")

    return result


# --------------------------------------------------------------------------- #
# VRAM budget helper                                                          #
# --------------------------------------------------------------------------- #

def estimate_vram_gb(combo: str) -> float:
    """Rough VRAM estimate for a combo to check 48GB fit."""
    per_phase = {"E": 4.0, "P": 7.0, "D": 16.0, "A": 1.5}
    return sum(per_phase.get(ph, 0) for ph in combo)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="exp08b — Full EPDA interference matrix")
    p.add_argument("--combo", type=str, required=True,
                   help="Combo to test: EP/ED/EA/PD/PA/DA/EPD/EPA/EDA/PDA/EPDA, "
                        "or ALL (all 11), PAIRS (6 pairs only), SMOKE")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--k", type=int, default=10, help="NitroGen denoise steps")
    p.add_argument("--output", type=str, default=None, help="Output JSON (single combo)")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (multi combo)")
    p.add_argument("--e-model", type=str, default=None, help="Vision encoder model")
    p.add_argument("--d-model", type=str, default=None, help="Decode model")
    p.add_argument("--p-model", type=str, default=None, help="Prefill model")
    p.add_argument("--e-image-size", type=int, default=448, help="Image size for E phase")
    args = p.parse_args()

    # Resolve combos
    if args.combo == "ALL":
        combos = ALL_COMBOS
    elif args.combo == "PAIRS":
        combos = PAIR_COMBOS
    elif args.combo == "SMOKE":
        combos = PAIR_COMBOS  # smoke test with CPU
    elif args.combo in ALL_COMBOS:
        combos = [args.combo]
    else:
        p.error(f"Unknown combo: {args.combo}. Choose from {ALL_COMBOS + ['ALL', 'PAIRS', 'SMOKE']}")

    use_cuda = args.combo != "SMOKE"
    if use_cuda:
        import torch
        torch.cuda.set_device(args.gpu)

    # Default model paths (xdlab23)
    _7B = "/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct"
    _3B = "/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-3B-Instruct"

    # Determine which phases we need across all combos
    needed_phases = set()
    for c in combos:
        needed_phases.update(c)

    print(f"\n{'#'*60}")
    print(f"exp08b — EPDA Interference Matrix")
    print(f"Combos: {combos}")
    print(f"Phases needed: {sorted(needed_phases)}")
    print(f"GPU: {args.gpu}, warmup: {args.warmup}, iterations: {args.iterations}")
    print(f"{'#'*60}")

    # VRAM check
    max_combo = max(combos, key=len)
    vram_est = estimate_vram_gb(max_combo)
    print(f"\nLargest combo '{max_combo}' estimated VRAM: ~{vram_est:.1f}GB (48GB available)")

    # Build payloads — each phase gets its own model instance for thread safety.
    # VRAM budget: E(ViT ~1.5GB) + D(7B ~15GB) + P(3B ~7GB) + A(DiT ~1.5GB) ≈ 25GB / 48GB
    phase_fns: Dict[str, Callable] = {}

    if use_cuda:
        import torch

        if "E" in needed_phases:
            print("\n[init] Building E (vision encode) payload...")
            phase_fns["E"] = build_vision_encode_payload(
                args.gpu, args.e_model or _7B, args.e_image_size
            )
            _log_vram(args.gpu, "after E init")

        if "D" in needed_phases:
            print("[init] Building D (LLM decode) payload...")
            phase_fns["D"] = build_llm_decode_payload(args.gpu, args.d_model or _7B)
            _log_vram(args.gpu, "after D init")

        if "P" in needed_phases:
            print("[init] Building P (LLM prefill) payload...")
            phase_fns["P"] = build_llm_prefill_payload(args.gpu, args.p_model or _3B)
            _log_vram(args.gpu, "after P init")

        if "A" in needed_phases:
            print("[init] Building A (NitroGen DiT) payload...")
            phase_fns["A"] = build_nitrogen_action_payload(args.gpu, k=args.k)
            _log_vram(args.gpu, "after A init")
    else:
        for ph in needed_phases:
            phase_fns[ph] = build_smoke_payload(ph)

    # Run combos
    all_results = {}
    for i, combo in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] Running combo {combo}...")
        result = run_combo(combo, phase_fns, args.iterations, args.warmup, use_cuda)
        result["gpu"] = args.gpu
        result["warmup"] = args.warmup
        result["iterations"] = args.iterations
        result["action_k"] = args.k
        result["use_cuda"] = use_cuda
        all_results[combo] = result

        # Save individual result
        if args.output_dir:
            out_path = Path(args.output_dir) / f"results_{combo}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  [saved] {out_path}")

    # Save single combo result
    if args.output and len(combos) == 1:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results[combos[0]], f, indent=2)
        print(f"\n[saved] {out_path}")

    # Save summary matrix
    if len(combos) > 1 and args.output_dir:
        summary = build_summary_matrix(all_results)
        summary_path = Path(args.output_dir) / "interference_matrix.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] Summary matrix: {summary_path}")
        print_summary_table(summary)


def build_summary_matrix(all_results: Dict[str, Dict]) -> Dict:
    """Build a condensed interference matrix from all combo results."""
    matrix = {
        "experiment": "exp08b",
        "description": "EPDA pair-wise interference matrix",
        "combos": {},
    }

    for combo, result in all_results.items():
        entry = {
            "phases": result["phases"],
            "inflation": result["inflation"],
            "isolated_median": {ph: result["isolated"][ph]["median_ms"] for ph in result["phases"]},
            "colocated_median": {ph: result["colocated"][ph]["median_ms"] for ph in result["phases"]},
        }
        matrix["combos"][combo] = entry

    return matrix


def print_summary_table(summary: Dict) -> None:
    """Print a readable summary table."""
    print(f"\n{'='*70}")
    print("INTERFERENCE MATRIX SUMMARY")
    print(f"{'='*70}")
    print(f"{'Combo':>6s} | {'Phase':>5s} | {'Alone(ms)':>10s} | {'Coloc(ms)':>10s} | {'Inflation':>10s}")
    print(f"{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for combo, entry in summary["combos"].items():
        for ph in entry["phases"]:
            alone = entry["isolated_median"][ph]
            coloc = entry["colocated_median"][ph]
            inf = entry["inflation"][ph]
            print(f"{combo:>6s} | {ph:>5s} | {alone:>8.2f}ms | {coloc:>8.2f}ms | {inf:>8.3f}x")
        if len(entry["phases"]) > 1:
            print(f"{'-'*6}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")


if __name__ == "__main__":
    main()
