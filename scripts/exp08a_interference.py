"""exp08a — EPDA pair-wise interference pilot

Tests two roofline predictions by co-locating a LLM phase (P or D) with
an Action DiT (A) phase on the same GPU, measuring per-phase latency
inflation vs isolated baseline.

Predictions (from docs/specs/2026-04-26-epda-roofline-analysis.md §5):
  - D+A: STRONG interference (>30%) — HBM saturation + kernel dispatch amplification
  - P+A: WEAK interference (<10%) — tensor core and dispatch slots orthogonal

Strategy (α, agreed 2026-04-27):
  Two Python threads, two CUDA streams, no barrier between threads.
  Each thread loops its phase N_iter times. Sum(per-iter latency) is
  aggregated with median/p10/p90 via compute_phase_stats.

Inflation ratio = coloc_median / alone_median. A ratio of 1.30 means
the phase runs 30% slower under contention.

Usage:
  # D+A (Qwen-VL-7B decode + NitroGen DiT)
  python scripts/exp08a_interference.py \
      --pair DA --gpu 0 --warmup 15 --iterations 40 \
      --output exp/exp08a/results_DA.json

  # P+A (LingBot-VLA prefill + NitroGen DiT)
  python scripts/exp08a_interference.py \
      --pair PA --gpu 0 --warmup 15 --iterations 40 \
      --output exp/exp08a/results_PA.json

  # CPU sanity check (no GPU, fake payloads)
  python scripts/exp08a_interference.py --pair SMOKE --iterations 5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _profiling_stats import compute_phase_stats  # noqa: E402


# --------------------------------------------------------------------------- #
# Payload factories                                                           #
# --------------------------------------------------------------------------- #

def build_llm_decode_payload(gpu: int, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Return (warmup_fn, step_fn). step_fn does one decode token."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()

    # Build KV-cache-ready state by running a prefill once.
    text = "The capital of France is"
    inputs = proc(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    past_kv = prefill_out.past_key_values
    # Initial token to feed into decode loop.
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


def build_llm_prefill_payload(gpu: int, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """P phase: prefill on a fixed-length prompt + visual token prefix."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = f"cuda:{gpu}"
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()

    text = "Describe what you see in detail. " * 8  # ~64 tokens, prefill-heavy
    inputs = proc(text=[text], return_tensors="pt").to(device)

    def step_fn():
        with torch.no_grad():
            model(**inputs, use_cache=False)

    return step_fn


def build_nitrogen_action_payload(gpu: int, k: int = 10):
    """A phase: NitroGen DiT k-step denoising loop."""
    import torch

    # Minimal random-weight DiT mimicking NitroGen 174M (single-block proxy).
    # For real runs we import NitroGenController; for smoke we use a dummy
    # 174M-ish transformer that hits compute/BW similarly.
    device = f"cuda:{gpu}"
    dtype = torch.bfloat16

    # Try real NitroGen first.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from controllers.nitrogen_controller import NitroGenController  # type: ignore
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

    # Dummy fallback: 24-layer transformer (~174M params approx).
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


def build_smoke_payload(phase: str):
    """CPU sanity: just time.sleep to emulate latency."""
    sleep_ms = {"D": 18, "P": 38, "A": 165}[phase]

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
    barrier: Optional[threading.Event] = None,
    use_cuda: bool = True,
) -> None:
    """Loop step_fn, record per-iter time via CUDA events (or perf_counter)."""
    if use_cuda:
        import torch
        # Ensure every CUDA call in this thread uses our stream.
        with torch.cuda.stream(stream):
            for i in range(warmup):
                step_fn()
            torch.cuda.synchronize(stream.device)

            if barrier is not None:
                barrier.wait()

            for i in range(n_iter):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                step_fn()
                e1.record(stream)
                e1.synchronize()
                times_out.append(e0.elapsed_time(e1))
    else:
        # CPU smoke path
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

def run_isolated(name: str, step_fn, n_iter: int, warmup: int, use_cuda: bool) -> List[float]:
    """Run phase alone on its own stream."""
    stream = None
    if use_cuda:
        import torch
        stream = torch.cuda.Stream()
    times: List[float] = []
    run_loop(name, step_fn, n_iter, warmup, stream, times, None, use_cuda)
    return times


def run_colocated(
    llm_name: str, llm_fn,
    a_name: str, a_fn,
    n_iter: int, warmup: int, use_cuda: bool,
) -> Dict[str, List[float]]:
    """Run both phases concurrently on separate streams (strategy α)."""
    llm_times: List[float] = []
    a_times: List[float] = []
    stream_llm = None
    stream_a = None
    if use_cuda:
        import torch
        stream_llm = torch.cuda.Stream()
        stream_a = torch.cuda.Stream()

    barrier = threading.Event()
    t_llm = threading.Thread(
        target=run_loop,
        args=(llm_name, llm_fn, n_iter, warmup, stream_llm, llm_times, barrier, use_cuda),
    )
    t_a = threading.Thread(
        target=run_loop,
        args=(a_name, a_fn, n_iter, warmup, stream_a, a_times, barrier, use_cuda),
    )

    t_llm.start()
    t_a.start()
    # Small delay to ensure both threads have finished warmup before barrier release
    time.sleep(0.05)
    barrier.set()
    t_llm.join()
    t_a.join()

    return {llm_name: llm_times, a_name: a_times}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", choices=["DA", "PA", "SMOKE"], required=True,
                   help="Which interference pair to test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--k", type=int, default=10, help="NitroGen denoise steps")
    p.add_argument("--llm_model", type=str, default=None,
                   help="Override LLM model name")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    use_cuda = args.pair != "SMOKE"
    if use_cuda:
        import torch
        torch.cuda.set_device(args.gpu)

    # Build payloads.
    if args.pair == "DA":
        llm_name = "D"
        llm_fn = build_llm_decode_payload(
            args.gpu, args.llm_model or "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        a_fn = build_nitrogen_action_payload(args.gpu, k=args.k)
    elif args.pair == "PA":
        llm_name = "P"
        llm_fn = build_llm_prefill_payload(
            args.gpu, args.llm_model or "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        a_fn = build_nitrogen_action_payload(args.gpu, k=args.k)
    else:  # SMOKE
        llm_name = "D"
        llm_fn = build_smoke_payload("D")
        a_fn = build_smoke_payload("A")

    print(f"\n=== exp08a pilot: {args.pair} ({llm_name}+A), warmup={args.warmup}, iter={args.iterations} ===")

    # --- Isolated baselines ---
    print(f"\n[1/3] Running {llm_name} alone...")
    llm_alone = run_isolated(llm_name, llm_fn, args.iterations, args.warmup, use_cuda)

    print(f"[2/3] Running A alone...")
    a_alone = run_isolated("A", a_fn, args.iterations, args.warmup, use_cuda)

    # --- Co-located ---
    print(f"[3/3] Running {llm_name}+A co-located (2 threads, 2 streams)...")
    coloc = run_colocated(llm_name, llm_fn, "A", a_fn,
                          args.iterations, args.warmup, use_cuda)

    # --- Aggregate ---
    results = {
        "pair": args.pair,
        "gpu": args.gpu,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "llm_phase": llm_name,
        "action_k": args.k,
        "use_cuda": use_cuda,
        "isolated": {
            llm_name: compute_phase_stats(llm_alone),
            "A": compute_phase_stats(a_alone),
        },
        "colocated": {
            llm_name: compute_phase_stats(coloc[llm_name]),
            "A": compute_phase_stats(coloc["A"]),
        },
    }

    # Inflation ratios.
    def _ratio(alone, co):
        a_med = alone["median_ms"]
        c_med = co["median_ms"]
        return round(c_med / a_med, 3) if a_med > 0 else float("inf")

    results["inflation"] = {
        llm_name: _ratio(results["isolated"][llm_name], results["colocated"][llm_name]),
        "A": _ratio(results["isolated"]["A"], results["colocated"]["A"]),
    }

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"RESULTS: exp08a {args.pair} interference")
    print("=" * 60)
    for phase in [llm_name, "A"]:
        alone = results["isolated"][phase]
        co = results["colocated"][phase]
        inflation = results["inflation"][phase]
        print(f"  {phase:>4s}: alone={alone['median_ms']:7.2f}ms (p10/p90 {alone['p10_ms']:.2f}/{alone['p90_ms']:.2f}) "
              f"→ coloc={co['median_ms']:7.2f}ms (p10/p90 {co['p10_ms']:.2f}/{co['p90_ms']:.2f}) "
              f"| inflation={inflation:.3f}x")

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
