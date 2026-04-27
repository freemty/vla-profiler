#!/usr/bin/env python
"""exp08b self-consistency gate.

Runs the same phase (A only) in 3 different combos' isolated sections and asserts
the isolated median is within ±5% across combos. The 2026-04-27 broken harness
showed A isolated 51.2ms in EA/PA but 36.8ms in DA — a 39% order effect that
invalidates pairwise comparisons. A passing gate is a prerequisite for any exp08
data being trusted.

Usage:
    python scripts/verify_exp08b_baseline.py --gpu 0 --iterations 20

Exits 0 if the gate passes, 1 otherwise. Prints per-combo baseline + spread.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_combo(gpu: int, combo: str, iterations: int, out_path: Path) -> dict:
    subprocess.run(
        [
            "python",
            "scripts/exp08b_interference_matrix.py",
            "--gpu", str(gpu),
            "--combo", combo,
            "--iterations", str(iterations),
            "--warmup", "15",
            "--output", str(out_path),
        ],
        check=True,
    )
    return json.load(open(out_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Max relative spread of A's isolated median across combos")
    args = parser.parse_args()

    scratch = Path("exp/exp08b/_verify")
    scratch.mkdir(parents=True, exist_ok=True)

    combos = ["EA", "PA", "DA"]
    baselines: dict[str, float] = {}
    for c in combos:
        out = scratch / f"result_{c}.json"
        data = run_combo(args.gpu, c, args.iterations, out)
        baselines[c] = statistics.median(data["isolated"]["A"]["all_ms"])
        print(f"{c}: A isolated median = {baselines[c]:.2f} ms")

    vals = list(baselines.values())
    spread = (max(vals) - min(vals)) / statistics.mean(vals)
    print(f"\nA-baseline relative spread: {spread * 100:.1f}% (tol {args.tol * 100:.1f}%)")

    if spread > args.tol:
        print("\n❌ GATE FAIL — A's isolated baseline drifts across combos.")
        print("   Fix order effects / state leakage before trusting exp08b numbers.")
        return 1
    print("\n✅ GATE PASS — A's isolated baseline is stable across combos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
