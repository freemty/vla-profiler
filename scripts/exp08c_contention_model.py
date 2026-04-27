"""exp08c — GPU kernel-level contention model

Fits exp08a/b interference data to build an analytical model predicting
colocation inflation. Three candidate models:

  M1: Additive resource model
      inflation(X,Y) = 1 + α_X * util_Y  (linear in co-runner's utilization)

  M2: Bottleneck resource model
      inflation(X) = max(1, Σ_resource [demand_X(r) + demand_Y(r)] / capacity(r))

  M3: Empirical interaction matrix
      inflation(X|Y) = β_{X,Y}  (learned per-pair constant)

Usage:
  # Fit from exp08a pilot data
  python scripts/exp08c_contention_model.py \
      --data-dir exp/exp08a/ --output exp/exp08c/model_pilot.json

  # Fit from full exp08b matrix
  python scripts/exp08c_contention_model.py \
      --data-dir exp/exp08b/ --output exp/exp08c/model_full.json

  # Predict inflation for a new combo
  python scripts/exp08c_contention_model.py \
      --model exp/exp08c/model_full.json --predict EPD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Phase resource profiles (from roofline analysis + profiling data)           #
# --------------------------------------------------------------------------- #

# Normalized resource demand vectors: [compute_util, bw_util, dispatch_rate]
# Values from docs/specs/2026-04-26-epda-roofline-analysis.md
PHASE_RESOURCE_PROFILE = {
    "E": np.array([0.60, 0.10, 0.30]),   # moderate compute, low BW, moderate dispatch
    "P": np.array([0.80, 0.19, 0.40]),   # high compute, moderate BW
    "D": np.array([0.05, 0.73, 0.20]),   # low compute, high BW (memory-bound)
    "A": np.array([0.15, 0.05, 0.85]),   # low compute, low BW, high dispatch (latency-bound)
}

RESOURCE_NAMES = ["compute", "hbm_bw", "dispatch"]


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_interference_data(data_dir: str) -> List[Dict]:
    """Load all results_*.json from a directory."""
    records = []
    data_path = Path(data_dir)
    for f in sorted(data_path.glob("results_*.json")):
        with open(f) as fh:
            data = json.load(fh)

        # Normalize: exp08a uses "pair"/"llm_phase", exp08b uses "combo"/"phases"
        if "combo" in data:
            combo = data["combo"]
            phases = data["phases"]
        elif "pair" in data:
            combo = data["pair"]
            llm_ph = data["llm_phase"]
            phases = [llm_ph, "A"]
        else:
            continue

        inflation = data.get("inflation", {})
        isolated = data.get("isolated", {})
        colocated = data.get("colocated", {})

        for ph in phases:
            if ph in inflation and ph in isolated and ph in colocated:
                records.append({
                    "combo": combo,
                    "phase": ph,
                    "co_runners": [p for p in phases if p != ph],
                    "isolated_ms": isolated[ph]["median_ms"],
                    "colocated_ms": colocated[ph]["median_ms"],
                    "inflation": inflation[ph],
                })

    return records


# --------------------------------------------------------------------------- #
# Model M1: Additive resource contention                                      #
# --------------------------------------------------------------------------- #

def fit_m1_additive(records: List[Dict]) -> Dict:
    """Fit: inflation(X|Y) = 1 + Σ_i α_i * profile_X[i] * profile_Y[i]

    Each resource dimension has a learned sensitivity coefficient α_i.
    """
    n = len(records)
    if n == 0:
        return {"model": "M1_additive", "coefficients": [], "r2": 0.0}

    # Build feature matrix: for each record, compute element-wise product
    # of target phase profile and sum of co-runner profiles
    X = np.zeros((n, len(RESOURCE_NAMES)))
    y = np.zeros(n)

    for i, rec in enumerate(records):
        ph = rec["phase"]
        co_runners = rec["co_runners"]
        ph_profile = PHASE_RESOURCE_PROFILE.get(ph, np.zeros(3))
        co_profile = sum(PHASE_RESOURCE_PROFILE.get(c, np.zeros(3)) for c in co_runners)
        X[i] = ph_profile * co_profile
        y[i] = rec["inflation"] - 1.0  # target is inflation above 1.0

    # Least squares: y = X @ α
    alpha, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ alpha + 1.0
    y_actual = y + 1.0
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": "M1_additive",
        "coefficients": {name: round(float(alpha[i]), 4) for i, name in enumerate(RESOURCE_NAMES)},
        "r2": round(r2, 4),
        "n_samples": n,
    }


# --------------------------------------------------------------------------- #
# Model M2: Bottleneck resource saturation                                    #
# --------------------------------------------------------------------------- #

def fit_m2_bottleneck(records: List[Dict]) -> Dict:
    """Fit: inflation(X|Y) = max_i(1, (profile_X[i] + profile_Y[i]) * γ_i)

    Each resource has a scaling factor γ_i (learned).
    Uses iterative least squares on the max-binding resource.
    """
    n = len(records)
    if n == 0:
        return {"model": "M2_bottleneck", "coefficients": [], "r2": 0.0}

    y = np.array([rec["inflation"] for rec in records])

    # For each resource, compute total demand
    demands = np.zeros((n, len(RESOURCE_NAMES)))
    for i, rec in enumerate(records):
        ph = rec["phase"]
        co_runners = rec["co_runners"]
        total_profile = PHASE_RESOURCE_PROFILE.get(ph, np.zeros(3))
        for c in co_runners:
            total_profile = total_profile + PHASE_RESOURCE_PROFILE.get(c, np.zeros(3))
        demands[i] = total_profile

    # Fit γ per resource (simple: γ_i = median(y / demand_i) for demand_i > 0)
    gammas = np.ones(len(RESOURCE_NAMES))
    for j in range(len(RESOURCE_NAMES)):
        mask = demands[:, j] > 0.01
        if mask.any():
            gammas[j] = float(np.median(y[mask] / demands[mask, j]))

    # Predict: inflation = max over resources of (demand * gamma)
    y_pred = np.max(demands * gammas, axis=1)
    y_pred = np.maximum(y_pred, 1.0)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": "M2_bottleneck",
        "coefficients": {name: round(float(gammas[i]), 4) for i, name in enumerate(RESOURCE_NAMES)},
        "r2": round(r2, 4),
        "n_samples": n,
    }


# --------------------------------------------------------------------------- #
# Model M4: Asymmetric vulnerability-aggressiveness                           #
# --------------------------------------------------------------------------- #

def fit_m4_asymmetric(records: List[Dict]) -> Dict:
    """Fit: inflation(X|Y) = 1 + vulnerability(X) * aggressiveness(Y)

    Key insight: interference is asymmetric. D is fragile (high vulnerability)
    but low aggressiveness; A is robust (low vulnerability) but high aggressiveness.
    Each phase has two learned scalars: v (how much it suffers) and a (how much
    it disrupts others).

    For N-phase combos: inflation(X|{Y,Z}) = 1 + v_X * Σ a_Y
    """
    n = len(records)
    phases_seen = sorted(set(r["phase"] for r in records))
    if n == 0:
        return {"model": "M4_asymmetric", "vulnerability": {}, "aggressiveness": {}, "r2": 0.0}

    phase_idx = {ph: i for i, ph in enumerate(phases_seen)}
    n_phases = len(phases_seen)

    # Build feature matrix: each row has v_X * a_Y features
    # For pair (X, Y): feature[phase_idx[X], phase_idx[Y]] = 1
    # Target: inflation(X) - 1
    X = np.zeros((n, n_phases * 2))  # [v_0..v_N, a_0..a_N]
    y = np.zeros(n)

    for i, rec in enumerate(records):
        ph = rec["phase"]
        co_runners = rec["co_runners"]
        y[i] = rec["inflation"] - 1.0

    # Since inflation = 1 + v_X * Σ a_Y is nonlinear in (v, a),
    # use alternating least squares:
    # Init: v = ones, fit a. Then fix a, fit v. Repeat.
    v = np.ones(n_phases)
    a = np.ones(n_phases)

    for iteration in range(20):
        # Fix v, fit a: y_i = v_{X_i} * Σ_j a_{Y_j} → linear in a
        A_mat = np.zeros((n, n_phases))
        for i, rec in enumerate(records):
            ph_i = phase_idx[rec["phase"]]
            for co in rec["co_runners"]:
                co_i = phase_idx[co]
                A_mat[i, co_i] = v[ph_i]
        a_new, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        a = np.maximum(a_new, 0.01)  # non-negative

        # Fix a, fit v: y_i = v_{X_i} * Σ_j a_{Y_j} → linear in v
        V_mat = np.zeros((n, n_phases))
        for i, rec in enumerate(records):
            ph_i = phase_idx[rec["phase"]]
            aggr_sum = sum(a[phase_idx[co]] for co in rec["co_runners"])
            V_mat[i, ph_i] = aggr_sum
        v_new, _, _, _ = np.linalg.lstsq(V_mat, y, rcond=None)
        v = np.maximum(v_new, 0.01)

    # Predict
    y_pred = np.zeros(n)
    for i, rec in enumerate(records):
        ph_i = phase_idx[rec["phase"]]
        aggr_sum = sum(a[phase_idx[co]] for co in rec["co_runners"])
        y_pred[i] = v[ph_i] * aggr_sum

    y_actual = y
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    vuln = {phases_seen[i]: round(float(v[i]), 4) for i in range(n_phases)}
    aggr = {phases_seen[i]: round(float(a[i]), 4) for i in range(n_phases)}

    return {
        "model": "M4_asymmetric",
        "vulnerability": vuln,
        "aggressiveness": aggr,
        "r2": round(r2, 4),
        "n_samples": n,
    }


# --------------------------------------------------------------------------- #
# Model M3: Empirical interaction matrix                                      #
# --------------------------------------------------------------------------- #

def fit_m3_empirical(records: List[Dict]) -> Dict:
    """Fit: β_{X,Y} = observed inflation for phase X co-running with Y.

    No generalization — just a lookup table. Useful as upper bound for R².
    """
    matrix: Dict[str, Dict[str, float]] = {}
    for rec in records:
        ph = rec["phase"]
        co_key = "+".join(sorted(rec["co_runners"]))
        if ph not in matrix:
            matrix[ph] = {}
        matrix[ph][co_key] = rec["inflation"]

    y = np.array([rec["inflation"] for rec in records])
    r2 = 1.0  # perfect fit by construction

    return {
        "model": "M3_empirical",
        "matrix": matrix,
        "r2": r2,
        "n_samples": len(records),
    }


# --------------------------------------------------------------------------- #
# Prediction                                                                  #
# --------------------------------------------------------------------------- #

def predict_inflation(model: Dict, combo: str) -> Dict[str, float]:
    """Predict inflation for each phase in a combo using a fitted model."""
    phases = list(combo)
    predictions = {}

    if model["model"] == "M1_additive":
        coeffs = np.array([model["coefficients"][r] for r in RESOURCE_NAMES])
        for ph in phases:
            co_runners = [p for p in phases if p != ph]
            ph_profile = PHASE_RESOURCE_PROFILE.get(ph, np.zeros(3))
            co_profile = sum(PHASE_RESOURCE_PROFILE.get(c, np.zeros(3)) for c in co_runners)
            inflation = 1.0 + float(np.dot(ph_profile * co_profile, coeffs))
            predictions[ph] = round(max(inflation, 1.0), 3)

    elif model["model"] == "M2_bottleneck":
        gammas = np.array([model["coefficients"][r] for r in RESOURCE_NAMES])
        for ph in phases:
            co_runners = [p for p in phases if p != ph]
            total = PHASE_RESOURCE_PROFILE.get(ph, np.zeros(3))
            for c in co_runners:
                total = total + PHASE_RESOURCE_PROFILE.get(c, np.zeros(3))
            inflation = float(np.max(total * gammas))
            predictions[ph] = round(max(inflation, 1.0), 3)

    elif model["model"] == "M4_asymmetric":
        vuln = model["vulnerability"]
        aggr = model["aggressiveness"]
        for ph in phases:
            co_runners = [p for p in phases if p != ph]
            v = vuln.get(ph, 1.0)
            a_sum = sum(aggr.get(c, 1.0) for c in co_runners)
            inflation = 1.0 + v * a_sum
            predictions[ph] = round(max(inflation, 1.0), 3)

    elif model["model"] == "M3_empirical":
        for ph in phases:
            co_runners = [p for p in phases if p != ph]
            co_key = "+".join(sorted(co_runners))
            predictions[ph] = model["matrix"].get(ph, {}).get(co_key, float("nan"))

    return predictions


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="exp08c — GPU contention model")
    p.add_argument("--data-dir", type=str, help="Directory with results_*.json files")
    p.add_argument("--output", type=str, help="Output model JSON")
    p.add_argument("--model", type=str, help="Pre-fitted model JSON for prediction")
    p.add_argument("--predict", type=str, help="Combo to predict (e.g., EPD)")
    args = p.parse_args()

    if args.model and args.predict:
        with open(args.model) as f:
            models = json.load(f)
        print(f"\nPredicting inflation for combo: {args.predict}")
        for model_data in models["models"]:
            preds = predict_inflation(model_data, args.predict)
            print(f"\n  {model_data['model']} (R²={model_data['r2']:.4f}):")
            for ph, inf in preds.items():
                print(f"    {ph}: {inf:.3f}x")
        return

    if not args.data_dir:
        p.error("--data-dir required for fitting")

    records = load_interference_data(args.data_dir)
    if not records:
        print(f"[error] No interference data found in {args.data_dir}")
        sys.exit(1)

    print(f"\nLoaded {len(records)} data points from {args.data_dir}")
    print(f"Combos: {sorted(set(r['combo'] for r in records))}")
    print(f"Phases: {sorted(set(r['phase'] for r in records))}")

    # Fit all four models
    m1 = fit_m1_additive(records)
    m2 = fit_m2_bottleneck(records)
    m4 = fit_m4_asymmetric(records)
    m3 = fit_m3_empirical(records)

    output = {
        "experiment": "exp08c",
        "data_dir": args.data_dir,
        "n_records": len(records),
        "models": [m1, m2, m4, m3],
        "data": records,
    }

    # Report
    print(f"\n{'='*60}")
    print("CONTENTION MODEL FIT RESULTS")
    print(f"{'='*60}")
    for model in [m1, m2, m4, m3]:
        print(f"\n  {model['model']}:")
        print(f"    R² = {model['r2']:.4f}")
        if "coefficients" in model:
            for name, val in model["coefficients"].items():
                print(f"    {name}: {val:.4f}")
        if "vulnerability" in model:
            print("    vulnerability (how fragile):")
            for ph, val in model["vulnerability"].items():
                print(f"      {ph}: {val:.4f}")
            print("    aggressiveness (how disruptive):")
            for ph, val in model["aggressiveness"].items():
                print(f"      {ph}: {val:.4f}")

    # Cross-validation
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION (predict vs observed)")
    print(f"{'='*60}")
    print(f"  {'Combo':>6s} {'Phase':>5s} | {'Observed':>8s} | {'M1':>8s} {'M2':>8s} {'M4':>8s}")
    print(f"  {'-'*6} {'-'*5}-+-{'-'*8}-+-{'-'*8} {'-'*8} {'-'*8}")

    for rec in records:
        combo = rec["combo"]
        ph = rec["phase"]
        obs = rec["inflation"]
        pred_m1 = predict_inflation(m1, combo).get(ph, float("nan"))
        pred_m2 = predict_inflation(m2, combo).get(ph, float("nan"))
        pred_m4 = predict_inflation(m4, combo).get(ph, float("nan"))
        print(f"  {combo:>6s} {ph:>5s} | {obs:>7.3f}x | {pred_m1:>7.3f}x {pred_m2:>7.3f}x {pred_m4:>7.3f}x")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
