"""Consolidate per-model JSON results into the reproducibility matrix."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


LATENCY_MODELS = {
    "act": {"display": "ACT", "paradigm": "single-forward"},
    "lingbot_vla_4b": {"display": "LingBot-VLA", "paradigm": "flow-head"},
    "nitrogen500m": {"display": "NitroGen-500M", "paradigm": "flow-dit"},
    "pizero_real": {"display": "Pi-Zero", "paradigm": "flow-dit"},
    "fastwam_5step": {"display": "Fast-WAM", "paradigm": "skip-imagination"},
    "lingbotva": {"display": "LingBot-VA", "paradigm": "full-wam"},
    "qwen_vl_7b": {"display": "Qwen-VL-7B", "paradigm": "vlm-only"},
}

LIBERO_DIRS = {
    "fastwam": Path("exp/exp09a_fastwam_libero"),
    "pizero": Path("exp/exp09b_pizero_libero"),
    "lingbot_vla_4b": Path("exp/exp09c_lingbotvla_libero"),
    "lingbotva": Path("exp/exp09d_lingbotva_libero"),
}

SUITES = ["spatial", "object", "goal", "10"]


def load_latency(base_dir: Path, model_key: str) -> dict[str, Any] | None:
    p = base_dir / f"{model_key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_libero(model_key: str) -> dict[str, float] | None:
    d = LIBERO_DIRS.get(model_key)
    if d is None or not d.exists():
        return None
    suites = {}
    for suite in SUITES:
        f = d / f"results_{suite}.json"
        if f.exists():
            data = json.loads(f.read_text())
            suites[suite] = data.get("success_rate", data.get("overall_success_rate", 0.0))
    return suites if suites else None


def build_matrix(latency_dir: Path = Path("exp/exp09_latency_rerun")) -> dict[str, Any]:
    rows = []
    for key, meta in LATENCY_MODELS.items():
        row = {
            "model": key,
            "display": meta["display"],
            "paradigm": meta["paradigm"],
        }
        lat = load_latency(latency_dir, key)
        if lat:
            row["total_ms"] = lat.get("total_e2e_ms", lat.get("total_ms"))
            row["hz"] = lat.get("hz")
            row["phases"] = {
                k: v.get("mean_ms", v) if isinstance(v, dict) else v
                for k, v in lat.get("phases", {}).items()
            }
            row["weights"] = lat.get("weights", "unknown")
        lib = load_libero(key)
        if lib:
            row["libero"] = lib
            row["libero_avg"] = sum(lib.values()) / len(lib)
        rows.append(row)
    return {"version": "v0.9.0", "rows": rows}


def main():
    matrix = build_matrix()
    out = Path("exp/reproducibility_matrix.json")
    out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    print(f"Wrote {out} ({len(matrix['rows'])} models)")
    for row in matrix["rows"]:
        total = row.get("total_ms", "—")
        libero = row.get("libero_avg", "—")
        if isinstance(libero, float):
            libero = f"{libero:.1%}"
        print(f"  {row['display']:20s}  {str(total):>8s} ms  LIBERO: {libero}")


if __name__ == "__main__":
    main()
