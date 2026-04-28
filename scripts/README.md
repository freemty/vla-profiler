# scripts/

All automation for the VLLA project lives here. Each script maps to an
experiment, an infrastructure task, or a visualization pipeline.

**Entry rule:** never guess a script — find it in the table below first.

## Profiling runners (per experiment)

| Script | Exp | Env | Entry command |
|--------|-----|-----|---------------|
| `src/run_tasks.py` (via `launch_exp.sh`) | exp01a, exp02a, exp05a, exp05b, exp06a | `vit-probe` (conda) | `bash scripts/launch_exp.sh <GPU> <config>` |
| `launch_pizero.sh` | exp07a | `.venvs/pizero` (uv) | `bash scripts/launch_pizero.sh <GPU> pizero/profiling` |
| `profile_fastwam.py` | exp04a | `.venvs/lingbot-vla` (uv, WAM fork) | standalone, see `exp/exp04a/README.md` |
| `profile_lingbot_va.py` | exp04b | `.venvs/lingbot-vla` (uv) | standalone, see `exp/exp04b/README.md` |
| `exp08a_interference.py` | exp08a | `vit-probe` | `python scripts/exp08a_interference.py --pair DA --gpu 0` |
| `exp08b_interference_matrix.py` (via `launch_exp08b.sh`) | exp08b | `vit-probe` | `bash scripts/launch_exp08b.sh <GPU> [--pairs-only]` |
| `exp08c_contention_model.py` | exp08c | `vit-probe` (CPU-only, numpy) | `python scripts/exp08c_contention_model.py --data-dir exp/exp08b` |

`_profiling_stats.py` — shared helper used by the standalone `profile_*.py`
scripts to emit Hydra-PhaseTimer-compatible JSON. Not a runnable entry.

## Visualization

| Script | Purpose |
|--------|---------|
| `viz_attention_heatmap.py` | Per-layer attention heatmaps from captured Q/K (exp01b, exp05a) |
| `viz_attention_real_image.py` | Same, but feeds a real image through LingBot's patchifier |

## Validation / reproduce

| Script | Purpose |
|--------|---------|
| `wam_demo_reproduce.py` | Single-forward-pass validation for Fast-WAM / LingBot-VA / NitroGen (smoke test for exp04a/04b/06a) |

## Infrastructure (xdlab23)

| Script | Purpose |
|--------|---------|
| `sync_to_remote.sh` | Push local repo to xdlab23 via git bundle (GitHub blocked by firewall) |
| `launch_exp.sh` | Launch Hydra experiment on xdlab23 under `vit-probe` conda env |
| `launch_exp08b.sh <gpu> [--pairs-only\|--multi-only]` | Sequentially run pair / triple / quad combos on one GPU |
| `launch_pizero.sh` | Launch on xdlab23 under `.venvs/pizero` uv env |
| `run_remote.sh` | One-shot SSH launcher (wraps `launch_exp.sh` via ssh) |
| `run_local.sh` | Same entry for local GPU if available |
| `download-results.sh` | Rsync `/data1/ybyang/vlla/output/` back to local (xdlab23-specific) |
| `download-results.sh <EXP_NAME>` | Narrow download to a single model/exp subdir |
| `download_openvla.sh` | One-off OpenVLA 7B weight download |
| `download_pizero_ckpt.sh` | One-off Pi-Zero checkpoint download |
| `monitor_exp.sh` | `/loop`-compatible status poller against `exp/<id>/results/runs.log` |

## Setup (one-off per env)

| Script | Purpose |
|--------|---------|
| `setup_lingbot_vla.sh` | Create `.venvs/lingbot-vla` uv env, install LingBot-VLA repo |
| `setup_pizero.sh` | Create `.venvs/pizero` uv env, install openpi, download ckpt |
| `setup_fastwam.sh` | Install Fast-WAM inside `.venvs/lingbot-vla` (shared WAM env) |
| `install_libero.sh` | Install LIBERO benchmark into `vit-probe` conda env (PyPI, egl_probe workaround) |

## Test / misc

| Script | Purpose |
|--------|---------|
| `run_tests.sh` | `pytest tests/` wrapper |
| `run_viewer.sh` | Flask viewer on port 5001 |

## Conventions

- `launch_*.sh` / `run_*.sh` — executable shell entry points
- `profile_*.py` — standalone profilers for envs that can't load Hydra
- `exp08*.py` — one file per exp (no shared base, each is a small experiment)
- `setup_*.sh` — one-off environment provisioning (idempotent)
- `download_*.sh` — weights; `download-results.sh` — experiment outputs

## Which runner for my experiment?

1. **Hydra-based (VLM profiling, attention)** — use `src/run_tasks.py` through
   `launch_exp.sh`. Configs live in `configs/<model>/<task>.yaml`.
2. **VLA/WAM with uv env** — use the matching `profile_*.py` or `launch_pizero.sh`.
   These don't go through Hydra because their Python deps conflict with `vit-probe`.
3. **Co-location probe (exp08)** — use `exp08{a,b,c}_*.py`. Own entry, own
   threading model (two Python threads, two CUDA streams).
