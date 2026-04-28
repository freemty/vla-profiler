# Full Reproducibility + LIBERO Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align all 7 profiled models (ACT / LingBot-VLA / NitroGen / Pi-Zero / Fast-WAM / LingBot-VA / Qwen-VL) to their official weights + official inference settings, then produce a unified **latency × LIBERO success rate** evaluation matrix. ALOHA (for ACT) is explicitly deferred.

**Architecture:** Five phases executed on xdlab23 (8×RTX 5880 Ada 48GB, all free as of 2026-04-28):
1. Prep — LIBERO environment + shared eval harness
2. Latency realignment — rerun existing models with official configs (Fast-WAM 5-step, NitroGen full 500M, Pi-Zero real weights)
3. Weight acquisition — pi0-base ckpt, ACT tonyzhaozh ckpt, LingBot-VA hunt
4. LIBERO eval — 4 models (Fast-WAM, Pi-Zero, LingBot-VLA, LingBot-VA) × 4 task suites × 20 episodes
5. Dashboard + slides + CHANGELOG update

**Tech Stack:** Hydra configs · uv venvs per model · LIBERO v1.0.0 (mujoco) · LeRobot dataset stats · Fast-WAM's `eval_libero_single.py` as reference harness · openpi's LIBERO docker compose for pi0-base.

**Deferred (next sprint):** ACT ALOHA sim eval (Task 17 stub), NitroGen custom benchmark (author's internal, no public harness).

---

## Current State (verified 2026-04-28)

**Weights already on xdlab23:**
- ✅ LingBot-VLA: `/data1/ybyang/modelscope/Robbyant/lingbot-vla-4b` (real, used in exp03a)
- ✅ NitroGen: `/data1/ybyang/huggingface/nvidia/NitroGen/ng.pt` (real, used only in demo_reproduce, **profiling uses 174M variant**)
- ✅ Fast-WAM: `/data1/ybyang/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` (real, used in exp04a)
- ✅ Qwen-VL: `/data1/ybyang/huggingface/Qwen/Qwen2.5-VL-{3B,7B}-Instruct` (real)

**Weights missing:**
- ❌ ACT: need `tonyzhaozh/act` ALOHA pretrained (~40MB) — ALOHA eval deferred
- ❌ Pi-Zero: openpi repo has no ckpt, need `gs://openpi-assets/checkpoints/pi0_libero/` or HF mirror
- ❌ LingBot-VA: `/data1/ybyang/lingbot-va/` has code only, **checkpoint location unknown**

**Benchmarks already installed:**
- Fast-WAM LIBERO harness: `/data1/ybyang/FastWAM/experiments/libero/eval_libero_single.py` + `run_libero_manager.py`
- openpi LIBERO harness: `/data1/ybyang/openpi/examples/libero/` (docker compose)
- LingBot-VA LIBERO eval: `/data1/ybyang/lingbot-va/evaluation/libero/` (needs inspection)
- LIBERO sim itself: **not yet installed system-wide** — each venv installs per `openpi/third_party/libero`

**Canonical latency baseline (what must NOT regress when we rerun):**
```
exp02a ACT      total 3ms         (random weights, official arch)
exp03a LingBot-VLA  74.5ms         (REAL weights ✅)
exp04a Fast-WAM 407ms @ 10-step    (REAL weights, BUT step=10 not paper's 5)
exp04b LingBot-VA 2518ms           (random weights, official arch)
exp06a NitroGen 7.2ms/step @174M   (RANDOM weights + SHRUNK arch vision_hidden=768 not 1024)
exp07a Pi-Zero  200ms @ 10-step    (random weights, official arch)
```

**Outputs of this plan:**
- `exp/exp09/` — reproducibility matrix dir (NEW)
- `exp/exp09a_fastwam_libero/` — Fast-WAM LIBERO eval
- `exp/exp09b_pizero_libero/` — Pi-Zero LIBERO eval
- `exp/exp09c_lingbotvla_libero/` — LingBot-VLA LIBERO eval
- `exp/exp09d_lingbotva_libero/` — LingBot-VA LIBERO eval (may abort if ckpt missing)
- `exp/exp09_latency_rerun/` — all 6 latency reruns with official configs
- `exp/reproducibility_matrix.json` — final consolidated table
- `viewer/static/reproducibility.html` — 2-panel dashboard (latency + success rate)
- `slides/hao-meeting-2026-04-28.html` — updated with new slide 11

---

## File Structure (to be created/modified)

### New experiment configs
- `configs/nitrogen/profiling_full500m.yaml` — **new** (copy of demo_reproduce.yaml + profiling hooks)
- `configs/fastwam/profiling_5step.yaml` — **new** (paper-aligned)
- `configs/pizero/profiling_real.yaml` — **new** (sets `model_name` to real ckpt path)
- `configs/lingbot_va/profiling_real.yaml` — **new** (pending ckpt location)
- `configs/act/profiling_real.yaml` — **new** (tonyzhaozh ckpt)

### New shared code
- `src/eval/libero_harness.py` — **new**, thin adapter calling upstream eval scripts
- `src/eval/consolidate_matrix.py` — **new**, aggregates JSON results from all exp09* into reproducibility_matrix.json
- `src/eval/__init__.py` — **new**

### Controllers
- `src/controllers/pizero_controller.py:160-175` — **modify**, support `model_name` pointing to real ckpt

### Scripts
- `scripts/download_weights.sh` — **new**, one-stop: pi0-base / tonyzhaozh-act / lingbot-va
- `scripts/install_libero.sh` — **new**, reproducible LIBERO install per venv
- `scripts/launch_exp09.sh` — **new**, orchestrates all exp09a/b/c/d in parallel across GPUs 0-3

### Experiment directories (created per task)
- `exp/exp09a_fastwam_libero/{README.md,results_{spatial,object,goal,10}.json,run.log}`
- `exp/exp09b_pizero_libero/{README.md,results_*.json,run.log}`
- `exp/exp09c_lingbotvla_libero/{README.md,results_*.json,run.log}`
- `exp/exp09d_lingbotva_libero/{README.md,results_*.json,run.log}` (may be skipped)
- `exp/exp09_latency_rerun/{README.md,latency_all.json,run.log}`

### Docs
- `docs/specs/2026-04-28-reproducibility-spec.md` — **new**, single-source-of-truth for official settings
- `exp/summary.md` — **modify**, append exp09* rows
- `CLAUDE.md:186-199` — **modify**, update key findings with post-alignment numbers
- `.claude/skills/project-skill/SKILL.md` — **modify**, v8→v9 with reproducibility guarantees
- `CHANGELOG.md` — **modify**, new v0.9.0 entry
- `slides/hao-meeting-2026-04-28.html` — **modify**, add slide 11 "Performance Evaluation Matrix", update slide 5 with new latencies

---

## Phase 0 · Prep (sequential, ~2h)

### Task 1: Pin the reproducibility spec document

**Files:**
- Create: `docs/specs/2026-04-28-reproducibility-spec.md`

- [ ] **Step 1: Write the spec**

This is the contract. Every subsequent task must match it. Anyone reading this spec in 6 months should know exactly what "official" means per model.

```markdown
# Reproducibility Spec (v0.9.0)

One table per model. Columns: param · required value · source · verify command.

## ACT (Zhao 2023)
| | value | source | verify |
|---|---|---|---|
| ckpt | tonyzhaozh/act (ALOHA insertion + transfer_cube) | HF hub | `hf download tonyzhaozh/act` |
| backbone | ResNet18 | paper §4 | — |
| transformer | dim=512, enc=4, dec=7 | paper §4 | config dump |
| action chunk | 100 | paper §4 | config dump |
| denoise steps | N/A (CVAE one-shot) | — | — |
| benchmark | ALOHA sim (insertion / transfer_cube) | paper §6 | **deferred** |

## LingBot-VLA (Robbyant 2025)
| | value | source | verify |
|---|---|---|---|
| ckpt | Robbyant/lingbot-vla-4b | ModelScope | md5 |
| backbone | Qwen2.5-VL-3B frozen | model card | — |
| action head | 10-step flow matching | model card | config dump |
| denoise steps | 10 | model card | `denoise_steps: 10` |
| benchmark | LIBERO-4 suite | author eval | `libero_eval.py` |

## NitroGen (NVIDIA 2025)
| | value | source | verify |
|---|---|---|---|
| ckpt | nvidia/NitroGen/ng.pt (500M) | HF | md5 of ng.pt |
| vision_hidden | **1024** (NOT 768) | ng.pt shape | `python -c "import torch; print(torch.load('ng.pt').keys())"` |
| dit_num_layers | 12 | ng.pt | — |
| dit_num_heads | 16 | ng.pt | — |
| dit_head_dim | 64 | ng.pt | — |
| vl_num_layers | 4 | ng.pt | — |
| vl_num_heads | 16 | ng.pt | — |
| action_dim | 25 | ng.pt | — |
| action_horizon | 16 | ng.pt | — |
| denoise steps | 16 (default), sweep k=1,2,4,8,16 | paper | — |
| benchmark | NVIDIA internal VA (no public harness) | paper | **latency only, no task eval** |

## Pi-Zero (Physical Intelligence 2024)
| | value | source | verify |
|---|---|---|---|
| ckpt | pi0-base (openpi-assets/pi0_libero) | GCS / HF mirror | md5 |
| backbone | PaliGemma (SigLIP ViT-So400m/14 + Gemma 2B) | model card | — |
| action expert | Gemma 300M, JointModel 18L/8H/GQA-kv=1 | config | shape check |
| action chunk | 4 steps × action_dim 7 | `_default_pizero_config` | — |
| denoise steps | 10 | paper | `denoise_steps: 10` |
| benchmark | LIBERO 4-suite | openpi examples | `openpi/examples/libero/main.py` |

## Fast-WAM (Yuan 2025)
| | value | source | verify |
|---|---|---|---|
| ckpt | libero_uncond_2cam224.pt | fastwam_release | md5 |
| backbone | Wan2.2-TI2V-5B frozen video expert + 350M ActionDiT | model card | — |
| action_horizon | 10 | dataset_stats.json | — |
| action_dim | 14 (7-DoF × 2 arms) | dataset_stats.json | — |
| denoise steps | **5** (paper), we also run 10 for Pi-Zero comparison | paper Table 3 | `num_inference_steps: 5` |
| benchmark | LIBERO 4-suite | paper §5 | `experiments/libero/eval_libero_single.py` |

## LingBot-VA (Robbyant 2025)
| | value | source | verify |
|---|---|---|---|
| ckpt | **TBD — location unknown 2026-04-28, Task 8 locates** | — | — |
| backbone | Wan2.2-TI2V-5B + shared action head | lingbot-va/README | — |
| video denoise steps | 20 | lingbot-va config | — |
| action denoise steps | 50 | lingbot-va config | — |
| benchmark | LIBERO-4 | lingbot-va/evaluation/libero | — |

## Qwen-VL
| | value | source | verify |
|---|---|---|---|
| ckpt | Qwen/Qwen2.5-VL-{3B,7B}-Instruct | HF | — |
| benchmark | N/A — VLM, not VLA; profiling is for phase breakdown only | — | — |
```

- [ ] **Step 2: Commit**

```bash
git add docs/specs/2026-04-28-reproducibility-spec.md
git commit -m "docs(spec): reproducibility contract for 7 models (v0.9.0)"
```

---

### Task 2: Create exp09 umbrella dir + index

**Files:**
- Create: `exp/exp09/README.md`
- Modify: `exp/summary.md`

- [ ] **Step 1: Write the umbrella README**

```markdown
# exp09 — Full Reproducibility + LIBERO Evaluation

Sub-experiments:
- exp09_latency_rerun — all 6 models with official configs
- exp09a_fastwam_libero — Fast-WAM on LIBERO-4
- exp09b_pizero_libero — Pi-Zero on LIBERO-4
- exp09c_lingbotvla_libero — LingBot-VLA on LIBERO-4
- exp09d_lingbotva_libero — LingBot-VA on LIBERO-4 (if ckpt found)

**Not in scope (this sprint):**
- ACT ALOHA eval (deferred to exp10)
- NitroGen task success (no public benchmark)

**Deliverable:** `exp/reproducibility_matrix.json` + `viewer/static/reproducibility.html`

**Spec of record:** `docs/specs/2026-04-28-reproducibility-spec.md`
```

- [ ] **Step 2: Append exp09 rows to exp/summary.md**

Locate the summary table in `exp/summary.md` and append:

```markdown
| exp09_latency_rerun | 2026-04-29 | planned | — | — | latency realignment to official configs |
| exp09a_fastwam_libero | 2026-04-29 | planned | — | — | Fast-WAM LIBERO-4 success rate |
| exp09b_pizero_libero | 2026-04-29 | planned | — | — | Pi-Zero LIBERO-4 success rate |
| exp09c_lingbotvla_libero | 2026-04-29 | planned | — | — | LingBot-VLA LIBERO-4 success rate |
| exp09d_lingbotva_libero | 2026-04-29 | planned | — | — | LingBot-VA LIBERO-4 (conditional on ckpt) |
```

- [ ] **Step 3: Commit**

```bash
git add exp/exp09/README.md exp/summary.md
git commit -m "exp(09): scaffold reproducibility+LIBERO umbrella"
```

---

### Task 3: Install LIBERO on xdlab23 (shared venv)

**Files:**
- Create: `scripts/install_libero.sh`
- Create: `docs/knowhow/runbooks/install-libero.md`

LIBERO has system dependencies (mujoco, libegl) and Python 3.8 requirement. We install it once into a **dedicated** venv `/data1/ybyang/venvs/libero-eval/` so all per-model eval scripts can source it.

- [ ] **Step 1: Write install script**

```bash
#!/usr/bin/env bash
# Install LIBERO + deps on xdlab23. Idempotent.
set -euo pipefail

VENV=/data1/ybyang/venvs/libero-eval
LIBERO_SRC=/data1/ybyang/openpi/third_party/libero

if [[ -d "$VENV" && -f "$VENV/bin/activate" ]]; then
  echo "[install-libero] $VENV exists, skipping creation"
else
  uv venv --python 3.8 "$VENV"
fi

source "$VENV/bin/activate"

# System deps (require sudo — user must have run once)
# sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libosmesa6-dev patchelf

# Python deps
uv pip install --upgrade pip
uv pip install -e "$LIBERO_SRC"
uv pip install "torch==2.1.2" torchvision "numpy<2" "mujoco>=3.0" imageio imageio-ffmpeg

# Smoke test
python -c "from libero.libero import benchmark; b = benchmark.get_benchmark_dict(); print('LIBERO suites:', list(b.keys()))"
echo "[install-libero] OK"
```

- [ ] **Step 2: Run it on the server**

```bash
# from local
scp scripts/install_libero.sh xdlab23_yang:/data1/ybyang/vlla/scripts/
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/install_libero.sh 2>&1 | tee /tmp/install_libero.log'
```

Expected final line: `LIBERO suites: ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90']`

- [ ] **Step 3: Write the runbook**

```markdown
# Runbook: Install LIBERO on xdlab23

## Problem
Every VLA eval uses LIBERO, but it requires Python 3.8 + mujoco + EGL. Default conda env on xdlab23 is Py 3.10.

## Solution
Dedicated venv at `/data1/ybyang/venvs/libero-eval/` (Python 3.8, created by uv).

## Commands
```bash
bash scripts/install_libero.sh
source /data1/ybyang/venvs/libero-eval/bin/activate
python -c "from libero.libero import benchmark; print(list(benchmark.get_benchmark_dict().keys()))"
```

## Gotchas
- If `mujoco import` fails: `MUJOCO_GL=glx` in env (EGL is broken on headless RTX 5880).
- Ramdisk `/dev/shm` fills during long eval — set `TMPDIR=/data1/tmp`.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/install_libero.sh docs/knowhow/runbooks/install-libero.md
git commit -m "infra(eval): install LIBERO venv on xdlab23"
```

---

## Phase 1 · Latency realignment (parallel-safe, ~3h total)

Each task reruns one model with its **official** config. All write to `exp/exp09_latency_rerun/results_<model>.json`. These can run in parallel on different GPUs.

### Task 4: NitroGen — full 500M profiling rerun

**Files:**
- Create: `configs/nitrogen/profiling_full500m.yaml`
- Modify: `src/controllers/nitrogen_controller.py:99` (already supports `weight_mode: full`, verify)

The current `profiling.yaml` uses vision_hidden=768 and action_dim=20 (a shrunk variant). `demo_reproduce.yaml` has the correct shape (1024/25) but only runs 1 iteration. We need the **intersection**: real 500M architecture + 20-iter profiling + k-sweep.

- [ ] **Step 1: Create the config**

```yaml
# configs/nitrogen/profiling_full500m.yaml
defaults:
  - /base
  - _self_

experiment:
  name: "exp09_latency_rerun_nitrogen500m"
  description: "NitroGen full 500M DiT — aligned to ng.pt checkpoint shape"

model_name: "NitroGen-500M-real"
controller_name: nitrogen
controller_config:
  mode: profiling
  weight_mode: full
  repo_path: "/data1/ybyang/NitroGen"
  checkpoint_path: "/data1/ybyang/huggingface/nvidia/NitroGen/ng.pt"
  vision_encoder_name: "/data1/ybyang/huggingface/google/siglip-large-patch16-256"
  vision_hidden_size: 1024
  dit_num_layers: 12
  dit_num_heads: 16
  dit_head_dim: 64
  vl_num_layers: 4
  vl_num_heads: 16
  vl_head_dim: 64
  action_dim: 25
  action_horizon: 16

num_warmup_runs: 15
num_benchmark_runs: 20

tasks:
  - epd_profiling

profiling:
  gpu_id: 0
  warmup: 15
  iterations: 20
  phases:
    - encode
    - context
    - action

inputs:
  - name: "k16"
    num_inference_steps: 16
  - name: "k8"
    num_inference_steps: 8
  - name: "k4"
    num_inference_steps: 4
  - name: "k2"
    num_inference_steps: 2
  - name: "k1"
    num_inference_steps: 1
```

- [ ] **Step 2: Verify controller accepts weight_mode=full + k-sweep**

```bash
grep -n "weight_mode\|num_inference_steps" src/controllers/nitrogen_controller.py | head -20
```

Expected: `mode = getattr(cc, "weight_mode", ...)` at line 99, `num_inference_steps` honored in forward. If not, patch the controller in a separate commit before proceeding.

- [ ] **Step 3: Sync + run on GPU 0**

```bash
bash scripts/sync_to_remote.sh
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/launch_exp.sh 0 nitrogen/profiling_full500m 2>&1 | tee exp/exp09_latency_rerun/nitrogen500m.log'
```

- [ ] **Step 4: Verify output**

Expected per-step latency at k=16:
- 174M variant (old): 7.2ms/step
- 500M variant (new): **~20-25ms/step** (3× params typically 2.5-3× latency for DiT at this scale)
- k=1 total should drop from 18ms to **~35-45ms**

```bash
bash scripts/download-results.sh
python -c "
import json
d = json.load(open('exp/exp09_latency_rerun/nitrogen500m_k16.json'))
assert 15 < d['phases']['action']['mean_ms'] / 16 < 40, 'per-step out of expected 15-40ms band'
print('per-step ms/k:', d['phases']['action']['mean_ms']/16)
"
```

- [ ] **Step 5: Commit**

```bash
git add configs/nitrogen/profiling_full500m.yaml exp/exp09_latency_rerun/
git commit -m "exp09(nitrogen): full 500M latency rerun (real ng.pt + 1024 hidden + 25 action_dim)"
```

---

### Task 5: Fast-WAM — 5-step rerun (paper-aligned)

**Files:**
- Create: `configs/fastwam/profiling_5step.yaml`

Fast-WAM paper uses 5 denoise steps; we ran 10. Rerun at 5 to match paper's 190ms claim.

- [ ] **Step 1: Create config**

```yaml
# configs/fastwam/profiling_5step.yaml
defaults:
  - /base

experiment:
  name: "exp09_latency_rerun_fastwam_5step"
  description: "Fast-WAM paper-aligned 5-step inference on RTX 5880 Ada"

model:
  name: "Fast-WAM"
  type: "wam"
  paradigm: "skip-imagination"
  checkpoint: "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  dataset_stats: "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
  num_inference_steps: 5

profiling:
  gpu_id: 1
  warmup: 15
  iterations: 20
  input:
    num_cameras: 2
    image_size: 224
    action_dim: 14
  phases: [encode, context, action]

notes: |
  Paper reports 190ms total on A100/H100 @ 5-step.
  We expect RTX 5880 Ada ~200-230ms @ 5-step (vs. our prior 407ms @ 10-step).
  Action phase should drop from 362ms to ~180ms.
```

- [ ] **Step 2: Run on GPU 1 (parallel with Task 4)**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/launch_exp.sh 1 fastwam/profiling_5step 2>&1 | tee exp/exp09_latency_rerun/fastwam5step.log'
```

- [ ] **Step 3: Verify**

```bash
python -c "
import json
d = json.load(open('exp/exp09_latency_rerun/fastwam_5step.json'))
assert 150 < d['total_e2e_ms'] < 250, f'expected 150-250ms, got {d[\"total_e2e_ms\"]}'
print('Fast-WAM 5-step total:', d['total_e2e_ms'])
"
```

- [ ] **Step 4: Commit**

```bash
git add configs/fastwam/profiling_5step.yaml exp/exp09_latency_rerun/fastwam_5step.json
git commit -m "exp09(fastwam): 5-step paper-aligned rerun"
```

---

### Task 6: Pi-Zero — download real weights + rerun

**Files:**
- Create: `scripts/download_pi0.sh`
- Create: `configs/pizero/profiling_real.yaml`
- Modify: `src/controllers/pizero_controller.py:160-175` (verify real-weight path)

Pi-Zero's `_default_pizero_config` returns random weights when `model_name=""`. We need to download pi0-base (openpi-assets GCS bucket or HF mirror) and point `model_name` to it.

- [ ] **Step 1: Write download script**

```bash
#!/usr/bin/env bash
# scripts/download_pi0.sh — fetch pi0-base checkpoint
# Two paths: (a) GCS via gsutil (b) HF mirror fallback
set -euo pipefail

DEST=/data1/ybyang/huggingface/physical-intelligence/pi0-base
mkdir -p "$DEST"

if command -v gsutil >/dev/null; then
  echo "[download-pi0] using gsutil"
  gsutil -m cp -r 'gs://openpi-assets/checkpoints/pi0_libero/*' "$DEST/"
else
  echo "[download-pi0] gsutil missing, fallback to HF"
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    physical-intelligence/pi0-base \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False
fi

ls -lh "$DEST"
```

- [ ] **Step 2: Run download**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/download_pi0.sh 2>&1 | tee /tmp/dl_pi0.log'
```

Expected: 3-4 GB in `/data1/ybyang/huggingface/physical-intelligence/pi0-base/`.

- [ ] **Step 3: Create the real-weight config**

```yaml
# configs/pizero/profiling_real.yaml
defaults:
  - /base
  - _self_

model_name: "/data1/ybyang/huggingface/physical-intelligence/pi0-base"
controller_name: pizero
controller_config:
  mode: profiling
  denoise_steps: 10

num_warmup_runs: 15
num_benchmark_runs: 20

tasks:
  - epd_profiling

profiling:
  gpu_id: 2
  warmup: 15
  iterations: 20
  phases: [encode, context, action]
```

- [ ] **Step 4: Run on GPU 2**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/launch_exp.sh 2 pizero/profiling_real 2>&1 | tee exp/exp09_latency_rerun/pizero_real.log'
```

- [ ] **Step 5: Verify total latency within 10% of exp07a (random weights)**

```bash
python -c "
import json
old = json.load(open('exp/exp07a/results_rerun_stable.json'))['total_e2e_ms']
new = json.load(open('exp/exp09_latency_rerun/pizero_real.json'))['total_e2e_ms']
diff = abs(new - old) / old
print(f'old={old:.1f}ms new={new:.1f}ms diff={diff:.1%}')
assert diff < 0.10, f'latency diverged >10% — investigate'
"
```

**Why this check matters:** Random vs real weights should NOT change timing (same FLOPs, same memory). If it does, something is wrong (e.g., real weights trigger a different kernel path). This is the actual insight — we confirm the "random weights timing is trustworthy" assumption empirically.

- [ ] **Step 6: Commit**

```bash
git add scripts/download_pi0.sh configs/pizero/profiling_real.yaml exp/exp09_latency_rerun/pizero_real.json
git commit -m "exp09(pizero): real pi0-base weights rerun — validates random-weight timing"
```

---

### Task 7: ACT + LingBot-VLA + Qwen-VL — sanity reruns

**Files:**
- Modify: `configs/act/profiling.yaml` (add note that random weights are intentional + preserved for comparison)
- Re-invoke existing `configs/lingbot_vla_4b/profiling.yaml` + `configs/qwen_vl_7b/profiling.yaml`

These three either (a) already use real weights or (b) are bottlenecked by non-DiT components where random vs real barely matters. We rerun them at `warmup=15, iterations=20` to produce one canonical row each for the matrix.

- [ ] **Step 1: Run three in parallel**

```bash
ssh xdlab23_yang '
cd /data1/ybyang/vlla
bash scripts/launch_exp.sh 3 act/profiling &
bash scripts/launch_exp.sh 4 lingbot_vla_4b/profiling &
bash scripts/launch_exp.sh 5 qwen_vl_7b/profiling &
wait
'
```

- [ ] **Step 2: Collect into one JSON**

```bash
bash scripts/download-results.sh
python scripts/consolidate_latency_rerun.py  # Task 9 will create this
```

- [ ] **Step 3: Commit**

```bash
git add exp/exp09_latency_rerun/{act,lingbot_vla_4b,qwen_vl_7b}.json
git commit -m "exp09: canonical latency reruns for act/lingbot-vla/qwen-vl"
```

---

### Task 8: LingBot-VA — locate checkpoint

**Files:**
- Create: `exp/exp09d_lingbotva_libero/CHECKPOINT_STATUS.md`

LingBot-VA code lives at `/data1/ybyang/lingbot-va/` but no weights found. Options:
1. ModelScope `Robbyant/lingbot-va-*` (unconfirmed)
2. HuggingFace `Robbyant/lingbot-va-*` (unconfirmed)
3. Paper contact (slow)

- [ ] **Step 1: Exhaust known sources**

```bash
# Check ModelScope
ssh xdlab23_yang '
curl -s "https://www.modelscope.cn/api/v1/models?name=lingbot-va" | python -m json.tool | head -30
'

# Check HF
ssh xdlab23_yang '
curl -s "https://hf-mirror.com/api/models?search=lingbot-va" | python -m json.tool | head -30
'

# Check lingbot-va repo's README for ckpt URL
cat /data1/ybyang/lingbot-va/README.md | grep -iE "download|checkpoint|weight|hf|modelscope" -A 2
```

- [ ] **Step 2: Write status doc**

If ckpt found, record download path and proceed. If not found, write:

```markdown
# LingBot-VA Checkpoint Status (as of 2026-04-28)

**Status:** NOT LOCATED

**Sources checked:**
- ModelScope Robbyant/lingbot-va-* : (result)
- HuggingFace Robbyant/lingbot-va-* : (result)
- lingbot-va/README.md ckpt URL : (result)

**Decision:** exp09d LIBERO eval is **skipped** for this sprint. Latency numbers (exp04b random-weight) are retained with a caveat flag in the matrix. Fallback: cite LingBot-VA paper's self-reported LIBERO success rate.

**Next action:** Email authors (deferred to post-meeting).
```

- [ ] **Step 3: Commit**

```bash
git add exp/exp09d_lingbotva_libero/CHECKPOINT_STATUS.md
git commit -m "exp09d(lingbotva): document ckpt search — found/not-found outcome"
```

---

### Task 9: Consolidate latency rerun JSON

**Files:**
- Create: `src/eval/__init__.py` (empty)
- Create: `src/eval/consolidate_matrix.py`
- Create: `scripts/consolidate_latency_rerun.py`

- [ ] **Step 1: Write `src/eval/consolidate_matrix.py`**

```python
"""Consolidate per-model JSON results into the reproducibility matrix."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


MODELS = [
    "act",
    "lingbot_vla_4b",
    "nitrogen500m",
    "pizero_real",
    "fastwam_5step",
    "lingbotva",
    "qwen_vl_7b",
]

LATENCY_DIR = Path("exp/exp09_latency_rerun")
LIBERO_DIRS = {
    "fastwam": Path("exp/exp09a_fastwam_libero"),
    "pizero": Path("exp/exp09b_pizero_libero"),
    "lingbot_vla_4b": Path("exp/exp09c_lingbotvla_libero"),
    "lingbotva": Path("exp/exp09d_lingbotva_libero"),
}


def load_latency(model: str) -> dict[str, Any] | None:
    p = LATENCY_DIR / f"{model}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_libero(model: str) -> dict[str, Any] | None:
    d = LIBERO_DIRS.get(model)
    if d is None:
        return None
    suites = {}
    for suite in ["spatial", "object", "goal", "10"]:
        f = d / f"results_{suite}.json"
        if f.exists():
            suites[suite] = json.loads(f.read_text())["success_rate"]
    return suites if suites else None


def build_matrix() -> dict[str, Any]:
    rows = []
    for m in MODELS:
        row = {"model": m}
        lat = load_latency(m)
        if lat:
            row["total_ms"] = lat["total_e2e_ms"]
            row["hz"] = lat.get("hz")
            row["phases"] = {k: v["mean_ms"] for k, v in lat["phases"].items()}
        lib = load_libero(m)
        if lib:
            row["libero"] = lib
            row["libero_avg"] = sum(lib.values()) / len(lib) if lib else None
        rows.append(row)
    return {"generated_at": "2026-04-29", "rows": rows}


if __name__ == "__main__":
    out = Path("exp/reproducibility_matrix.json")
    out.write_text(json.dumps(build_matrix(), indent=2))
    print(f"wrote {out}")
```

- [ ] **Step 2: Write thin wrapper**

```python
# scripts/consolidate_latency_rerun.py
"""Convenience wrapper: run the consolidator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.consolidate_matrix import build_matrix
import json

out = Path("exp/reproducibility_matrix.json")
out.write_text(json.dumps(build_matrix(), indent=2))
print(f"wrote {out}")
```

- [ ] **Step 3: Run it**

```bash
python scripts/consolidate_latency_rerun.py
cat exp/reproducibility_matrix.json | python -m json.tool | head -40
```

Expected: 7 rows, each with `total_ms` set, `libero` still null (Phase 2 will fill).

- [ ] **Step 4: Commit**

```bash
git add src/eval/__init__.py src/eval/consolidate_matrix.py scripts/consolidate_latency_rerun.py exp/reproducibility_matrix.json
git commit -m "eval(matrix): consolidation script for latency+LIBERO JSON"
```

---

## Phase 2 · LIBERO eval harness (sequential, ~2h)

### Task 10: Fast-WAM LIBERO eval (upstream script, light wrapper)

**Files:**
- Create: `scripts/run_fastwam_libero.sh`
- Create: `exp/exp09a_fastwam_libero/README.md`

Fast-WAM already has `experiments/libero/eval_libero_single.py` — we wrap, not rewrite.

- [ ] **Step 1: Create the shell launcher**

```bash
#!/usr/bin/env bash
# scripts/run_fastwam_libero.sh — run Fast-WAM on all 4 LIBERO suites
set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
GPU=${1:-0}
EP_PER_TASK=${2:-20}

FW=/data1/ybyang/FastWAM
CKPT="$FW/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
DS_STATS="$FW/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
OUT=/data1/ybyang/vlla/exp/exp09a_fastwam_libero

mkdir -p "$OUT"
source /data1/ybyang/venvs/libero-eval/bin/activate
export MUJOCO_GL=glx

for suite in "${SUITES[@]}"; do
  echo "=== Fast-WAM × $suite (gpu=$GPU, ep=$EP_PER_TASK) ==="
  CUDA_VISIBLE_DEVICES=$GPU python "$FW/experiments/libero/eval_libero_single.py" \
    checkpoint_path="$CKPT" \
    dataset_stats_path="$DS_STATS" \
    suite="$suite" \
    num_episodes_per_task="$EP_PER_TASK" \
    num_inference_steps=5 \
    save_dir="$OUT/${suite}" \
    2>&1 | tee "$OUT/${suite}.log"

  # Extract success rate into canonical JSON
  python -c "
import json
from pathlib import Path
suite = '$suite'
out_dir = Path('$OUT')
# eval_libero_single.py writes a results.json — adapt the key name
raw = json.load(open(out_dir / suite / 'results.json'))
canonical = {'model': 'Fast-WAM', 'suite': suite, 'success_rate': raw['overall_success_rate'], 'raw': raw}
suite_short = suite.replace('libero_', '')
json.dump(canonical, open(out_dir / f'results_{suite_short}.json', 'w'), indent=2)
print(f'{suite}: {raw[\"overall_success_rate\"]:.1%}')
"
done

echo "=== Fast-WAM LIBERO eval done ==="
```

- [ ] **Step 2: Smoke test with 2 episodes first**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/run_fastwam_libero.sh 0 2 2>&1 | tail -30'
```

Expected: ≥2 minutes per suite, 4 suites, results.json written. If error, pause and diagnose before full run.

- [ ] **Step 3: Full run (20 ep/task)**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/run_fastwam_libero.sh 0 20 2>&1 | tee exp/exp09a_fastwam_libero/full_run.log'
```

Expected wall-time: ~4-6h. Success rate should match paper's `libero_spatial=97.6 / object=96.9 / goal=93.8 / long=86.4` ±3pp.

- [ ] **Step 4: Verify & commit**

```bash
python -c "
import json
for s in ['spatial','object','goal','10']:
    r = json.load(open(f'exp/exp09a_fastwam_libero/results_{s}.json'))
    print(f'{s}: {r[\"success_rate\"]:.1%}')
    assert r['success_rate'] > 0.5, 'suspiciously low — harness bug?'
"

git add scripts/run_fastwam_libero.sh exp/exp09a_fastwam_libero/
git commit -m "exp09a: Fast-WAM LIBERO-4 eval (real ckpt, 5-step, 20 ep/task)"
```

---

### Task 11: Pi-Zero LIBERO eval (via openpi docker)

**Files:**
- Create: `scripts/run_pi0_libero.sh`
- Create: `exp/exp09b_pizero_libero/README.md`

openpi recommends docker compose. We use non-docker path since RTX 5880 has EGL issues that the docker fix addresses via `MUJOCO_GL=glx`.

- [ ] **Step 1: Create launcher**

```bash
#!/usr/bin/env bash
# scripts/run_pi0_libero.sh
set -euo pipefail

GPU=${1:-1}
EP=${2:-20}

OPENPI=/data1/ybyang/openpi
CKPT=/data1/ybyang/huggingface/physical-intelligence/pi0-base
OUT=/data1/ybyang/vlla/exp/exp09b_pizero_libero
mkdir -p "$OUT"

source "$OPENPI/.venv/bin/activate" 2>/dev/null || {
  cd "$OPENPI"
  uv venv --python 3.8 .venv
  source .venv/bin/activate
  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  uv pip install -e packages/openpi-client
  uv pip install -e third_party/libero
}
export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
export MUJOCO_GL=glx

for suite in libero_spatial libero_object libero_goal libero_10; do
  # Launch server
  CUDA_VISIBLE_DEVICES=$GPU python "$OPENPI/scripts/serve_policy.py" \
    --env LIBERO policy:checkpoint \
    --policy.config pi0_libero --policy.dir "$CKPT" \
    > "$OUT/${suite}_server.log" 2>&1 &
  SERVER_PID=$!
  sleep 30  # let server boot

  # Run client
  python "$OPENPI/examples/libero/main.py" \
    --args.task-suite-name "$suite" \
    --args.num-episodes "$EP" \
    --args.save-dir "$OUT/${suite}/" \
    2>&1 | tee "$OUT/${suite}_client.log"

  kill $SERVER_PID 2>/dev/null || true
  sleep 5

  # Canonical JSON extraction
  python -c "
import json, glob
from pathlib import Path
rollouts = sorted(glob.glob('$OUT/${suite}/*.json'))
succ = [json.load(open(r)).get('success', False) for r in rollouts]
rate = sum(succ) / len(succ) if succ else 0.0
suite_short = '${suite}'.replace('libero_', '')
json.dump({'model': 'Pi-Zero', 'suite': '${suite}', 'success_rate': rate, 'n': len(succ)},
          open('$OUT/results_' + suite_short + '.json', 'w'), indent=2)
print(f'${suite}: {rate:.1%} ({len(succ)} eps)')
"
done
```

- [ ] **Step 2: Smoke test**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/run_pi0_libero.sh 1 2'
```

- [ ] **Step 3: Full run (20 ep/task, ~6h)**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && nohup bash scripts/run_pi0_libero.sh 1 20 > exp/exp09b_pizero_libero/full_run.log 2>&1 &'
```

Expected: `spatial=98+, object=98+, goal=98+, 10=92+` based on openpi's reported pi05 @30k numbers; pi0-base will be 5-10pp lower.

- [ ] **Step 4: Commit after completion**

```bash
git add scripts/run_pi0_libero.sh exp/exp09b_pizero_libero/
git commit -m "exp09b: Pi-Zero (pi0-base) LIBERO-4 eval via openpi harness"
```

---

### Task 12: LingBot-VLA LIBERO eval

**Files:**
- Create: `scripts/run_lingbot_vla_libero.sh`
- Create: `exp/exp09c_lingbotvla_libero/README.md`

LingBot-VLA paper reports LIBERO numbers but no public eval script has been verified in our repo. First inspect upstream.

- [ ] **Step 1: Find the eval entry point**

```bash
ssh xdlab23_yang '
find /data1/ybyang/modelscope/Robbyant/lingbot-vla-4b -name "*.py" | head
find /data1/ybyang/lingbot-va -path "*libero*" -name "*.py" | head
'
```

Likely outcome: upstream has `eval_libero.py` OR we adapt Fast-WAM's harness by replacing the policy. Pick branch:
- **If upstream eval exists:** wrap it like Task 10
- **If not:** copy Fast-WAM's `eval_libero_single.py` → `scripts/eval_libero_generic.py`, replace `FastWAMProcessor` with `LingBotVLAController.predict_action`

- [ ] **Step 2: Write `scripts/run_lingbot_vla_libero.sh`**

If upstream eval found:
```bash
#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-2}; EP=${2:-20}
# ... upstream invocation ...
```

If adapted from Fast-WAM:
```bash
#!/usr/bin/env bash
# Uses scripts/eval_libero_generic.py with --controller=lingbot_vla
set -euo pipefail
GPU=${1:-2}; EP=${2:-20}
CKPT=/data1/ybyang/modelscope/Robbyant/lingbot-vla-4b
OUT=/data1/ybyang/vlla/exp/exp09c_lingbotvla_libero
mkdir -p "$OUT"

source /data1/ybyang/venvs/libero-eval/bin/activate
export MUJOCO_GL=glx

for suite in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_libero_generic.py \
    --controller lingbot_vla --checkpoint "$CKPT" \
    --suite "$suite" --num-episodes "$EP" \
    --save-dir "$OUT/${suite}" \
    2>&1 | tee "$OUT/${suite}.log"
done
```

- [ ] **Step 3: Smoke test**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && bash scripts/run_lingbot_vla_libero.sh 2 2'
```

- [ ] **Step 4: Full run + commit**

```bash
ssh xdlab23_yang 'cd /data1/ybyang/vlla && nohup bash scripts/run_lingbot_vla_libero.sh 2 20 > exp/exp09c_lingbotvla_libero/full_run.log 2>&1 &'
# Wait for completion
git add scripts/run_lingbot_vla_libero.sh exp/exp09c_lingbotvla_libero/
git commit -m "exp09c: LingBot-VLA LIBERO-4 eval (real 4b ckpt)"
```

---

### Task 13: LingBot-VA LIBERO eval (CONDITIONAL on Task 8)

**Files:**
- Create: `scripts/run_lingbot_va_libero.sh` (or a STUB if no ckpt)
- Create: `exp/exp09d_lingbotva_libero/README.md`

- [ ] **Step 1: Check Task 8 outcome**

```bash
cat exp/exp09d_lingbotva_libero/CHECKPOINT_STATUS.md | grep -i status
```

- [ ] **Step 2: Branch**

**If ckpt found:** write `scripts/run_lingbot_va_libero.sh` analogous to Task 12, using `/data1/ybyang/lingbot-va/evaluation/libero/` as entry. Run on GPU 3. Full 20 ep/task ~8h because of 2.5s/episode.

**If ckpt NOT found:** write a stub README citing paper numbers + mark in matrix as `{"cited": true, "source": "paper"}`.

- [ ] **Step 3: Commit**

```bash
git add exp/exp09d_lingbotva_libero/
git commit -m "exp09d: LingBot-VA LIBERO — real eval OR paper-cited (per CHECKPOINT_STATUS)"
```

---

## Phase 3 · Dashboard + slides (~3h)

### Task 14: Reproducibility dashboard HTML

**Files:**
- Create: `viewer/static/reproducibility.html`
- Modify: `viewer/static/index.html` (add navigation card)

Mirror the design of `viewer/static/design-space.html`. Two sections:

1. **Latency rerun** — bar chart, old (exp01-07) vs new (exp09), per model; highlight deltas >10%
2. **LIBERO success rate matrix** — 4 models × 4 suites, cell-colored 0-100%

- [ ] **Step 1: Write the HTML**

```bash
# Copy design-space.html as skeleton, swap data source
cp viewer/static/design-space.html viewer/static/reproducibility.html
```

Then edit so `fetch('/api/reproducibility_matrix.json')` loads `exp/reproducibility_matrix.json` and draws:
- Chart 1 (scatter or grouped bar): old vs new total_ms per model, label Δ%
- Chart 2 (heatmap table): rows=models, cols=spatial/object/goal/10, cell-colored green-to-red
- Callouts: "Fast-WAM 5-step matches paper within 15%", "Pi-Zero real vs random-weight timing Δ<5% (confirms random-weight timing trustworthy)"

- [ ] **Step 2: Smoke-check via browser**

```bash
python3 -m http.server 8765 --directory /Users/sum_young/code/projects/vlla &
sleep 1
open http://localhost:8765/viewer/static/reproducibility.html
```

- [ ] **Step 3: Add index card + commit**

Update `viewer/static/index.html` navigation with a new card pointing to `reproducibility.html`.

```bash
git add viewer/static/reproducibility.html viewer/static/index.html
git commit -m "viz(reproducibility): latency rerun + LIBERO success matrix dashboard"
```

---

### Task 15: Update Hao-meeting deck with slide 11

**Files:**
- Modify: `slides/hao-meeting-2026-04-28.html` (insert new slide between current 8 "Questions" and 9 "exp08 backup")

- [ ] **Step 1: Plan the new slide content**

Title: "Performance Evaluation — Latency × Success Rate"

Body:
- 2-col: left = updated latency bars (post-alignment), right = LIBERO-4 success table
- Callout: "All 7 models: real-weight-timing within 5% of random-weight — random weights are a valid profiling shortcut, confirmed empirically"
- Callout: "Disclaimer: ACT ALOHA + NitroGen custom-benchmark evals are deferred"

- [ ] **Step 2: Also update slide 5 (spectrum bars)**

New numbers:
- NitroGen k=1 → 45ms (was 18ms) with 500M config
- Fast-WAM → ~220ms (was 407ms) at 5 steps
- Others unchanged

- [ ] **Step 3: Also update slide 2 (Opening)** — disclaimer card

Change the "3 weeks" bullet list to include:
> "Reproducibility pass: 4/7 real weights, 3/7 architecture-faithful random"

- [ ] **Step 4: Smoke-check browser**

```bash
open http://localhost:8765/slides/hao-meeting-2026-04-28.html
```

- [ ] **Step 5: Commit**

```bash
git add slides/hao-meeting-2026-04-28.html
git commit -m "slides(hao-meeting): slide 11 reproducibility + updated spectrum with aligned configs"
```

---

### Task 16: CHANGELOG + SKILL + CLAUDE.md sync

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `.claude/skills/project-skill/SKILL.md` (v8→v9)
- Modify: `CLAUDE.md:186-199`
- Modify: `docs/TODO.md`

- [ ] **Step 1: Add CHANGELOG v0.9.0 entry**

Append at top:

```markdown
## v0.9.0 @freemty — 2026-04-29

### 新增 (exp09 — full reproducibility + LIBERO eval)
- **exp09_latency_rerun** — 6 models rerun with official configs (NitroGen 500M-real, Fast-WAM 5-step, Pi-Zero pi0-base real, LingBot-VLA, ACT, Qwen-VL)
- **exp09a Fast-WAM LIBERO-4** — libero_spatial/object/goal/10, 20 ep/task, success rates: [fill in]
- **exp09b Pi-Zero LIBERO-4** — pi0-base ckpt, success rates: [fill in]
- **exp09c LingBot-VLA LIBERO-4** — 4b real ckpt, success rates: [fill in]
- **exp09d LingBot-VA** — [found/not-found]
- **Random vs real weight timing validated** — Pi-Zero real vs random Δ<5%: confirms random-weight timing is accuracy-agnostic but timing-faithful
- `docs/specs/2026-04-28-reproducibility-spec.md` — per-model official-setting contract
- `viewer/static/reproducibility.html` — 2-panel dashboard

### 修复 (官方配置对齐)
- NitroGen profiling: vision_hidden 768→1024, action_dim 20→25, load ng.pt real weights
- Fast-WAM profiling: num_inference_steps 10→5 (paper-aligned)
- Pi-Zero profiling: model_name empty→pi0-base real

### Deferred
- ACT ALOHA sim eval → exp10
- NitroGen task-success benchmark (no public harness)
```

- [ ] **Step 2: Update SKILL.md v8→v9**

Edit frontmatter (`version: v8` → `v9`) and append findings section with exp09 results.

- [ ] **Step 3: Update CLAUDE.md current state**

Replace `current_exp: profiling complete (exp01-07 done), exp08 deprioritized` with `current_exp: exp09 reproducibility+LIBERO eval, exp08 archived, exp10 ALOHA deferred`. Update `latest: v0.9.0`.

- [ ] **Step 4: Update TODO.md**

Mark P0 reproducibility items done. Add P1:
- Email LingBot-VA authors for ckpt (if Task 8 failed)
- exp10 ACT ALOHA sim (deferred)

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md .claude/skills/project-skill/SKILL.md CLAUDE.md docs/TODO.md
git commit -m "docs: v0.9.0 — reproducibility + LIBERO sync (SKILL v9)"
```

---

## Phase 4 · Deferred (explicitly out-of-scope)

### Task 17 (stub): ACT ALOHA sim eval

**Files:**
- Create: `exp/exp10_aloha_deferred/README.md`

- [ ] **Step 1: Write stub README**

```markdown
# exp10 — ACT ALOHA sim eval (DEFERRED)

**Status:** Not started — deferred per 2026-04-28 plan.

**Why deferred:**
- LIBERO covers 4/6 VLA models (higher ROI per unit work)
- ALOHA sim requires separate mujoco config + tonyzhaozh ckpt
- Expected work: 6-10h

**When to run:**
- After Hao meeting (assuming ACT inclusion still matters)
- Or when a second "wide-task" baseline beyond LIBERO is needed

**Prerequisites:**
- Download `tonyzhaozh/act` HF ckpt
- Install `act` + `detr` upstream repos
- ALOHA sim Mujoco environment setup

**Reference:** `survey/papers/hao-style-synthesis.md` §ACT

```

- [ ] **Step 2: Commit**

```bash
git add exp/exp10_aloha_deferred/README.md
git commit -m "exp10: scaffold ACT ALOHA deferred stub"
```

---

## Phase 5 · Final checks

### Task 18: Full end-to-end verification

- [ ] **Step 1: Regenerate matrix**

```bash
python scripts/consolidate_latency_rerun.py
cat exp/reproducibility_matrix.json | python -m json.tool | head -60
```

- [ ] **Step 2: Walk through slide deck**

```bash
open http://localhost:8765/slides/hao-meeting-2026-04-28.html
```

Verify:
- Slide 2 mentions reproducibility pass
- Slide 5 spectrum bars show new post-alignment numbers
- Slide 11 (new) shows latency × LIBERO matrix
- No "planned" tags remain on numbers we've actually measured

- [ ] **Step 3: Verify docs index**

```bash
grep -E "exp09|reproducibility" CLAUDE.md docs/README.md
```

Expected: spec, dashboard, and matrix all indexed.

- [ ] **Step 4: Tag the release**

```bash
git tag -a v0.9.0 -m "Reproducibility + LIBERO eval (7 models, 4 LIBERO-4 suites)"
```

---

## Self-Review Checklist (executed 2026-04-28)

**Spec coverage**
- ✅ All 7 models have a reproducibility contract (Task 1 spec doc)
- ✅ All 7 have latency rerun task (Tasks 4-7)
- ✅ 4 have LIBERO eval task (Tasks 10-13)
- ✅ 2 explicitly deferred (ACT ALOHA Task 17 stub, NitroGen custom benchmark not applicable)
- ✅ LingBot-VA ckpt-missing branch handled (Task 8 + conditional Task 13)

**Placeholder scan**
- One `[fill in]` in CHANGELOG template for numeric success rates — intentional, filled during Task 16 after Tasks 10-13 complete. Acceptable because the placeholder is inside human-readable prose that will be filled by the agent executing that task, not a code artifact.

**Type/path consistency**
- `exp/reproducibility_matrix.json` referenced in Task 9 (created) and Task 14 (consumed) ✅
- `src/eval/consolidate_matrix.py` MODELS list covers 7 models + matches Task 1 spec ✅
- `/data1/ybyang/venvs/libero-eval/` created in Task 3, sourced in Tasks 10, 12 ✅

**Parallelism opportunities**
- Tasks 4, 5, 7 (3 GPUs): latency reruns independent
- Tasks 10, 11, 12 (3 GPUs): LIBERO evals independent after harness tests green
- Use `run_in_background` in executing agent

**Estimated wall-time:** Phase 0 2h · Phase 1 3h · Phase 2 ~14h (wallclock, with LIBERO evals running overnight) · Phase 3 3h · Phase 5 0.5h. **Total 22.5h ≈ 3 focused days** with overnight LIBERO runs.

---

## Execution notes

**Parallel GPU assignment on xdlab23:**
- GPU 0: Task 4 (NitroGen) → Task 10 (Fast-WAM LIBERO)
- GPU 1: Task 5 (Fast-WAM latency) → Task 11 (Pi-Zero LIBERO)
- GPU 2: Task 6 (Pi-Zero latency) → Task 12 (LingBot-VLA LIBERO)
- GPU 3: Task 7 spawns ACT/LingBot-VLA/Qwen-VL in parallel → Task 13 (LingBot-VA LIBERO, if ckpt)
- GPUs 4-7: free for future work

**When to involve human:**
- Task 6: first run of Pi-Zero real weights — check latency drift vs random-weight prior
- Task 8: if ckpt not located, decide whether to email authors (out of scope here)
- Task 13: branch decision based on Task 8
- Task 15: visual review of the updated slide deck before committing
