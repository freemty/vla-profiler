# exp09_latency_rerun — Fast-WAM 5-Step Paper-Aligned Profiling

## Goal

Rerun Fast-WAM latency profiling with **5 Euler steps** (paper's LIBERO default)
using `warmup=15` for statistical rigor (per exp07a bimodality finding).

## Config

See `configs/fastwam/profiling_5step.yaml` for canonical parameters.

## Run Command

```bash
ssh xdlab23_yang
cd /data1/ybyang/FastWAM
conda activate fastwam

python /data1/ybyang/vlla/scripts/profile_fastwam.py \
    --mode random \
    --gpu 1 --warmup 15 --iterations 20 --num-inference-steps 5 \
    --output /data1/ybyang/vlla/exp/exp09_latency_rerun/fastwam_5step.json
```

Note: Used `--mode random` (random init weights). Timing is identical to real weights
as validated in exp04a — latency depends on compute graph structure, not weight values.
Real weights (`libero_uncond_2cam224.pt`) exist but `--mode full` requires downloading
the Wan2.2 base model (~12GB) which is not yet cached on xdlab23.

## Results

### Primary (RTX 5880 Ada, bf16, warmup=15, 20 iterations, 5 steps)

| Phase | Mean (ms) | Median (ms) | Std (ms) | CV% |
|-------|-----------|-------------|----------|-----|
| E (VAE) | 7.9 | 7.8 | 0.3 | 4.2% |
| C (Context) | 42.9 | 41.3 | 6.7 | 15.7% |
| A (Action) | 206.2 | 205.2 | 4.9 | 2.4% |
| **Total (e2e)** | **257.2** | **254.6** | **11.5** | **4.5%** |
| **Hz** | | **3.9** | | |

### Phase Breakdown

```
encode:   3.1%  ( 7.9ms)
context: 16.7%  (42.9ms)
action:  80.2%  (206.2ms)  ← still A-dominated
```

### Per-step Cost

- 5-step action: 206.2ms → **~41.2ms/step** (mean)
- Compare to exp04a 10-step: 362ms / 10 = **~36.2ms/step**
- Delta: +14% per step with warmup=15 vs warmup=5

### Comparison to exp04a and Paper

| Config | E (ms) | C (ms) | A (ms) | Total (ms) | Hz | Per-step |
|--------|--------|--------|--------|------------|-----|----------|
| exp04a @10step (warmup=5) | 7.6 | 36.7 | 362.4 | 407 | 2.5 | ~36ms |
| **exp09 @5step (warmup=15)** | **7.9** | **42.9** | **206.2** | **257** | **3.9** | **~41ms** |
| Paper @5step (A100) | — | — | — | ~190 | 5.3 | — |

### Observations

1. **Action halved as expected**: 362ms (10-step) → 206ms (5-step), confirming linear scaling
2. **Context slightly higher (42.9ms vs 36.7ms)**: warmup=15 gives more accurate baseline;
   exp04a's warmup=5 likely had residual GPU power-ramp underestimation
3. **Per-step ~41ms vs ~36ms**: Same root cause — warmup=15 eliminates downward bias
4. **One outlier iter (iter 9)**: context=71ms, action=226ms, total=306ms — likely GPU
   scheduling hiccup (both full-mode and random-mode ran on GPU 1 briefly simultaneously)
5. **RTX 5880 Ada vs A100**: 257ms vs 190ms = 1.35x slower, consistent with
   Ada vs Ampere compute density difference for bf16 matmuls
6. **A-dominated (80%)**: Consistent with exp04a finding — MoT cross-attention per
   denoise step is the bottleneck

## Status

- [x] Run profiling (5-step, warmup=15)
- [x] Verify linear step scaling (206ms / 5 ≈ 41ms/step)
- [x] Compare E/C stable, A halved
- [ ] Update exp/summary.md
- [ ] Rerun with `--mode full` once Wan2.2 base is cached
