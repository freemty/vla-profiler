# exp08a — EPDA pair-wise interference pilot

## Motivation

Test two roofline-derived predictions (see `docs/specs/2026-04-26-epda-roofline-analysis.md` §5):

| Pair | Roofline prediction | Hypothesis |
|------|---------------------|------------|
| **D + A** | **STRONG** interference (>30% inflation) | HBM BW saturation (D=73% peak) + kernel-dispatch amplification collide |
| **P + A** | **WEAK** interference (<10% inflation) | Tensor core (P) and kernel dispatch (A) use orthogonal resources, may even help fill bubbles |

If both predictions verify → roofline co-design motivation is empirically grounded (strong evidence for EPDA disaggregation). If they diverge → roofline model needs refinement.

## Design (strategy α, no barrier)

- **2 Python threads, 2 CUDA streams**: one runs LLM phase, one runs NitroGen DiT action loop
- Per-thread warmup (15) then sync barrier release → simultaneous measurement of N=40 iterations
- Per-iteration latency via CUDA events scoped to each stream
- **No cross-thread barrier mid-loop** — realistic serving (two independent requests)

## Payloads

| Phase | Model | Config |
|-------|-------|--------|
| D | Qwen2.5-VL-7B (decode, 1 token at a time, KV-cache) | bf16, sdpa |
| P | Qwen2.5-VL-3B (prefill, ~64 text tokens) | bf16, sdpa |
| A | NitroGen 174M DiT (k=10 denoise steps) | bf16 |

Co-location combines 7B (D) or 3B (P) with 174M (A) — aggregate VRAM well under 48GB.

## Metric

```
inflation_ratio(phase) = median(phase, coloc) / median(phase, alone)
```

Inflation > 1.2 = strong interference. < 1.05 = orthogonal.

## Status

- [x] Script ready (`scripts/exp08a_interference.py`, CPU smoke test passed 2026-04-27)
- [ ] D+A pair run on xdlab23
- [ ] P+A pair run on xdlab23
- [ ] Inflation comparison figure
- [ ] README update with findings + Hao-meeting-ready one-liner

## Run

```bash
# On xdlab23, vit-probe env
cd /data1/ybyang/vlla
/home/ybyang/miniconda3/envs/vit-probe/bin/python scripts/exp08a_interference.py \
    --pair DA --gpu 0 --warmup 15 --iterations 40 \
    --output exp/exp08a/results_DA.json

/home/ybyang/miniconda3/envs/vit-probe/bin/python scripts/exp08a_interference.py \
    --pair PA --gpu 0 --warmup 15 --iterations 40 \
    --output exp/exp08a/results_PA.json
```

## Predictions (record before running)

| Pair | Phase | Alone (ms, from prior exp) | Coloc predicted | Inflation predicted |
|------|-------|----------------------------|-----------------|---------------------|
| D+A | D | 18 (exp01a single_img) | ~25-30 | **1.4-1.7x** |
| D+A | A | 72 (exp06a k=10) | ~90-120 | **1.3-1.6x** |
| P+A | P | ~40 (exp03a C) | ~42-45 | **1.0-1.1x** |
| P+A | A | 72 | ~75-80 | **1.0-1.1x** |

## References

- Spec: `docs/specs/2026-04-26-epda-disaggregation-spec.md`
- Roofline: `docs/specs/2026-04-26-epda-roofline-analysis.md`
- Prior baselines: exp01a (D), exp03a (P proxy via LingBot-VLA C), exp06a (A)
