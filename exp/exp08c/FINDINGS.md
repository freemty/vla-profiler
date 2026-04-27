# exp08c Findings — GPU Contention Model

## Model Comparison (12 data points, 6 pairs from exp08b)

| Model | R² | Description |
|-------|-----|-------------|
| M1 Additive | 0.21 | Symmetric resource product model — fails because interference is asymmetric |
| M2 Bottleneck | -3.44 | Max-resource saturation model — predicts same inflation for both sides of a pair |
| **M4 Asymmetric** | **0.94** | vulnerability × aggressiveness — captures that D/P are fragile while A/E are robust |
| M3 Empirical | 1.00 | Lookup table (no generalization, R² upper bound) |

## M4 Learned Parameters

| Phase | Vulnerability (v) | Aggressiveness (a) | Interpretation |
|-------|-------------------|-------------------|----------------|
| D (decode) | **1.52** | 0.87 | Very fragile, moderately disruptive |
| P (prefill) | **1.61** | 1.02 | Very fragile, moderately disruptive |
| E (encode) | 0.23 | 0.96 | Robust, moderately disruptive |
| A (action DiT) | 0.20 | **1.12** | Very robust, most disruptive |

### Formula
```
inflation(X|Y) = 1 + v_X * a_Y
```

### Key Insights

1. **Asymmetry is the dominant signal.** D/P vulnerability is 7-8× that of E/A. Symmetric models (M1/M2) fundamentally cannot fit this data.

2. **A is a stealth disruptor.** DiT action denoising barely slows down under contention (v=0.20) but is the most aggressive co-runner (a=1.12). Likely due to bursty kernel dispatch that disrupts SM scheduling for memory-bound phases.

3. **E is the safest co-locate candidate.** Low vulnerability (0.23) AND low-to-moderate aggressiveness (0.96). The EP pair is the only outlier where E experiences notable inflation (1.70x vs predicted 1.24x) — likely tensor-core contention specific to E+P.

4. **D and P form a "fragile pair".** Both highly vulnerable, and mutually disruptive (PD: P=2.42x, D=2.48x). This is the DistServe phenomenon — PD disaggregation is validated.

## Prediction Accuracy

| Combo | Phase | Observed | M4 Predicted | Error |
|-------|-------|----------|-------------|-------|
| DA | D | 2.654x | 2.700x | +0.05 |
| DA | A | 1.162x | 1.177x | +0.01 |
| EA | E | 0.971x | 1.261x | **+0.29** |
| EA | A | 1.226x | 1.196x | -0.03 |
| ED | E | 1.035x | 1.202x | +0.17 |
| ED | D | 2.586x | 2.460x | -0.13 |
| EP | E | 1.698x | 1.237x | **-0.46** |
| EP | P | 2.420x | 2.543x | +0.12 |
| PA | P | 2.881x | 2.796x | -0.09 |
| PA | A | 1.192x | 1.208x | +0.02 |
| PD | P | 2.415x | 2.389x | -0.03 |
| PD | D | 2.476x | 2.545x | +0.07 |

Mean absolute error: 0.12x. Largest error on EP(E): model underestimates E vulnerability when co-running with P specifically — suggests compute-specific interaction term needed for M5.

## Implications for EPDA Disaggregation

1. **Safe to co-locate**: E+A (both robust, low mutual interference)
2. **Must disaggregate**: P+D (DistServe), P+A, D+A (LLM phases too fragile)
3. **Optimal split**: {E, A} on GPU1, {P} on GPU2, {D} on GPU3 — or {E,A} + {P,D disaggregated}
4. **M4 can predict triple/quad inflation** from pair data — pending validation with exp08b triple combos
