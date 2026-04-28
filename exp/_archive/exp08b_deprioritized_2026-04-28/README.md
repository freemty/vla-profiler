# exp08b — Full EPDA Interference Matrix

> 完整 6×6 pair-wise + 4 triple + 1 quad EPDA 干扰矩阵

## Goal
量化 EPDA 四阶段在同一 GPU 上共置时的 pair-wise 延迟膨胀，产出 DistServe-style interference motivation figure。

## Model Combo
- **E**: Qwen2.5-VL-7B vision encoder (ViT)
- **P**: Qwen2.5-VL-3B prefill (~64 tokens)
- **D**: Qwen2.5-VL-7B decode (with KV cache)
- **A**: NitroGen DiT 174M (k=10 denoise steps)

## Experiment Matrix

### Pairs (6)
| Combo | Phase 1 Resource | Phase 2 Resource | Predicted Interference |
|-------|-----------------|-----------------|----------------------|
| EP | Tensor core (near ridge) | Tensor core (at ridge) | STRONG (compute contention) |
| ED | Tensor core | HBM BW-saturated | MODERATE (orthogonal) |
| EA | Tensor core | Kernel dispatch | WEAK-MODERATE |
| PD | Tensor core (at ridge) | HBM BW-saturated | STRONG (DistServe proven) |
| PA | Tensor core | Kernel dispatch | MODERATE (exp08a: P 2.27x) |
| DA | HBM BW-saturated | Kernel dispatch | STRONG (exp08a: D 3.52x) |

### Triples (4) + Quad (1)
EPD, EPA, EDA, PDA, EPDA

## Setup
```bash
# xdlab23
conda activate vit-probe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Run all 6 pairs (~4h estimated)
bash scripts/launch_exp08b.sh 0

# Run single pair for testing
python scripts/exp08b_interference_matrix.py \
    --combo DA --gpu 0 --warmup 15 --iterations 40 \
    --output exp/exp08b/results_DA.json
```

## Baseline (from exp08a pilot)
| Pair | Phase 1 Inflation | Phase 2 Inflation |
|------|------------------|------------------|
| PA | P: 2.27x | A: 1.15x |
| DA | D: 3.52x | A: 1.58x |

## Status
- [ ] EP pair
- [ ] ED pair
- [ ] EA pair
- [ ] PD pair
- [ ] PA pair (rerun with warmup=15)
- [ ] DA pair (rerun with warmup=15)
- [ ] Triple combos (EPD, EPA, EDA, PDA)
- [ ] Quad combo (EPDA)
- [ ] Interference matrix figure

## Findings
_Pending experiment completion._
