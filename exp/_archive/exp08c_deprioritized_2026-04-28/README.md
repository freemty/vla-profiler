# exp08c — GPU Kernel-Level Contention Model

> 拟合 exp08a/b 干扰数据，建立 GPU 资源争用预测模型

## Goal
回答："给定两个 EPDA 阶段共置，能否不跑实验就预测延迟膨胀？"
Roofline 严重低估（2-28x），需要 kernel-level model 补位。

## Three Candidate Models

### M1: Additive Resource Contention
```
inflation(X|Y) = 1 + Σ_i α_i * profile_X[i] * profile_Y[i]
```
每个资源维度（compute/BW/dispatch）有一个学习系数 α_i。inflation 与 co-runner 的资源需求线性相关。

### M2: Bottleneck Resource Saturation
```
inflation(X|Y) = max_i(demand_X[i] + demand_Y[i]) * γ_i
```
取最饱和的资源维度为 bottleneck。类似 roofline 但有学习系数。

### M3: Empirical Interaction Matrix
```
β_{X,Y} = observed inflation
```
纯查表，无泛化能力，但作为 R² 上界。

## Resource Profiles (from roofline analysis)
| Phase | Compute | HBM BW | Dispatch |
|-------|---------|--------|----------|
| E | 0.60 | 0.10 | 0.30 |
| P | 0.80 | 0.19 | 0.40 |
| D | 0.05 | 0.73 | 0.20 |
| A | 0.15 | 0.05 | 0.85 |

## Usage
```bash
# Fit from pilot data
python scripts/exp08c_contention_model.py \
    --data-dir exp/exp08a/ --output exp/exp08c/model_pilot.json

# Fit from full matrix
python scripts/exp08c_contention_model.py \
    --data-dir exp/exp08b/ --output exp/exp08c/model_full.json

# Predict new combo
python scripts/exp08c_contention_model.py \
    --model exp/exp08c/model_full.json --predict EPD
```

## Status
- [x] Model architecture design (M1/M2/M3)
- [ ] Pilot fit (exp08a data, 4 points)
- [ ] Full fit (exp08b data, ~24 points)
- [ ] Cross-validation analysis
- [ ] Predict → verify on held-out combos

## Findings
_Pending exp08b data._
