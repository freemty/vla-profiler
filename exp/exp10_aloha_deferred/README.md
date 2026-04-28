# exp10 — ACT ALOHA Sim Eval (DEFERRED)

**Status:** Not started — deferred per 2026-04-28 reproducibility plan.

## Why deferred

- LIBERO covers 4/6 VLA models (higher ROI per unit work)
- ALOHA sim requires separate mujoco config + `tonyzhaozh/act` checkpoint
- Expected work: 6-10h

## When to run

- After Hao meeting (if ACT inclusion still matters for the design space story)
- Or when a second benchmark beyond LIBERO is needed

## Prerequisites

1. Download `tonyzhaozh/act` HuggingFace checkpoint
2. Install `act` + `detr` upstream repos
3. ALOHA sim Mujoco environment (insertion + transfer_cube tasks)

## Reference

- ACT paper: Zhao et al. 2023 "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
- Profiling baseline: exp02a (random weights, 3ms total, 300Hz)
