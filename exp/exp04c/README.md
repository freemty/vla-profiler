# exp04c — Fast-WAM 5-step paper-aligned latency

**Parent:** exp04a (10-step, 407ms)

**What changed:** num_inference_steps 10→5 to match paper. warmup 5→15.

**Key finding:** 257ms total / 3.9Hz (vs paper's 190ms on A100, 1.35x ratio reasonable for RTX 5880). Per-step ~41ms (vs exp04a 36ms — warmup=15 eliminates power-ramp underestimate).

**Config:** `configs/fastwam/profiling_5step.yaml`
**Weights:** Random (Wan2.2-TI2V-5B base downloading for full rerun)
