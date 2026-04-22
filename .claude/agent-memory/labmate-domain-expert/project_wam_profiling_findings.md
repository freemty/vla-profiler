---
name: WAM Profiling Key Findings (exp04a/04b)
description: Fast-WAM + LingBot-VA profiling results — action bottleneck thesis, per-step memory-bandwidth bound, step reduction as top optimization
type: project
---

exp04a (Fast-WAM, skip-imagination WAM) and exp04b (LingBot-VA, full WAM) profiled 2026-04-21.

**Key findings:**
1. Action bottleneck is "heavy DiT x many steps", not diffusion/flow itself. LingBot-VLA flow matching 10 steps = 0.48ms; Fast-WAM 30-layer MoT x 10 steps = 362ms.
2. WAM action denoising is memory-bandwidth bound (same as LLM decode). 5B DiT per-step ~28.5ms; theoretical weight-loading floor on RTX 5880 Ada = ~10.4ms (2.7x ratio is reasonable).
3. Step reduction is highest ROI: single-step distillation could bring Fast-WAM from 407ms to ~76ms (13Hz), matching Flow VLA.
4. "Skip-imagination" gives 5x speedup but loads full 5B video expert for KV cache only ("phantom parameter" problem).
5. Per-step cost puzzle: Fast-WAM ActionDiT (350M) = ~32ms/step vs LingBot-VA full DiT (5B) = ~28.5ms/step. Likely caused by MoT cross-attention overhead to video KV cache. Needs per-layer profiling to confirm.

**Data quality notes:**
- exp04a @20step: high quality (CV=0.3%)
- exp04a @10step: medium quality (CV=11.3%, power management suspected)
- exp04b: needs rerun with 20+ iterations (only 10 iter, CV=16%)

**Why:** These findings shape the PhD research direction — WAM inference optimization should focus on step distillation and batching (memory-bandwidth bound regime), not kernel-level attention optimization.

**How to apply:** When analyzing future WAM experiments, compare per-step cost to the ~28-32ms baseline. When advising on optimization, prioritize step reduction over quantization for WAM models.
