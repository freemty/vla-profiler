---
name: WAM/VA Profiling Target Selection
description: Top-3 profiling targets for WAM/VA paradigm — DDP-WM (latent WAM), AnoleVLA (SSM VLA), GigaWorld-Policy (action-centered WAM); selected 2026-04-20
type: project
---

User identified gap in profiling suite: no WAM paradigm models profiled yet. Selected top-3 targets:

1. **DDP-WM** (arXiv:2602.01780) — Latent world model with dynamic decomposition, ~30ms, ~30Hz. Teaches: rollout pipeline breakdown, primary/secondary dynamics compute ratio, GRU+MLP bottleneck pattern (compute-bound, unlike AR decode which is memory-bound).

2. **AnoleVLA** (arXiv:2603.15046) — State Space Model (Mamba/RWKV) backbone VLA, 3x faster than baseline. Teaches: SSM vs Transformer inference characteristics, O(n) vs O(n^2) in practice, KV-cache elimination implications for serving.

3. **GigaWorld-Policy** (arXiv:2603.17240) — Action-centered WAM with optional video generation. Teaches: cost of "imagination" (video-on vs video-off), conditional compute serving challenge, ERA disaggregation validation.

**Why:** These cover all major paradigms missing from exp01-03 (latent WM, SSM VLA, action-centered WAM), all fit on single 48GB GPU, all in interesting 10-200ms range.

**How to apply:** Next step is verifying open-source availability for each, then implementing controllers. Survey doc at `survey/papers/efficient-wam-va-profiling-targets.md`.
