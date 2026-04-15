# Experiment Summary

Cross-experiment flight recorder. One row per experiment.

| Exp ID | Motivation | Status | Key Finding |
|--------|-----------|--------|-------------|
| exp01a | Qwen2.5-VL-7B E/P/D profiling on RTX 5880 Ada | **done** | Vision Encode = 534ms (60%), Prefill = 354ms (40%), Decode = 19ms (2%). **Encode is the dominant bottleneck**, not prefill. |
