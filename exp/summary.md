# Experiment Summary

Cross-experiment flight recorder. One row per experiment.

| Exp ID | Motivation | Status | Key Finding |
|--------|-----------|--------|-------------|
| exp01a | Qwen2.5-VL-7B E/P/D profiling on RTX 5880 Ada (per-input) | **done** | text: E=0/P=20ms/D=18ms/tok. single_img: E=253ms/P=156ms/D=18.6ms/tok. multi_img: E=541ms/P=332ms/D=21ms/tok. **Encode scales linearly with images, decode per-token stable ~18-21ms.** |
