# Experiment Summary

Cross-experiment flight recorder. One row per experiment.

| Exp ID | Motivation | Status | Key Finding |
|--------|-----------|--------|-------------|
| exp01a | Qwen2.5-VL-7B E/P/D profiling on RTX 5880 Ada (per-input) | **done** | text: E=0/P=20ms/D=18ms/tok. single_img: E=253ms/P=156ms/D=18.6ms/tok. multi_img: E=541ms/P=332ms/D=21ms/tok. **Encode scales linearly with images, decode per-token stable ~18-21ms.** |
| exp01b | Qwen2.5-VL-7B attention analysis (5 layers: 0,7,14,21,27) | **done** | **Pos 2 (first visual patch) is a universal attention sink** — receives 12K-18K attention across all layers (12-28x more than #2). Text→Visual Gini >0.91 (extreme sparsity, supports token pruning). Layer 21 entropy lowest (3.44 vs 4.0-4.2). |
| exp02a | ACT (LeRobot) E/A profiling on RTX 5880 Ada (3 resolutions) | **done** | Total ~3ms (850x faster than VLM). Encode (ResNet18) ~2.5-2.8ms (80%), Action ~0.4-0.8ms. Resolution has minimal impact on encode. VLA latency lower bound baseline. |
| exp03a | LingBot-VLA-4B (Qwen2.5-VL-3B backbone) E/P/D profiling on RTX 5880 Ada | **in progress** | TBD. Measuring VLA overhead vs pure VLM. Compare exp01a (7B) scaling and exp02a (ACT) baseline. |
