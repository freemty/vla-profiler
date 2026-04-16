---
name: StarVLA Framework Summary
description: StarVLA (arXiv:2604.05014) -- modular VLA dev platform unifying 4 architectures; engineering framework not algorithmic; key bridge to vlla profiling
type: reference
---

**StarVLA** (2026, arXiv:2604.05014, github.com/starVLA/starVLA)

- Modular open-source VLA framework by community (Jinhui Ye, Weiyu Guo co-founders)
- Unifies 4 VLA architectures: FAST (AR discrete), OFT (MLP parallel), PI (Flow-Matching), GR00T (dual-system VLM+Flow)
- VLM backbones: Qwen2.5-VL-7B, Qwen3.5 (0.8B-9B), Qwen3-VL-4B, InternVL, Florence-2
- WM4A feature: Cosmos-Predict2/Wan2.2 DiT as action backbone (World Action Model integration)
- Training: SFT, multimodal co-train, cross-embodiment co-train, RL post-training (RLinf)
- Benchmarks: LIBERO (claims SOTA), SimplerEnv, RoboCasa, RoboTwin, BEHAVIOR, Calvin

**Relationship to our work:**
- Engineering framework (like HF Transformers), NOT systems/inference work
- Zero inference profiling or serving capability -- exact gap our vlla fills
- Potential model zoo for vlla: one StarVLAController adapter could cover all 4 frameworks
- WM4A connects to FastVideo tech stack -- profiling WM4A bottleneck argues for FastVideo->VLA transfer
- Four frameworks have completely different serving profiles -- ideal testbed for VLA serving research

**Deep dive saved at:** `docs/papers/starvla-framework-deep-dive.md`
