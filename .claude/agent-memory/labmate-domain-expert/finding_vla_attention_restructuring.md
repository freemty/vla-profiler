---
name: VLA Attention Restructuring Finding
description: exp05a key finding — VLA fine-tuning completely restructures VLM attention patterns; Gini collapses from 0.91 to 0.07; sink migrates from Pos 2 to boundary token; VLM pruning methods cannot transfer to VLA
type: project
---

exp05a (LingBot-VLA-4B, Qwen2.5-VL-3B backbone, 36 layers) vs exp01b (Qwen2.5-VL-7B, vanilla VLM, 28 layers):

1. **Sink migration:** Pos 2 (first visual patch, 12-28x ratio in VLM) -> Pos 64 (visual-text boundary, 3.7x in L0, disappears by L14+)
2. **Gini collapse:** Text->Visual Gini from >0.91 (extreme sparse) to 0.07-0.45 (near-uniform). VLA reads the ENTIRE visual scene uniformly.
3. **Entropy flattening:** From V-shaped profile (L21=3.44) to flat (4.79-4.90). No information bottleneck layer in VLA.
4. **Core conclusion:** Attention structure is determined by training objective, not architecture. VLM token pruning (FastV, FlashVLM etc.) likely fails on VLA.

**Why:** All 4 pre-experiment predictions were wrong. The assumption that "attention is architecture property" was falsified.

**How to apply:**
- Never assume VLM efficiency techniques transfer to VLA without validation
- VLA token pruning needs spatial/semantic signals, not attention-score signals
- Boundary token engineering may be a fruitful VLA optimization direction
- Cross-modal Token Economy (landscape 6.3) research must train separate estimators for VLM vs VLA

**Caveat:** Single-sample, single-run. Also confounded by model size (7B vs 3B). Need vanilla 3B baseline to isolate fine-tuning effect.
