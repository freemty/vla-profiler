# exp06b — NitroGen full 500M real-weight profiling

**Parent:** exp06a (174M shrunk variant, random weights)

**What changed:** Aligned to official ng.pt checkpoint (vision_hidden=1024, action_dim=25, vl_num_heads=16).

**Key finding:** Per-step 7.1ms — nearly identical to exp06a's 7.2ms. DiT hidden dim (16×64=1024) was already correct in exp06a; the "shrunk" params (vision_hidden, action_dim) only affect cross-attn I/O, not DiT core. DiT is 181M, not the headline "500M" (which includes 316M SigLIP).

**Config:** `configs/nitrogen/profiling_full500m.yaml`
