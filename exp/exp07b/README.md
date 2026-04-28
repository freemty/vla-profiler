# exp07b — Pi-Zero real-weight profiling

**Parent:** exp07a (random weights, 200ms)

**What changed:** Loaded real checkpoint `bridge_beta_step19296_2024-12-26` from allenzren/open-pi-zero HF repo.

**Key finding:** Total 225ms (median), Action 183ms. vs exp07a 200ms — delta 12%. Action Expert mean delta only -2.6%. Confirms random-weight timing is faithful within ~12% system noise.

**Config:** `configs/pizero/profiling_real.yaml`
**Weights:** Real (11.77GB, bridge_beta fine-tuned on LIBERO/ALOHA)
