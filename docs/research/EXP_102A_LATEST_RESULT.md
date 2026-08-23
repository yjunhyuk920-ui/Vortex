# EXP-102A latest result — Reality-First Causal Draft/Verify Gate

- source commit: `8fef54bcd80932616adb8c682b55d420fc9faadd`
- raw result: `results/exp_102a/8fef54bcd80932616adb8c682b55d420fc9faadd/result.json`
- decision: `REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE`
- deterministic core: `1d4b2b6576a833bdfe89245d7ca82a65c65108244bb94a540fef2b4afda90d15`
- authoritative arm: `REAL_EXECUTOR_ONLY`
- forbidden grants: `['future_target_tokens', 'perfect_selector', 'free_transforms', 'free_metadata', 'free_workspace', 'free_repair', 'free_fallback', 'free_N_over_A', 'unmeasured_compression', 'peak_throughput_as_measurement']`
- integrity failures: `[]`
- raw p50/p95 minimum committed tokens: `85 / 68`

## Untouched holdout

- `cross_family_text_bridge`: K `96`, minimum A `1`, latency p50/p95 `12.944473 / 17.076776`, N/A p95 `17.000000`, exact token+state `True`, pass `False`
- `same_family`: K `96`, minimum A `2`, latency p50/p95 `72.812976 / 85.345943`, N/A p95 `48.500000`, exact token+state `True`, pass `False`

## Claim boundary

This is an executable E2/E3 CPU Gate with real public draft and target checkpoints. It charges draft generation, target candidate positions, mismatch repair, draft-state rebuild, N/A, cache bytes, and wall time. Target 405B execution, physical 8-GiB residency, target storage/H2D, CUDA/SASS, and same-machine native-4B-Q4 acceptance remain `NOT TESTED`.
