

<!-- EXP087A_RESULT:START -->
## EXP-087A lossless entropy-stationary perfect-block Gate

- Decision: `CONDITIONAL_LOSSLESS_ENTROPY_STATIONARY_PATH_REQUIRES_TARGET_BASELINE_AND_TARGET_ENTROPY`
- Exact weight roundtrip mismatches: `0`
- Native block/sequential mismatches: `0`
- DEV-W zlib ratio: `1.261972x`
- DEV-W empirical Shannon ratio: `1.521866x`
- K=128 zlib traffic fraction: `0.619070913%`
- K=128 full-MLP exact-vector p50: `100.000000000%`
- Deterministic core: `26064d1b8648ea8048256fc629d5178f3669425bf21120513613b07411707b2f`

This Gate preserves exact weights on a small public checkpoint and uses perfect future activations. Target 405B entropy, target native 4B baseline, target effective compute, CUDA, 8 GiB, and latency remain `NOT TESTED`.
<!-- EXP087A_RESULT:END -->
