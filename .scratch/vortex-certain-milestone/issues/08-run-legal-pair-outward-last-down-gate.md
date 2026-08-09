# Run the Legal Pair and Outward-Bound Last-Down Gate

Type: task
Status: resolved
Blocked by: Preregister the Legal Causal-Pair and Outward-Bound Gate

## Question

When the frozen
[Legal Pair and Outward-Bound Gate](../../../docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md)
is executed exactly once on the new 24-prompt population and unchanged pinned
Qwen3.5-0.8B payload, can pair-only `Q_hat/Z_hat`, the maximum-residual-energy
page, and the strict final outward certificate preserve 24/24 native greedy
decisions with zero fallback and false accept, or does the first untouched row
reject the legal Atlas primary path before backward-layer integration?

## Answer

The first untouched row rejects the legal Atlas primary path. Pair-only MGS
reached rank 16 and the target-free selector chose page 0, but the strict final
certificate was unresolved. Exact dense completion replayed and produced one
valid fallback, which violates the required 24/24 zero-fallback Gate.

All 19 controls passed with zero leakage, malformed state, false accept, or
dense replay mismatch. Independent verification performed zero model forwards
and rebuilt deterministic-core SHA-256
`57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.

The frozen down radius was `23.4205200666`, dominated by unread residual
`16.3298572850` and pair-image `6.2376633562`. Post-hoc necessary-condition
analysis shows actual hidden separation `38.9079080403` versus an ideal
top-two row-margin limit `4.1978252811`, so tightening only RMS implementation
rounding cannot rescue the same certificate family.

Decision:
`REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH`. Evidence:
`../../../results/exp_083b`.
