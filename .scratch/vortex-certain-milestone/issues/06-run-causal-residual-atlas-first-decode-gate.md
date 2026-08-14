# Run the Causal Residual Atlas First-Decode Gate

Type: task
Status: resolved
Blocked by: Preregister the Cheapest Surviving Gate

## Question

When the frozen
[Causal Residual Atlas Cheapest Real-Weight Gate](../../../docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md)
is executed on the already pinned unchanged Qwen3.5-0.8B payload, do all 18
token states, all six families, and all 36 layer-11 q/down branches survive the
one-page exact-reference output contract, or does the first valid failure
reject the registered rank-16/page-64 path before bound propagation?

## Answer

They survive the frozen favorable oracle. EXP-083A completed all 1,296 page
candidates and passed 18/18 token states, 3/3 in every family, and 36/36
projection branches. Mean/p95 KL was
`0.007225545020063708/0.037660752986209814`; 234 controls passed with zero
control, leakage, malformed-state, or baseline-trace failures.

The separate full model replay returned the same decision and deterministic-
core SHA-256
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`,
and both bundles independently verify. Evidence:
[primary](../../../results/exp_083a/summary.json),
[reproduction](../../../results/exp_083a_reproduction/summary.json), and
[authority](../../../docs/research/EXPERIMENT_083A_CAUSAL_RESIDUAL_ATLAS_GATE.md).

This resolves favorable page existence only. The native dense center and
exact-reference selector remain illegal. A post-hoc minimum-radius selector
failed 2/18 tokens, so the next ticket must freeze legal pair construction,
target-free selection, and outward certification before E2.
