# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Current auxiliary source](experiments/native_decision_geometry_20260907/REPORT.md).
[Scoped obligations](experiments/native_decision_geometry_20260907/LEDGER.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed an interval upper-envelope index for scalar affine scores with exact
BF16->FP32 product domain, monotone double rounding and first-index native ties.
Produces ID and selected score bits, not all logits or same-seed softmax sampling.
Coefficient domain[-1,1] on2^-16 grid; all finiteBF16 input. Original coefficients
are not modified, domain restriction is auxiliary not a relaxed mission.

9 synthetic matrices67328queries match; separate scalar causal256steps plus2history
pairs retain context. No fullTransformer/KV/publiccheckpoint/CUDA/target execution.
V4096 queryfields p95~2.7% but source~13.24x and cold+8 lower byteaccount~3.46x.
Free head leaves99.47954% of counted405B projections. No whole-body speed construction.
State quotient and same-distribution sampling substitutions have explicit counterexamples.

13tests and51scientificfiles regenerate; archived22texts contain source, proofs,
preregistration, fixed results/manifests and retained logs/history. Full binaries
are in userZIP and regenerate, not all embedded inGit. Persistence is independent.
[Prior state unchanged](docs/research/history/pre_native_decision_geometry_20260907/RESEARCH_STATE.md).
