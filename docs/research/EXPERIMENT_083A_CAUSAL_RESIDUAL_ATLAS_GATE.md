# EXP-083A -- Causal Residual Atlas First-Decode Gate

Date opened: 2026-08-09
Status: SOURCE IMPLEMENTED, NO MODEL RESULT
Phase: C small-real-checkpoint favorable-oracle falsification
Evidence ceiling: E1

## Immutable authority

The scientific contract is the already committed and hash-pinned
`docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`. Its SHA-256 is
`8241f46ec5e24f6a484b32327c7d9fb2c841828f908193d37e25ac06d17b8837`
at preregistration commit `1373f928f7d4a71e252f328e754f21eea5eef948`.

EXP-083A may implement that contract but may not alter its rank, page width,
prompt population or order, teacher position, layer, projection order, oracle,
quality threshold, failure classification, or stop rule after observing a
checkpoint result.

## Execution boundary

The runner performs one unchanged prefill per evaluation prompt and captures
prompt-position inputs and images for layer-11 `q_proj` and `down_proj`. It
then verifies the EXP-076 teacher trace at prompt-last and teacher position 1.
Each projection branches the immutable prefix cache across all page candidates
plus dense-patch and all-page identity controls.

Only one projection is patched at a time; every other operation is unchanged
and dense. Every candidate is replayed at batch size one from an independently
cloned immutable prefix cache. An initial vectorized implementation was rejected
by its control because changing the batch width changed the pre-patch BF16
activation. The batch-one path retains a separate final-logit row and
exact-reference KL for every page and requires bitwise-equal pre-patch input
and native projection output on every replay.

## Frozen stop

The first valid token for which either required projection has no
top-1-preserving page produces
`REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH` and stops. If no such
failure occurs, all 18 token states, six families, 36 branches, mean KL, and
p95 KL must pass before the candidate may advance to a separate legal-center
and outward-bound Gate.

No result is recorded in this source document yet.
