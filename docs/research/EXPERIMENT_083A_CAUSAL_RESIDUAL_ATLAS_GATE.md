# EXP-083A -- Causal Residual Atlas First-Decode Gate

Date opened: 2026-08-09
Status: GATE PASSED, INDEPENDENTLY VERIFIED AND REPRODUCED
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
and dense. An initial full-model vectorized implementation was rejected by its
control because changing the batch width changed the pre-patch BF16 activation.
A second implementation replayed every candidate through all 24 layers at
batch size one; it produced no scientific row before the fixed 1,804-second
command timeout and is classified as infrastructure failure only.

The corrected runner executes the prefix and pre-branch decode at batch size
one and verifies that the captured layer-11 and projection inputs are
bitwise-identical to the unchanged baseline. A first suffix implementation
expanded all 18 q candidates in one batch but was stopped by a 1,804-second
command timeout before a scientific row. A fourth attempt serialized suffixes;
one candidate still had not completed after at least 1,989 seconds and 2,525
CPU-seconds, so that implementation was terminated as an infrastructure-only
failure. The current path restores one shared suffix batch and grants it a
two-hour command window. Dense-patch and all-page identity arms must retain the
frozen top-1. Every page still receives a separate final-logit row and
exact-reference KL.

## Frozen stop

The first valid token for which either required projection has no
top-1-preserving page produces
`REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH` and stops. If no such
failure occurs, all 18 token states, six families, 36 branches, mean KL, and
p95 KL must pass before the candidate may advance to a separate legal-center
and outward-bound Gate.

## Authoritative result

The complete registered population executed from source commit
`1c7dd78097beaa7bc159a8f3459451b876a1338e`. The primary bundle is
`results/exp_083a`; the separate model replay is
`results/exp_083a_reproduction`.

MEASURED:

- 18/18 token states, 3/3 in each of six families, and 36/36 projection
  branches preserved the unchanged greedy top-1;
- all 1,296 page candidates were evaluated;
- selected mean KL was `0.007225545020063708` and nearest-rank p95 KL was
  `0.037660752986209814`, below the frozen `0.02/0.05` limits;
- 234 controls passed, with zero control failures, leakage failures, malformed
  states, or baseline-trace mismatches; and
- the independent verifier rebuilt the selection, aggregation, and
  deterministic core with SHA-256
  `ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`.

The separate model replay produced the same decision, every registered row,
and the same deterministic-core SHA-256, then independently verified again.
The authoritative decision is therefore:

```text
PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE
```

DERIVED FROM MEASURED ROWS: 1,234/1,296 (`95.216049%`) individual pages
preserved top-1. Every branch had at least two preserving pages; the q/down
minima were 8/2. This makes page existence materially less speculative, but
does not supply a selector.

An explicitly post-hoc, non-authoritative audit chose the page with minimum
`certified_unread_radius`, without target logits. It achieved only 34/36
branches and 16/18 tokens, with p95 KL `0.075846638542797`; it therefore does
not satisfy the Gate. Across oracle-selected branches, even the tightest
current certificate radius was `19.852750x` the exact unread error. The next
Gate must improve both legal selection and enclosure rather than treating the
oracle pass as a runtime.

The runner's original Windows memory API signature returned a null internal
peak-RSS field. External OS lifetime counters observed primary/reproduction
CPU-process peaks of `3,513,356,288` and `3,516,215,296` bytes. The source is
fixed after the run. These are CPU working-set observations, not GPU VRAM or a
physical performance Gate; primary timing also included brief contention from
a stale earlier attempt.

## Claim boundary after the pass

This is a positive E1 necessary-condition result on one unchanged 0.8B
checkpoint. It proves neither a target-free page selector nor legal pair-built
`Z=WQ`, an outward-rounded end-to-end certificate, simultaneous projection or
layer composition, dense-operation replacement, physical speed, 8 GiB GPU
residency, 122B/405B scaling, or E2-E7. The private Ubuntu server was not
contacted. Only a separately preregistered legal causal-pair and bound Gate is
authorized next.
