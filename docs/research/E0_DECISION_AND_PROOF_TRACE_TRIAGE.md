# E0 Decision-Certificate and Proof-Carrying-Trace Triage

## Scope

This is an E0 equation audit, not EXP-082 and not a model run. It tests whether
the post-EXP-081A suggestion can become a core executor before any interval
propagator, proof runtime, kernel, or larger checkpoint work is opened.

## Direct final-decision certificate

EXP-068 already granted the complete winning output row, all metadata, and an
independently optimal coordinate reveal order for every competitor. Absolute
unread-contribution certificates still required p50/p90 global work fractions
of `13.7697%/19.2524%` on the registered small checkpoints. Repeating that
coordinate/page certificate under a new name is prohibited.

An exact maximum-inner-product tree can exploit correlations between rows that
EXP-068's per-row absolute bound ignores. That does not supply a core 405B
route: the registered 405B `lm_head` contains `2,101,346,304` of
`403,747,897,344` non-embedding coefficient uses, or `0.520459999%`. Making the
entire head free would still leave `99.479540001%` of dense linear work.

Decision:

```text
REJECT_OUTPUT_HEAD_ONLY_DECISION_CERTIFICATE_AS_CORE
RETAIN_EXACT_MIPS_INDEXING_AS_AUXILIARY
```

## Proof-carrying target trace

For each fixed matrix `W[m,n]`, preprocessing may choose a random field vector
`r[m]` and retain `z = W^T r`. A proposer supplies a claimed result `y`; the
verifier checks:

```text
r^T y == z^T x
```

Six independent challenges are granted. The audit also grants an exact
finite-field trace and free proof of every nonlinear transition. These grants
make the result deliberately favorable.

Using the frozen EXP-080A 405B non-embedding shape rows:

```text
matrix instances                         883
dense coefficient uses                  403,747,897,344
claimed trace elements/token             19,997,952
six-challenge verifier operations       234,659,328
verifier operation fraction             0.058120260%
verifier traffic                         513.724 MiB
verifier traffic fraction               0.266838924%
resident W^T r sidecar                    0.427185 GiB
union soundness bound over matrices      9.003e-54
p50 target fraction                      1.185185185%
```

The verifier is inside the logical target. The source of `y` is not. With a
same-machine dense proposer, the fully charged totals are:

```text
local operation fraction                100.058120260%
local traffic fraction                  100.266838924%
```

An external prover makes only the local verifier cheap by moving the target
execution to extra hardware. A prerecorded trace is prefix-specific advice,
already closed by EXP-052 through EXP-054 and EXP-069. A hypothetical local
sublinear trace generator would itself be the missing VORTEX executor; the
proof protocol does not construct it.

Primary-source alignment is explicit: SafetyNets, Slalom, embedded proofs, and
GKR-style systems are delegation/verification protocols with a computation
prover, not algorithms that locally produce the inference result for verifier
cost. See:

- <https://arxiv.org/abs/1706.10268>
- <https://arxiv.org/abs/1806.03287>
- <https://eprint.iacr.org/2017/1038>

Decision:

```text
REJECT_PROOF_CARRYING_TRACE_AS_A_STANDALONE_LOCAL_CORE
RETAIN_RANDOM_LINEAR_TRACE_VERIFICATION_AS_AUXILIARY
```

## E0 scorecard

| Requirement | Result |
|---|---|
| Optimistic verifier operations/traffic/storage | `0.058120260%` / `0.266838924%` / `0.427185 GiB` |
| Fully charged same-machine source | `100.058120260%` operations, `100.266838924%` traffic |
| Order-of-magnitude core path | absent without an independent trace generator |
| New information source | external prover only; forbidden by the no-added-hardware objective |
| Scale behavior | verifier fraction improves; proposer remains one target execution |
| Arbitrary checkpoint | verifier preprocessing is automatic; trace generation is unresolved |
| Correctness | probabilistic field equality, fail closed; float/nonlinear equivalence granted free |
| Cheapest falsification | the local proposer-plus-verifier conservation equation above |
| Authorized implementation | none; do not open EXP-082 for this family |

All numbers are `DERIVED`. No Transformer, 405B checkpoint, GPU, Ubuntu host,
physical proof system, or target latency was run or measured.

