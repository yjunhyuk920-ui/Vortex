# E0 Average-Oracle Amplifier Frontier

## Verdict

```text
OMEGA-XORLIFT: REJECT SELF-CONTAINED 8 GiB FORM AT SOURCE GATE
KEEP HIRAHARA--SHIMIZU ERROR-CORRECTION THEOREM VALID IN ITS MODEL
DO NOT RELABEL AN AMPLIFIER AS AN APPROXIMATE ANSWER SOURCE
KEEP COLD-BACKED GLOBAL NONLINEAR PROBE GAP OPEN
KEEP NO SURVIVING CANDIDATE
```

This audit removes the former `2.5%` premise.  It tests only whether a new
finite-field error-correction route supplies the information needed by the
unchanged-output, `8 GiB`, complete p50 contract.  No model, checkpoint,
backend, kernel, or hardware action was run.

## Elementary explanation

An error-correcting machine can turn a slightly blurry answer into a correct
one.  It cannot take an empty answer box and invent the missing answer.

`OMEGA-XORLIFT` therefore has two separate components:

```text
approximate random-matrix oracle  ->  new answer information
error-correcting reduction        ->  amplify that information
```

The cited reduction supplies the second component.  The first component is a
premise.  A small causal draft model is not automatically such an oracle: the
premise is about uniformly random finite-field matrices and query vectors,
not ordinary Transformer activations from one fixed checkpoint.

## Published result being tested

Hirahara and Shimizu prove worst-case-to-average-case error-correcting
reductions for matrix multiplication and online matrix-vector multiplication
over finite fields.  For the OMv result, an average-case data structure with
query time `O_tilde(n)` and expected coordinate accuracy strictly better than
the random `1/p` baseline is equivalent, asymptotically, to a worst-case exact
data structure with `O_tilde(n)` query time.

The theorem is an oracle reduction.  Its definitions give the oracle its own
matrix-dependent preprocessing data structure; they do not construct a small
one, cap its state at `8 GiB`, preserve BF16/FP32 accumulation order, or prove
that a target-derived causal predictor works on uniform random matrices.

## OMEGA-XORLIFT equation

The proposed binary surrogate would compile an arbitrary binary checkpoint
`W` into state `z(W)`, use a predictor on randomized finite-field queries, and
feed those noisy coordinates to a list-decoding/worst-case reduction:

```text
z(W) = Preprocess(W), |z(W)| <= S
g_z(x) approximately equals W*x for uniform x
ExactReduce[g_z](u) = W*u with negligible failure probability.
```

This differs from EXP-081A.  Syndrome repair required one query's output error
to occupy a small residual code.  OMEGA-XORLIFT permits dense errors but needs
an oracle with a uniform average-case advantage on randomized inputs and, in
the published worst-case lift, randomized matrix instances.

## Cheapest source Gate: parity-prediction capacity

Give the proposed self-contained oracle every favorable concession:

- reduce each coefficient and output coordinate to one GF(2) bit;
- give the complete `8 GiB` state to this one bit plane;
- permit arbitrary nonlinear preprocessing and arbitrary global mixing;
- make query computation, routing, randomness, and output free;
- average all other independently random query vectors into the predictor's
  internal randomness;
- charge no verifier, decoder, KV, metadata, or native numerical state;
- prohibit only cold checkpoint probes, because this is the self-contained
  oracle arm.

For a row `w in F_2^n`, its exact output bit on query `x` is the parity

```text
chi_w(x) = (-1)^(w dot x).
```

Fix one hot state `z` and one output row.  Let its possibly randomized Boolean
predictor have correctness at least `1/2 + epsilon` for a row `w` under uniform
`x`.  Its Walsh coefficient at `w` is at least `2*epsilon`.  Parseval's identity
therefore permits at most

```text
L <= 1/(4*epsilon^2)                                  (1)
```

different row parities with that advantage.  This remains true for randomized
prediction by applying Parseval to its expected signed output, whose squared
norm is at most one.

If the population has `M` independently selectable rows and `D` arbitrary
binary coefficients, one fixed state can represent at most `L^M` checkpoints.
There are at most `2^S` states.  Universal coverage therefore requires

```text
2^S * L^M >= 2^D,

S + M*log2(1/(4*epsilon^2)) >= D.                     (2)
```

This is a nonlinear-state counting Gate.  It does not assume a linear sketch,
separate per-matrix state, or a particular predictor architecture.

## Registered finite substitution

Re-deriving the population from
`results/exp_071/raw/tensor_rows.jsonl` while excluding only the embedding gives

```text
D binary coefficient bits            403,747,897,344
M output-row parity functions          19,997,952
S complete hot bits                    68,719,476,736
hot fraction                           17.020392474626%
missing bits per row                   16,753.136551582882
```

Solving (2) favorably for a common advantage gives

```text
log2(epsilon)  <= -8,377.568275791441
log10(epsilon) <= -2,521.899341736204.
```

Thus a universal hot-only oracle can guarantee at most roughly
`10^-2522` advantage over random guessing at this state point.  Two concrete
state floors make the scale clearer:

| Required average accuracy | Advantage | Minimum state | Multiple of 8 GiB |
|---|---:|---:|---:|
| `51%` | `1%` | `46.9761628441 GiB` | `5.872020x` |
| `75%` | `25%` | `46.9977852702 GiB` | `5.874723x` |

The `75%` row also connects to the elementary two-call linear
self-correction identity

```text
f(u) = f(r) + f(r+u) over F_2.
```

If a predictor differs from a parity on less than one quarter of uniformly
random inputs, the union bound makes this pair correct with probability above
one half.  The exhaustive controls verify the identity and the finite Walsh
list bound on every Boolean truth table of dimension three.  The stronger
published reduction tolerates much lower accuracy, but it still consumes an
oracle satisfying its displayed advantage premise; it does not evade (2).

## Cold-backed boundary

Equation (2) does not cover an oracle that reads the original checkpoint while
answering each randomized query.  Such an oracle returns to the active global
adaptive-probe problem.  It must specify and charge

```text
oracle preprocessing state
+ every cold probe per oracle call
+ number of amplification calls
+ random-instance construction
+ list decoding and verification
+ failure amplification
+ native numerical lifting
+ fallback.
```

The error-correcting theorem cannot make these terms zero.  In particular, a
target-derived 4B proposer tested only on causal activations does not meet the
uniform-random oracle premise, and a full target evaluator used as the oracle
merely restores the dense cost.

## Native-semantics boundary

The published algebra is over associative finite fields.  Reference
Transformer execution uses Q4/BF16 products, FP32 or implementation-specific
accumulation, rounding, nonlinearities, and a token decision.  A future route
would need an exact native-order reduction or a proof that its field result
determines the native result.  This audit grants that missing lift for free at
the source Gate; the hot-only oracle still fails.

## Decision and next admissible work

```text
REJECT_SELF_CONTAINED_AVERAGE_ORACLE_AMPLIFIER_AS_CORE
DO_NOT_CALL_ERROR_CORRECTION_A_NEW_ANSWER_SOURCE
DO_NOT_RUN_A_MODEL_OR_HARDWARE_GATE_FOR_OMEGA-XORLIFT
KEEP A CONCRETE COLD-BACKED APPROXIMATE ORACLE OUTSIDE THIS REJECTION
KEEP THE GENERAL FINITE-WORD RANK-ONE PROBE TICKET CLAIMED
```

Reopening requires a concrete oracle constructor, not another reduction.  It
must show where its above-random exact coordinate information comes from and
close the complete state, build, call-count, probe, traffic, operation,
verification, native-order, miss, and fallback equations before E1.

## Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_average_oracle_amplifier_frontier.py `
  --output-dir results\e0_average_oracle_amplifier_frontier

.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_average_oracle_amplifier_frontier.py
```

Expected focused result: `8 passed`.  Authority:

```text
results/e0_average_oracle_amplifier_frontier/summary.json
results/e0_average_oracle_amplifier_frontier/checksums.sha256
```

## Primary source

- Shuichi Hirahara and Nobutaka Shimizu, *Error-Correction of Matrix
  Multiplication Algorithms*, ECCC TR25-031 (2025):
  https://eccc.weizmann.ac.il/report/2025/031/
