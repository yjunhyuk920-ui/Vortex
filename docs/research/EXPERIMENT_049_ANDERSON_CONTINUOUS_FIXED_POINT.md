# EXP-049 — Anderson-Accelerated Continuous Block Fixed-Point Gate

## Final status

```text
Scientific decision: REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
Phase: A/B/C-observation
Evidence: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

Authoritative evidence:

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact size 105493 bytes
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

## Question tested

EXP-048 showed that an already-correct 96-token future proposal could be verified with one exact target pass at 1.0416667% logical target-stream fraction, but hard Jacobi and sequential early-layer drafting could not produce such a proposal.

EXP-049 removed the per-token draft loop and tested whether a small number of full causal target passes over continuous future-token embeddings, combined with damped Picard or bounded Anderson acceleration, could produce a long exact block.

## Frozen map and conditions

For exact prefix embeddings and future block `Z`:

```text
L(Z) = target_logits(prefix || Z)
P_i = top-k-softmax(L_i / temperature)
F(Z)_i = sum_v P_i(v) E(v)
R(Z) = F(Z) - Z
```

Logits were aligned at `prefix_length - 1 + i`.

Conditions:

- S0 hard synchronous Jacobi;
- S1 fixed damped Picard variants;
- S2 Anderson histories 2/4/8 with float64 constrained solve;
- S3 exact future-state oracle, non-deployable;
- S4 hidden triangular adversarial models.

Fixed sweep:

```text
blocks K = 64, 128, 256
target solver pass checkpoints = 1, 2, 4
top-k = 1 or 8
damping = 0.5 or 1.0
initialization = zero, repeated last token, repeated exact next token from a counted pass
```

Anderson used regularization `1e-8`, coefficient clipping `[-10,10]`, normalization to sum one, condition limit `1e12`, finite checks, and fail-closed Picard fallback.

## Pinned corpus

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Six held-out families: English narrative, Korean, code, mathematics, structured JSON, and identifier boundary.

```text
3 models
18 cases
1,458 fixed trajectory rows
excluded states 0
```

## Correctness, causal, and numerical result

MEASURED:

```text
9 EXP-049 tests passed
repository validation passed
selected exact-output mismatches 0
selected target-future information uses 0
unhandled numerical failures 0
selected numerical fallbacks 0
S3 exact future-state alignment failures 0
peak RSS 684684 KiB
```

The retained exact verifier committed only the longest proposal-matching prefix plus the exact first-mismatch target correction.

## Favorable real-checkpoint upper bound

The exact reference was allowed to choose the best pre-registered S1/S2 variant, block size, and pass count independently for each case. This is a non-deployable favorable selector.

MEASURED:

```text
oracle-best S1/S2 p50 matching prefix 4.5
oracle-best S1/S2 maximum matching prefix 6
oracle-best S1/S2 p90 target-equivalent fraction 1.6877859596
                                                   = 168.778596%
model median prefixes 1M 4.5 / 3M 5.0 / 8M 4.0
```

Every selected result used:

```text
block size 64
total target solver passes 4
```

Selection distribution:

```text
17/18 S1_last_k1_l1 hard top-1 Picard
1/18  S1_last_k8_l1
0/18  Anderson
```

The exact verifier added one target pass. Selected committed-token counts were 5–7 because the first mismatch correction adds one token beyond the 4–6 matching proposal tokens.

## Jacobi and Anderson comparison

After four target solver passes:

```text
S0 hard Jacobi p50 matching prefix 4
S2 Anderson p50 matching prefix 1
S2 / S0 improvement 0.25x
```

Anderson improved the committed linear-contraction positive control in unit tests, but did not improve the causal Transformer block proposals. Continuous residual reduction was not equivalent to exact-token prefix propagation.

## Hidden triangular lower-bound audit

The causal oracle family was:

```text
F(Z)_0 = E(t_0)
F(Z)_i = E(t_i) only if hard(Z_{i-1}) == t_{i-1}
         E(decoy) otherwise
```

MEASURED on two 64-position chains:

```text
Picard prefixes by round   1, 2, 3, 4
Anderson prefixes by round 1, 2, 3, 3
Anderson numerical fallbacks 0
hidden suffix transcript indistinguishability true
one-new-exact-position-per-round barrier observed true
```

Declared interface:

- one synchronous black-box causal target block evaluation per round;
- exact prefix, fixed initialization, all previous target outputs/states, and arbitrary continuous/Anderson history arithmetic allowed;
- no external future information.

Before an exact predecessor is resolved, two targets with different hidden suffixes return the same decoy suffix. The solver cannot identify which exact suffix is present from the available transcript. Continuous embeddings and Anderson extrapolation do not add this missing information.

This is a worst-case universal construction. It does not claim that every real prompt advances exactly one position.

## Pre-registered Gate outcome

Required:

```text
p50 exact matching prefix >=16
p90 target-equivalent fraction <=10%
Anderson p50 >=4x max(1, Jacobi p50)
zero exact mismatch
zero S1/S2 future information
zero unhandled numerical failure
non-degrading model-size trend
no universal triangular round barrier
```

Observed:

```text
prefix FAIL: 4.5 <16
traffic FAIL: 168.778596% >10%
Anderson improvement FAIL: 0.25x <4x
universal barrier FAIL: observed true
exactness PASS
causality PASS
numerical safety PASS
model-size trend PASS
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

## Scientific interpretation

- The exact verifier remains valid auxiliary machinery.
- Picard/Anderson code and numerical fault handling remain reusable references.
- Target-only synchronous rounds do not supply enough future information for the arbitrary-model exact mission.
- Average-case real-checkpoint prefixes did not come close to the early Gate even with reference-selected variants.
- Anderson hyperparameter tuning cannot fix the hidden-predecessor information barrier.
- No target-only continuous fixed-point GPU backend is justified.

A future mechanism must import a new information source, compile target-specific transition advice, relax exactness, or change hardware assumptions.

## Next mechanism boundary

EXP-050 tests already-published, unmodified **external target-independent draft checkpoints** as the new information source. Cross-checkpoint proposals are exact-verified and fully charged.

Universal warning: a fixed target-independent draft can be contradicted at the first token by an arbitrary target.

PROJECTED 4B draft arithmetic:

```text
4B/405B = 0.0098765432 target streams per proposed token
required total fraction = 0.01185185185
perfect proposal minimum after draft cost = 507 tokens
```

## Projection and claim boundary

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required fraction: 1.185185%
EXP-049 favorable p90 / required: about 142.405x

405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
physical block weight reuse: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
complete real operation replacement: false
Phase D: NOT TESTED
E6/E7: not achieved
```

## Reproduce

```bash
python -m pytest -q tests/exp_049
python scripts/run_validation.py
bash experiments/exp_049/reproduce.sh
cd results/exp_049 && sha256sum -c checksums.sha256
```
