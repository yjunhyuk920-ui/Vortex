# Next Experiment

## Closed Gate — EXP-055

Authority: `results/exp_055/summary.json`; workflow `30820909775`; source head `c15b1bb94496ad629bf8911d30d47a7cbe792595`; artifact `8858805996`; ZIP SHA-256 `983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8`.

Exact identical/sign-related grouping was correct and ideal structured cases improved below 10% at n=64, but general p50/p90 operations were 62.5%/250%, query bytes 63.64%/200%, and dense/unique p50 250%. Decision:

```text
REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY
```

## EXP-056 — Exact Prototype Plus Sparse-Residual Dictionary Gate

### Mechanism change

Generalize exact repetition without changing the model. Automatically compile each exact weight column as:

```text
column_i = prototype[group_i] + exact_sparse_residual_i
score = bias + sum_g popcount(active members_g) * prototype_g
              + sum_active_i residual_i
```

Prototype selection is deterministic and weight-derived. Every nonzero residual scalar, index, membership mask, prototype read, popcount, multiply/add, compile search, and fallback is charged. No approximation, training, target adapter, or runtime state table is allowed.

### Conditions

```text
G0 independent signed modular top-1 reference
G1 deterministic exact prototype construction
G2 exact sparse residual reconstruction
G3 scalar and packed evaluator
G4 repeated/sign-related/sparse/low-rank controls
G5 dense-random and forced-unique adversaries
G6 exhaustive small-domain and deterministic larger-domain validation
```

### Registered search

Test prototype counts 1/2/4/8 and deterministic medoid/most-frequent candidates. Select only by fully accounted operation then byte cost; charge every attempted compilation. Residuals remain exact signed integers.

### Early rejection Gate

```text
exact mismatch >0
runtime state table used
p50 operations >10% or p90 >25%
p50 bytes >10% or p90 >25%
dense-random/unique p50 >25%
projected storage >1 TiB
compile amortization >1,000,000 queries
savings degrade with input/classes
```

Failure decision:

```text
REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY
```

### Evidence boundary

Phase A/B, E1. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.

### Next exact action

1. implement deterministic exact prototype and sparse-residual compiler;
2. add independent reconstruction/evaluation validators;
3. execute the registered structured and adversarial matrix;
4. freeze all accounting, binaries, checksums, and decision;
5. promote to real checkpoint extraction only if the universal synthetic Gate survives.
