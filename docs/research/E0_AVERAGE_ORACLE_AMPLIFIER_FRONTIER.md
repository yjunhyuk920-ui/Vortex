# E0 Average-Oracle Amplifier Frontier

## Corrected verdict

```text
HIRAHARA--SHIMIZU AVERAGE-DISTANCE AMPLIFIER: VALID IN ITS MODEL
COMMON-PER-ROW FOURIER GATE: VALID BUT STRICTLY NARROWER THAN THE PAPER PREMISE
OMEGA-ROWLOTTERY DIRECT SOURCE: REJECT AT THE RANK-COVERAGE GATE
UNCHARGED OMEGA-XORLIFT COMPOSITION: REJECT AS A VORTEX CORE
GENERAL COMPRESSED COLD-BACKED ORACLE: OPEN
TARGET NOT ACHIEVED
```

This correction removes the former `2.5%` premise and also fixes a quantifier
error in the first OMEGA-XORLIFT audit. The published premise averages Hamming
error over output coordinates. It does **not** require every row predictor to
have the same positive advantage. The earlier Fourier calculation remains a
sound theorem for that stronger common-row condition, but it cannot reject
the paper's average-distance oracle by itself.

No model, checkpoint, backend, kernel, or hardware action was run.

## Elementary explanation

An error corrector can repair a worksheet when some answers are already
right. It still needs a machine that writes those partly right answers.

There are two ways the partly right answers might be distributed:

```text
every row is slightly better than guessing
or
a few rows are exactly right and all other rows are guesses.
```

The paper allows the second case because it measures the average over all
rows. The original audit accidentally tested only the first case.

That observation suggests a concrete source, `OMEGA-ROWLOTTERY`: compute a
small subset of encoded rows exactly, guess the rest, randomly move the exact
subset between calls, and error-correct the combined answers. Its cheapest
Gate fails for a different reason. To recover an arbitrary `n`-coordinate
answer, all calls together need at least `n` independent row equations. A
direct row equation contains `n` arbitrary checkpoint coefficients. Thus the
calls read at least `n*n` coefficients in total: one full matrix again.

## 1. What the publication actually assumes

Hirahara and Shimizu define normalized Hamming distance as the fraction of
disagreeing output coordinates. Their OMv theorem assumes a randomized
preprocessed oracle satisfying

```text
E_{A,v}[dist(O(A;v), A*v)] <= 1 - 1/p - epsilon
```

for uniform finite-field matrices and vectors. It constructs a randomized
worst-case exact data-structure algorithm with query time near-linear in the
matrix dimension, multiplied by a function of `p` and `epsilon`. In the
small-field theorem, both preprocessing and query phases make up to

```text
2^poly(p,1/epsilon) * O(log n)
```

oracle calls. The paper permits polynomial preprocessing and does not bound
the total derived data-structure state by `8 GiB`, preserve native BF16/FP32
order, or count VORTEX storage traffic.

The reduction is therefore a real amplifier, but every oracle data structure
created in preprocessing and every repeated query call must be materialized
and charged before it becomes a VORTEX constructor.

## 2. Counterexample to the former common-row inference

Let a binary oracle keep an exact fraction `f` of rows and output baseline
guesses for all other rows. Its average coordinate accuracy is

```text
f * 1 + (1-f) * 1/2 = 1/2 + f/2.                 (1)
```

At the registered one-bit population, granting all `8 GiB` to exact rows
gives

```text
f                                      17.020392474626%
average GF(2) coordinate accuracy      58.510196237313%
average advantage                       8.510196237313%
```

This satisfies the **accuracy part** of the paper's premise with a constant
advantage. It does not satisfy the near-linear query-time part: directly
forming a constant fraction of dense row products still takes a constant
fraction of `n^2` coefficient operations and reads.

Equation (1) is the decisive scope correction. An average-distance theorem
cannot be replaced by a theorem that assumes a common advantage for every
row.

## 3. The Fourier Gate that remains valid

For a stronger oracle that promises correctness `1/2 + epsilon` on **every**
binary row parity, fix one hot state and one output row. Parseval's identity
bounds the number of row parities predicted with that advantage by

```text
L <= 1/(4*epsilon^2).
```

For `M` independently selectable rows, `D` arbitrary binary coefficients, and
`S` hot bits, universal common-row coverage requires

```text
2^S * L^M >= 2^D,

S + M*log2(1/(4*epsilon^2)) >= D.                  (2)
```

This permits arbitrary nonlinear global preprocessing, randomized prediction,
and free query computation. Re-deriving the frozen non-embedding population
gives

```text
D binary coefficient bits            403,747,897,344
M output-row functions                 19,997,952
S complete hot bits                    68,719,476,736
common-row log2(epsilon) ceiling       -8,377.568275791441
common-row log10(epsilon) ceiling      -2,521.899341736204
```

The calculation and exhaustive three-bit truth-table controls remain valid.
Its conclusion is now explicitly scoped:

```text
REJECT COMMON-PER-ROW HOT-ONLY ORACLE
DO NOT USE THIS TO REJECT AN AVERAGE-DISTANCE ORACLE
```

## 4. New direct-source candidate: OMEGA-ROWLOTTERY

`OMEGA-ROWLOTTERY` makes the concentration counterexample constructive:

```text
preprocess transformed/encoded rows of A
each oracle call evaluates k exact encoded row forms
all other coordinates receive baseline guesses
the amplifier permutes/encodes calls and decodes A*v exactly.
```

Give this source favorable exact field arithmetic, free guessing, free
permutations, free code construction, free decoding, and no metadata. Let the
true output have dimension `n`. If all calls together expose fewer than `n`
independent row forms, their observation matrix has rank below `n`; two output
vectors share the same observations, so exact recovery for every `A,v` is
impossible. Hence

```text
total independent exact row forms >= n.                         (3)
```

A direct encoded form `q^T A` contains `n` arbitrary field coefficients and
its product with `v` reads/evaluates those coefficients. Combining (3) gives

```text
direct coefficient payload >= n*n.                              (4)
```

Changing which rows are exact, applying an invertible row transform, or
splitting the forms across many oracle calls does not reduce this rank floor.
It only redistributes the same complete row basis.

For the frozen model-wide one-bit substitution, the favorable direct floor is

```text
403,747,897,344 bits
= 47.00244140625 GiB
= 5.87530517578125 * the complete 8 GiB hot grant
= 9.155779392620% of the reported 551.22 GB DFloat denominator.
```

Thus direct row evaluation is not a traffic reduction. If all row forms are
resident, exact basis queries recover the arbitrary checkpoint and require at
least the same one-bit information content. If they are cold, evaluating a
rank-covering set reads at least that content per direct composition. Q4,
native products, code expansion, repeated calls, decoding, verification, and
state only increase the cost.

This is a scoped constructor rejection. It does not prove that every succinct
nonlinear approximate oracle must explicitly evaluate a rank-covering list of
row forms.

## 5. Remaining open oracle

A surviving OMEGA-XORLIFT source must be materially different from direct
row lottery. It must provide above-baseline average coordinates while using a
compressed nonlinear cold-backed query mechanism, and close

```text
all oracle preprocessing structures
+ all transformed checkpoint payloads
+ every probe on every repeated call
+ address generation and computation
+ list decoding and verification
+ failure amplification and fallback
+ exact native numerical lifting.
```

A causal proposer measured only on ordinary Transformer activations does not
automatically satisfy the uniform random-matrix/vector premise. A full target
evaluator used as the oracle merely restores dense work.

## 6. Native-semantics boundary

The publication works over associative finite fields. Reference Transformer
execution uses quantized/BF16 products, FP32 or implementation-specific
accumulation, rounding, nonlinearities, and a token decision. A future route
needs an exact native-order reduction or a proof that its field result
determines the native result. This audit grants that missing lift for free at
the row-lottery source Gate, which still fails.

## Decision

```text
RETRACT_THE_CLAIM_THAT_THE_COMMON_ROW_GATE_COVERS_THE_PUBLISHED_PREMISE
KEEP_THE_COMMON_ROW_FOURIER_THEOREM_AS_A_SCOPED_GATE
REJECT_DIRECT_OMEGA-ROWLOTTERY_AS_A_TRAFFIC_REDUCTION
REJECT_AN_UNCHARGED_AMPLIFIER_AS_A_CORE
KEEP_A_CONCRETE_COMPRESSED_COLD_BACKED_ORACLE_OPEN
KEEP_THE_GENERAL_FINITE_WORD_RANK_ONE_PROBE_TICKET_CLAIMED
```

## Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_average_oracle_amplifier_frontier.py `
  --output-dir results\e0_average_oracle_amplifier_frontier

.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_average_oracle_amplifier_frontier.py
```

Expected focused result: `11 passed`. Authority:

```text
results/e0_average_oracle_amplifier_frontier/summary.json
results/e0_average_oracle_amplifier_frontier/checksums.sha256
```

## Primary source

- Shuichi Hirahara and Nobutaka Shimizu, *Error-Correction of Matrix
  Multiplication Algorithms*, ECCC TR25-031 (2025):
  https://eccc.weizmann.ac.il/report/2025/031/
