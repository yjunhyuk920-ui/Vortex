# E0 Global-Advice Synergy Frontier

## Verdict

```text
REJECT_NAIVE_GLOBAL_ADVICE_DIVISION_AND_CKL_TILE_SUM_AS_TARGET_BOUND
KEEP_THE_SINGLE_MATRIX_CKL_THEOREM_VALID_IN_ITS_MODEL
KEEP_MODEL_WIDE_NONLINEAR_DIRECT_SUM NOT ESTABLISHED
KEEP_UNIVERSAL_2.5% NOT ESTABLISHED
KEEP_NO_SURVIVING_CANDIDATE
```

This audit asks whether the global 8 GiB advice allowance can be divided by a
matrix or tile count and inserted into the single-matrix exact rank-one lower
bound of Chakraborty--Kamma--Larsen (CKL). It cannot. A finite XOR example
shows that conditional information can be shared synergistically across every
matrix. The existing theorem remains valid for one matrix, but the proposed
model-wide lift needs a new direct-sum theorem that also charges the probes
used to unlock cross-matrix advice.

The audit also grants several invalidly favorable composition steps and shows
that the resulting finite lower-bound screen is still far below the requested
DFloat block budget. This does not prove the VORTEX target feasible. It proves
that this particular shortcut cannot prove it impossible.

## Elementary explanation

Imagine three sealed boxes. Each box contains a three-bit number. Before the
boxes are sealed, one small receipt is written:

```text
receipt = box 1 XOR box 2 XOR box 3
```

The receipt is only three bits long. Yet if boxes 2 and 3 are opened, the
receipt reconstructs every bit of box 1. The same is true for any chosen box:

```text
box 1 = receipt XOR box 2 XOR box 3
box 2 = receipt XOR box 1 XOR box 3
box 3 = receipt XOR box 1 XOR box 2
```

It is therefore wrong to say that each box received only one third of the
receipt. The same three-bit receipt can reveal three conditional bits about
each box after the other boxes are known.

There is no magic free information. To exploit the receipt, the other boxes
must already be known or read. A correct model-wide lower bound must count
those cross-box reads. The audited single-matrix theorem does not provide that
joint accounting, so dividing 8 GiB by the number of matrices is not a proof.

## 1. The single-matrix theorem being tested

For an arbitrary binary `n x n` matrix in the systematic exact model, CKL
Theorem 1.3 gives the asymptotic tradeoff

```text
t * r = Omega(n^3 / log n),
```

where `r` is the redundancy/advice associated with that matrix and `t` is the
number of probed matrix bits needed to answer an exact GF(2) rank-one query
`u^T M v`. The theorem statement covers approximately

```text
n <= r <= n^2/4.
```

The displayed proof first treats the narrower `r <= n^2/64` regime and gives
the finite inequality

```text
2048 * t * r / n^2 * log2(n) >= n/4,
```

or

```text
t * r >= n^3 / (8192 * log2(n))
```

inside that proof regime. The asymptotic `Omega` statement does not authorize
setting its hidden constant to one at the registered finite size.

Most importantly, the proof analyzes one target matrix. It conditions on the
other information made available to that problem instance. It does not state
that one globally mixed nonlinear 8 GiB string can be evenly split among
1,386 Transformer tiles, nor that separately selected worst-case queries can
be combined into one causal execution.

Primary source:
<https://cs.au.dk/~larsen/papers/BooleanMatrixVectorLB.pdf>.

## 2. Exact XOR synergy counterexample

Let `M_1, ..., M_m` be independent uniformly random `N`-bit matrix encodings
and let the global advice be

```text
R = M_1 XOR M_2 XOR ... XOR M_m.
```

Then `R` has only `N` bits of entropy. But, for every target `i`,

```text
M_i = R XOR (XOR of all M_j for j != i).
```

Consequently,

```text
H(R) = N,
I(M_i ; R | M_-i) = N,
sum_i I(M_i ; R | M_-i) = mN.
```

The last sum can be `m` times larger than the advice entropy. Conditional
information is not an additive resource that may be divided by instance
count.

The executable control exhausts all triples of three-bit words:

```text
cases                                      512
target recovery checks                    1,536
failures                                  0
advice histogram                          [64,64,64,64,64,64,64,64]
advice entropy                            3 bits
sum of conditional information            9 bits
conditional/advice amplification          3x
```

This is a counterexample to the division step, not a feasible VORTEX
constructor. Recovering a matrix through this XOR advice requires the other
matrices. Any target algorithm must pay to know or probe them unless they are
already resident for a separately justified reason.

## 3. Registered full-square diagnostic

Count only complete `16,384 x 16,384` squares and omit smaller or leftover
parts favorably:

```text
Q and O squares per layer                         2
full squares across Gate, Up, and Down per layer  9
full squares per layer                           11
layers                                           126
full squares total                               1,386
```

Illegally dividing the global 8 GiB advice evenly gives

```text
average per square       34,359,738,368 / 693 bits
decimal                  49,581,152.046... bits
fraction of one square   128/693 = 18.470418...%
```

That average is below the theorem statement's `n^2/4` endpoint but above the
displayed proof regime's `n^2/64` endpoint. This numerical coincidence does
not repair the invalid division.

Even if the theorem's hidden leading constant were counterfactually set to
one and the resulting separate-square values were illegally summed, the
screen would give only

```text
6,336 probes per square
8,781,696 probes across 1,386 squares
requested DFloat block budget  110,244,000,000 bits
requested / counterfactual sum 12,553.84x
```

The units are also intentionally favorable: each probe is treated as one
perfectly packed physical bit, with no address, word, cache-line, operation,
or verification cost. The value is a weakness diagnostic for this proof
route, not a certified lower bound.

## 4. Finite pigeonhole screen without an asymptotic constant

The registered non-embedding population contains

```text
D = 403,747,897,344 binary coefficient positions.
```

Relative to those positions, 8 GiB is

```text
68,719,476,736 / D = 16,384/96,261 = 17.020392...%.
```

The smallest square dimension for which the invalid evenly divided advice
would satisfy CKL's lower endpoint `r >= n` is `n=6`. This permits
`11,215,219,370` complete 36-bit tiles. Grant every tile a rounded-up local
cap of seven bits, which already grants more advice than the global budget.

For one such arbitrary tile, a finite pigeonhole argument is exact:

1. Seven advice bits cannot injectively name all 36-bit tiles.
2. Therefore two different tiles share an advice value.
3. They differ at some matrix entry.
4. A standard-basis rank-one query selects that entry.
5. An exact systematic query algorithm must probe at least one raw bit for
   some separately selected tile/query pair.

This proves one worst-case probe for one independently considered tile. It
does not prove that all tile-hard queries occur in one jointly legal
Transformer execution. Nevertheless, if that missing composition is granted
illegally and all separate one-bit floors are summed, the result is only

```text
invalid summed floor                  11,215,219,370 bits
fraction of requested block budget    10.173088...%
requested block / invalid floor       9.829856...x
```

The displayed CKL proof coefficient is not used for this one-bit floor. At
`n=6`, the average advice is outside the displayed `n^2/64` proof regime and
the counterfactual coefficient before integer rounding is only
`0.0016647...` probes.

## 5. What is proved

```text
PROVED     global advice cannot generally be divided by matrix count
PROVED     XOR advice gives an exact finite conditional-synergy witness
PROVED     one under-described independent tile has a separate hard query
MEASURED   all 1,536 exhaustive XOR recoveries succeed
DERIVED    the registered full-square and finite-reshape substitutions
REJECTED   naive advice division plus CKL tile summation as a target bound
```

## 6. What is not proved

```text
NOT PROVED a model-wide direct-sum lower bound
NOT PROVED simultaneous composition of separately hard tile queries
NOT PROVED a charge for probes that unlock cross-matrix advice
NOT PROVED native BF16/Q4 rank-one lower bounds from the GF(2) theorem
NOT PROVED universal 2.5% feasibility
NOT PROVED universal 2.5% impossibility
```

No model forward, checkpoint mutation, backend, kernel, download, hardware
action, or EXP-085 assignment occurred.

## 7. The missing theorem must charge synergy

A useful next theorem cannot assign `r/m` advice bits to matrix `i`. It must
start with one arbitrary global advice function

```text
R = f(M_1, ..., M_m),   |R| <= 8 GiB,
```

allow an adaptive exact query algorithm to probe any matrix, and then charge
all probes needed to answer a jointly composable family of causal rank-one
queries. In particular, if `R` reveals `M_i` only after other matrices are
known, the reads that reveal those other matrices must appear in the same
bound.

Until such a theorem or a complete positive constructor is supplied, the
correct state is:

```text
PRIMARY RESEARCH AND PROTOTYPE TRACK
NO SURVIVING CORE CANDIDATE
UNIVERSAL 2.5% NOT ESTABLISHED
GENERAL NONLINEAR NUMERICAL RANK-ONE GAP OPEN
```

## Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_global_advice_synergy_frontier.py `
  --output-dir results\e0_global_advice_synergy_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_global_advice_synergy_frontier -v
```

Authoritative artifacts:

- `results/e0_global_advice_synergy_frontier/summary.json`
- `results/e0_global_advice_synergy_frontier/checksums.sha256`
- `vortex_runtime/global_advice_synergy_frontier.py`
- `tests/test_global_advice_synergy_frontier.py`

Canonical summary SHA-256:

```text
0b926e70e7fc19e76ef3e5aa1839d58f1be5d6b55900440ed1d62114c397feda
```
