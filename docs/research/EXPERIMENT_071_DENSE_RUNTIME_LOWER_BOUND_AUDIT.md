# EXP-071 — Universal Exact Dense Runtime Lower-Bound Applicability Audit

## Question

Can known unconditional online matrix-vector cell-probe lower bounds rigorously rule out the VORTEX fixed objective under a conventional exact runtime model?

The answer must be one of:

```text
CERTIFY_CONVENTIONAL_EXACT_ONLINE_DENSE_RUNTIME_LOWER_BOUND
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

No asymptotic theorem, heuristic constant, or per-matrix result may be promoted into a model-wide 405B impossibility claim without all hypotheses.

## Primary sources

1. Clifford, Grønlund, Larsen, *New Unconditional Hardness Results for Dynamic and Online Problems*, FOCS 2015, arXiv:1504.01836, DOI:10.1109/FOCS.2015.71.
2. Chakraborty, Kamma, Larsen, *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector Multiplication*, STOC 2018, arXiv:1711.04467, DOI:10.1145/3188745.3188830.

## Source-audited statements

CGL15 Theorem 3 considers a static `n x n` matrix over a finite field, represented in `S` cells of `w` bits. It gives an asymptotic probe lower bound

```text
Omega(min(
  n log|F| / log(Sw / (n^2 log|F|)),
  n^2 log|F| / w
))
```

and permits extremely high average error; an exact runtime is therefore within its correctness class.

CKL18 Theorem 1.2 considers a **systematic** Boolean matrix-vector structure: the matrix remains read-only and an additional `r` bits are stored on the side. Its registered ranges are:

```text
n <= r <= n^2/4  =>  t r = Omega(n^3)
r < n             =>  t = Omega(n^2)
```

The randomized extension succeeds with probability at least `1-1/n`.

CKL18 Theorem 1.3 gives the analogous `F2` vector-matrix-vector result, weakened by a `log n` factor, and states the matrix-vector corollary.

All three statements use asymptotic notation with hidden constants. They do not provide a finite numerical probe count for Llama dimensions.

## Exact reduction

For `M,v` over `{0,1}`, run the exact dense projection over ordinary integer/float arithmetic:

```text
y_i = sum_j M_ij v_j
```

Then:

```text
Boolean semiring output = [y_i > 0]
F2 output               = y_i mod 2
```

For registered dimensions, all sums are exactly representable in float32. EXP-071 exhaustively checks every matrix/vector pair through dimension four and directly replays float32 through dimension three.

A finite-field embedding up to `F13` also fits signed Q4 representatives `[-6,6]`, but CGL15 remains asymptotic and weak in the large-side-information regime.

## Rectangular matrices

A runtime for arbitrary `m x n` matrices contains an arbitrary square subproblem only at:

```text
k = min(m,n)
```

An arbitrary `k x k` matrix can be placed in the top-left block and unused rows/columns zeroed.

Padding an `m x n` matrix to `max(m,n)^2` lets a square solver solve a rectangular problem. It does **not** let a rectangular solver solve every larger square matrix, so it cannot strengthen the lower-bound dimension to `max(m,n)`.

## 8 GiB side-information audit

The largest square subproblem in the registered Llama-3.1-405B tensor plan is:

```text
n = 16,384
n^2/4 = 67,108,864 bits = 8 MiB
```

The fixed hot/side-information allowance is:

```text
8 GiB = 68,719,476,736 bits
```

Thus the available side information is exactly `1024x` above the maximum CKL18 tradeoff range for every individual registered tensor family. No individual tensor satisfies the succinct-redundancy hypothesis when the runtime may devote the complete hot state to that tensor.

## Why side information cannot be divided by 884 tensors

The 8 GiB state is jointly computed from the whole model. Fixing all matrices except one still leaves the complete state as an arbitrary function of the remaining matrix.

Therefore this argument is invalid without a direct-sum theorem:

```text
8 GiB / 884 tensor instances
```

One side-information bit can encode a joint function of many matrices and need not belong to one tensor. Neither registered source proves a direct-sum lower bound for many matrices sharing one redundancy string and receiving adaptively generated, layer-dependent queries.

## CGL15 finite-size indicator

The experiment evaluates the displayed CGL15 expression with constants set to one, using the strongest direct signed-Q4 field embedding `F13`. These rows are explicitly labeled:

```text
UNIT-CONSTANT INDICATOR — NOT A CERTIFIED FINITE LOWER BOUND
```

Even illegally summing every per-instance indicator as though a direct-sum theorem existed gives only about `0.0249%` of packed Q4 cells, below the fixed `1.185185%` whole-execution budget. This number is diagnostic only; it is not theorem-backed because both finite constants and direct-sum composition are absent.

## Expected authoritative decision

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

Reasons:

1. CKL18 does not cover the 8 GiB per-matrix redundancy regime.
2. Neither paper proves the required model-wide direct sum.
3. The asymptotic Omega constants cannot be converted into a finite 405B traffic fraction.
4. Cell probes are not yet mapped to physical GPU, PCIe, or SSD transactions.
5. The theorems constrain conventional systematic/general data structures, not every conceivable execution model.

This result does not show the VORTEX objective is feasible. It only prohibits claiming that these papers prove it impossible.

## Claim boundary

Phase A/B theorem and reduction audit, evidence ceiling E1.

```text
405B execution: NOT TESTED
8 GiB GPU behavior: NOT TESTED
CUDA/PCIe/SSD: NOT TESTED
TTFT/tokens per second: NOT TESTED
model-wide impossibility: NOT CERTIFIED
```
