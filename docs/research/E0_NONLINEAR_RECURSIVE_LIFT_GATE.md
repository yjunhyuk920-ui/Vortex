# E0 Nonlinear Recursive-Lift Gate

## Status

- Date: 2026-08-12 Asia/Seoul
- Status: EXACT NECESSARY CONDITION; FREE RECURSIVE LIFT REJECTED
- Evidence: elementary two-site theorem and exhaustive controls through arity 4
- Model/checkpoint/hardware actions: none
- Candidate promoted: none

## Question

The adaptive nonlinear probe-degree Gate left a tiny capacity survivor:

```text
source       arbitrary 2 x 3 binary matrix
storage      seven bits
query        every rank-one parity
budget       two adaptive probes
```

Even if such a seed exists, it matters at target scale only if small encoded
blocks can be composed into large encoded blocks. The cheapest proposed
composition treats every position of several child blocks independently:

```text
Y[p] = f(X_1[p], ..., X_k[p]).
```

It then tries to answer a rank-one parity of `Y` from the matching rank-one
parities of `X_1,...,X_k`. This note tests exactly that recursion and nothing
broader.

## Two-site theorem

Choose a rank-one mask containing two positions in one row (or one column).
Let the `k` source bits at those positions be `x` and `y`. The only information
returned by the matching child queries is

```text
x XOR y.
```

The parent answer is

```text
f(x) XOR f(y).
```

Therefore a child-answer-only decoder exists for every source only if

```text
f(x) XOR f(y) depends only on x XOR y.                 (1)
```

Set

```text
h(x) = f(x) XOR f(0).
```

Equation (1) gives

```text
h(x XOR y) = h(x) XOR h(y).
```

Thus `h` is a homomorphism from `F_2^k` to `F_2`, so `h` is linear and `f` is
affine. The converse is immediate: for

```text
f(x) = c XOR a_1 x_1 XOR ... XOR a_k x_k,
```

the parent parity is the selected-position parity of `c` XOR the indicated
child parities.

Hence:

```text
entrywise black-box recursion preserves all matching rank-one parities
if and only if every parent cell function is affine.                  (2)
```

Adaptivity and unlimited decoder computation do not change (2), because the
two source assignments in the witness return identical child answers.

## Concrete nonlinear witness

For `f(x_0,x_1)=x_0 AND x_1`, compare the two selected-site assignments

```text
(x,y)   = (00,11)
(x',y') = (01,10).
```

Both give child parity `11`, but their parent parities are respectively `1`
and `0`. No decoder that sees only the two child parities can distinguish
them.

## Exhaustive controls

Every Boolean function was enumerated through arity four.

| Arity | All functions | Affine functions | Two-site factoring functions | Mismatches |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 4 | 0 |
| 2 | 16 | 8 | 8 | 0 |
| 3 | 256 | 16 | 16 | 0 |
| 4 | 65,536 | 32 | 32 | 0 |

The enumeration is only a control; the proof applies at every finite arity.

## What a nonlinear lift must pay

A nonlinear `f` can still be lifted if the representation exposes additional
cross-child correlations. In algebraic-normal form, a degree-two term such as
`x_i x_j` requires the query parity of the coordinatewise product block

```text
X_i AND X_j.
```

Higher-degree monomials require their corresponding Hadamard-product blocks or
another globally non-entrywise source that determines the same correlations.
Those sources contain new checkpoint information and must be charged for
storage, address discovery, probes, and native evaluation. They are not
provided by the child scalar answers.

## Decision

```text
REJECT_FREE_ENTRYWISE_RECURSION_OF_NONLINEAR_PROBE_SEEDS
DO_NOT_PROMOTE_THE_2X3_CAPACITY_SURVIVOR_FROM_LOCAL_EXISTENCE_ALONE
REQUIRE_CHARGED_CORRELATION_SOURCES_OR_A_NON_ENTRYWISE_GLOBAL_ENCODING
KEEP_GLOBAL_CROSS_MATRIX_NONLINEAR_ENCODINGS_OPEN
KEEP_NO_SURVIVING_CANDIDATE
```

This Gate does not prove that the unrestricted `2 x 3` seed is absent. It
shows that finding it would not by itself create a scalable architecture. The
next useful construction must give the global correlation equation first;
another local SAT search has no promotion path.

## Reproduction

```powershell
$env:PYTHONPATH = "."
python scripts/derive_nonlinear_recursive_lift_gate.py `
  --output-dir results/e0_nonlinear_recursive_lift_gate
python -m pytest -q tests/test_nonlinear_recursive_lift_gate.py
```
