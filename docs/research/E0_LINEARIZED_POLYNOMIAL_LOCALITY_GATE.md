# E0 Linearized-Polynomial Locality Gate

## Verdict

```text
PROMOTE LINEARIZED POLYNOMIAL AS AN EXACT REPRESENTATION IDENTITY
REJECT THE REPRESENTATION ITSELF AS A LOCAL EVALUATOR
REJECT IMPORTING A DENSE ORDINARY-POLYNOMIAL DATA STRUCTURE
CHANGE MECHANISM CLASS
NO SURVIVING CANDIDATE
```

This is the cheapest novelty Gate. No checkpoint, model, download, backend,
server, or hardware action occurred.

## Elementary explanation

Imagine writing the same large phone book in a different alphabet. The cover
looks different, but every phone number is still inside. A linearized
polynomial does exactly this to a binary matrix.

For a `16,384 x 16,384` matrix, it stores 16,384 special coefficients. Each
coefficient itself contains 16,384 bits. Therefore it stores

```text
16,384 x 16,384 bits,
```

exactly the original matrix size. The published fast algorithms can organize
computation better, but they still receive this complete coefficient list.

## 1. Exact identity

Identify an `n`-bit vector with one element of `GF(2^n)`. Every binary linear
map then has one unique linearized polynomial

```text
L_W(x) = sum_(i=0)^(n-1) a_i x^(2^i).
```

The evaluation map from these polynomials to binary linear maps is a
bijection. This is useful algebra, but not compression: `n` coefficients times
`n` base bits per coefficient is still `n^2` bits.

Puchinger and Wachter-Zeh explicitly state the bijection and give
subquadratic algorithms for linearized-polynomial operations and multipoint
evaluation. Their algorithms take the explicit coefficient list as input;
they do not provide a preprocessed low-cell-probe evaluator for an arbitrary
fixed linear map. Primary source: [Fast Operations on Linearized Polynomials
and their Applications in Coding Theory](https://arxiv.org/abs/1512.06520).

## 2. One necessary I/O calculation

Give the candidate every favorable assumption: keep only one bit per one of
the 405,849,243,648 registered parameters, read it only once for all 32
queries, use 32 GB/s, and make all computation and other traffic free.

```text
50,731,155,456 bytes / 32 GB/s / 32
  = 49.542144 ms/token.
```

The target is 20 ms/token. Thus even this one-bit coefficient sweep is
`2.4771072x` too slow. Reading the registered 551.22 GB exact compressed
checkpoint once per 32-query block is `538.30078125 ms/token`, before any
compute.

This rejects the direct coefficient-list evaluator. It does not prove that
every possible preprocessed data structure must read every coefficient.

## 3. Why the general polynomial data structure does not rescue it

A tempting result gives near-linear space and sublinear query complexity for
ordinary polynomial evaluation over suitable finite fields. Its size
parameter is the dense ordinary degree/number of coefficients. Here the top
ordinary exponent is

```text
2^(n-1) = 2^16,383,
```

not `n`. Feeding this sparse special polynomial into the dense interface
would require more than `2^16,383` extension-field coefficient slots. That is
not near the `n^2`-bit source. Primary source: [Fast, Algebraic Multivariate
Multipoint Evaluation in Small Characteristic and
Applications](https://arxiv.org/abs/2111.07572).

The only remaining version would be a new, specialized near-source-size data
structure that evaluates an arbitrary linearized polynomial with sublinear
physical probes. Because the evaluation map is a bijection onto every binary
matrix, that object is exactly the original arbitrary preprocessed MatVec
problem under another name. It is not a new constructor.

## 4. Claim boundary

This Gate proves only:

- the representation preserves all matrix information;
- published coefficient-list algorithms fail the registered full-sweep I/O
  floor;
- a dense ordinary-polynomial data structure has the wrong exponential degree
  parameter.

It does not reject all nonlinear/adaptive preprocessing and does not lift
binary field arithmetic to reference BF16/FP32 accumulation. The route is
closed because it adds no new locality mechanism, not because a universal
cell-probe impossibility theorem was proved.

## 5. Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_linearized_polynomial_locality_gate.py `
  --output-dir results\e0_linearized_polynomial_locality_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_linearized_polynomial_locality_gate -v
```
