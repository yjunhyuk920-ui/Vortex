# E0 Functional-Array-Code Novelty Gate

## Question

Can functional PIR, functional batch, or functional array codes supply the
missing exact cold-backed source for arbitrary Transformer MatVec queries?

These codes store linear forms of an information vector. A requested linear
form is recovered by reading a small recovery set and linearly combining its
symbols. This is exactly the systematic-linear sparse-span model already
bounded in the VORTEX covering audits; it is not arbitrary nonlinear
preprocessing.

Primary definitions:

- Zhang, Yaakobi, and Etzion, *Bounds on the Length of Functional PIR and
  Batch Codes*: <https://arxiv.org/abs/1901.01605>
- Nassar and Yaakobi, *Array Codes for Functional PIR and Batch Codes*:
  <https://arxiv.org/abs/2001.10770>

## Cheapest logic prototype

`scripts/prototype_functional_span_cover.py` exhaustively enumerates every
binary resident linear subspace for all registered toy ambient dimensions up
to six. For each subspace `H` and every input `x`, it finds the smallest raw
coordinate set `P` for which all coefficient vectors needed for the whole
product `W x` lie in `H + span(P)`.

The exact optima include:

| matrix | resident forms | minimum worst-case raw reads | raw coefficients |
|---|---:|---:|---:|
| `2 x 2` | 1 | 2 | 4 |
| `2 x 3` | 1 | 3 | 6 |
| `2 x 3` | 2 | 2 | 6 |
| `3 x 2` | 1 | 4 | 6 |
| `3 x 2` | 3 | 2 | 6 |

The witnesses are ordinary row-local parity/covering constructions. For
example, storing the parity of one three-bit row lets a query read either the
requested subset or its complement, whichever is cheaper. Exhaustive search
found no distinct whole-MatVec cross-row information source in these controls.

The relevant all-linear finite screen is also unfavorable. If every Q4
coefficient is treated as one `GF(16)` symbol and all 8 GiB are granted as
linear field-symbol advice, the hot-symbol rate is
`4.255098118656569%`. The standard q-ary sphere-covering necessary condition
places the corresponding all-linear Hamming-radius root at
`79.09389479630005%`. This number is **not** promoted to a Transformer
rank-one lower bound: the target query family is smaller, global advice may
mix matrices, and native BF16/FP32 arithmetic is not `GF(16)` arithmetic.

## Decision

```text
OMEGA-FUNCTIONALSPAN: REJECTED AT NOVELTY GATE
REASON: LINEAR RECOVERY-SET / COVERING-CODE DUPLICATE
GENERAL NONLINEAR ADAPTIVE RANK-ONE GAP: OPEN
```

Do not build a functional-PIR layout, recovery router, model experiment, or
kernel. Reopening requires a nonlinear, nonliteral decoder with a complete
finite-word query equation, not a new name for stored linear forms and sparse
recovery sets.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe scripts\prototype_functional_span_cover.py
```

This is an E0 mathematical prototype only. It executes no model, checkpoint,
backend, or hardware path.
