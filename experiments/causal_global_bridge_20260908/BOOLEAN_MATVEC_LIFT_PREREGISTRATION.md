# Preregistration — deterministic Boolean-MatVec lift gate

Date: 2026-09-09

This is a continuation of frozen Principle B in `PREREGISTRATION.md`, not a new
three-principle round. Principle B asks for a globally paid finite exact
matrix-effect producer rather than an undefined decoder. The specific candidate
below became concrete after the causal/global round was already persisted.

## Candidate being tested

Use a known succinct deterministic Boolean-semiring matrix-vector data structure
as the checkpoint-dependent producer, then transform its exact Boolean OR/AND
answers into the GF(2) parities already used by the nonlinear-router and causal
rank-one gates, and ultimately into native finite-word effects.

The motivating upper bound is Larsen--Williams, *Faster Online Matrix-Vector
Multiplication* (SODA 2017 / arXiv:1605.01695): their systematic cell-probe data
structure keeps the Boolean matrix read-only plus succinct redundancy and has
worst-case Boolean-semiring `Mv` query cost `O(n^(7/4)/sqrt(w))`. Chakraborty,
Kamma and Larsen, *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector
Multiplication* (STOC 2018 / arXiv:1711.04467), explicitly describe this prior
algorithm as deterministic; their faster `~O(n^(3/2))` upper construction is
randomized. These are cell-probe/Boolean-semiring results, not native VORTEX
executors.

## Frozen interface

For one unknown binary matrix `M in {0,1}^{n x n}`, expose only the black-box
Boolean-semiring product oracle

```text
B_M(s)[i] = OR_j (M[i,j] AND s[j])
```

for an adaptively selected binary subset vector `s`.

The lift algorithm is deterministic, receives a target GF(2) vector `v`, may
adapt later Boolean oracle vectors to all earlier returned Boolean product
vectors, and must output

```text
P_M(v)[i] = XOR_j (M[i,j] AND v[j])
```

for every binary `M` and `v`.

All matrix-dependent information in this frozen gate must enter through the
Boolean oracle responses. Inspecting or probing the Boolean data structure's
internal matrix/redundancy directly is explicitly outside this black-box gate;
that becomes a direct GF(2) data-structure candidate and remains a separate open
Principle-B route.

## Frozen strongest adversary

Use target `v = 1^n`. Pick one designated output row.

```text
R0       = 1^n
R_i      = 1^n with coordinate i changed to 0
```

All other rows of `M` are identical across the `n+1` matrices.

Parity of the designated row is opposite between `R0` and every `R_i`.

For any Boolean query subset `S`:

- if `S` is empty, both rows answer `0`;
- if `S` is nonempty and `S != {i}`, both `R0` and `R_i` answer `1`;
- only singleton `S={i}` distinguishes `R0` from `R_i`.

Thus along the transcript on `R0`, an exact deterministic lift appears to need
every singleton query `{i}`. The experiment/proof must either establish this
inductively for adaptive full-vector responses or exhibit a counterexample.

## Frozen decisions

### PROMOTE BLACK-BOX LIFT

Only if a finite deterministic lift is constructed that is exact for every
matrix and uses `o(n)` Boolean-semiring product calls in the worst case, with a
complete transcript proof and paid output/address work.

### REJECT BLACK-BOX LIFT

If the adversary proves at least `n` Boolean-semiring product calls in the worst
case. This rejects only black-box conversion of the Boolean-semiring interface;
it does not reject direct access to/reinterpretation of the data structure's
internal redundancy, arbitrary GF(2) data structures, or native finite-word
producers.

## Required artifacts

- finite adversary/checker implementation;
- exhaustive small-`n` controls over query transcripts or equivalent property
  checks;
- exact theorem and scope statement;
- a result JSON recording the `n`-call lower bound for representative widths;
- update Principle-B frontier/anti-repetition records if the gate closes;
- no 405B/CUDA/latency claim.

## O1--O6 expectation

This gate cannot close O1--O5 by itself. A rejection narrows one producer lift;
O6 may improve only for this scoped theorem. `THEORY_STATUS=NOT_ESTABLISHED` and
`HARDWARE_STATUS=NOT_TESTED` remain frozen unless an independent complete theory
or target run actually changes them.
