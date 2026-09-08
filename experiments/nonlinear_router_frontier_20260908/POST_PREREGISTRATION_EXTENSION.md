# Post-preregistration theorem extension

The preregistration selected globally nonlinear bounded-word routing and first
registered a weaker attack in which the first adaptive word was affine.

During the proof, the affine assumption became unnecessary. This file records
that strengthening rather than rewriting the preregistration after seeing the
result.

Let the binary checkpoint source be `V = F2^D`. Let every stored cell be an
arbitrary function `E_j : V -> {0,1}^w`. Let a deterministic exact query
algorithm make `t` adaptive cell probes for a linear parity query `q`.

For one query group sharing the same next address, choose the largest value
fiber of that word. It retains at least a `2^-w` fraction of the source set.
Repeat this for the first `t-1` probes. For every final route group the retained
source set `F` therefore satisfies

```text
|F| >= 2^(D-(t-1)w).
```

All query answers in that final group factor through one final `w`-bit word,
so their joint answer vector assumes at most `2^w` patterns on `F`.

Let `L` be the binary linear span of the query coefficient vectors in that
group and `r=dim L`. Projection onto a basis of `L` is a rank-`r` linear map on
`V`; every global projection fiber has exactly `2^(D-r)` source points. Hence

```text
|F| <= 2^w * 2^(D-r).
```

Combining the two inequalities gives

```text
r <= t*w.
```

At every probe depth the fixed prior values leave at most `S` possible next
addresses, so the complete query family is covered by at most `S^t` linear
subspaces of dimension at most `t*w`.

This argument uses none of the following:

- linear or affine stored cells;
- systematic storage;
- balanced fibers;
- nonadaptive second/later addresses;
- a linear decoder;
- a fixed path independent of checkpoint values.

It is still scoped to deterministic exact bounded-word access to one binary
source block. It does not cover global cross-matrix synergy, randomized-error
decoders, native Q4/BF16/FP32 arithmetic, a 32-token physical union, or the
whole VORTEX causal state machine.

The exact Segre intersection maximum used after this reduction is the binary
simplex-product generalized-weight result already present in the repository.
For normalized `m<=n` and subspace dimension `d=q*n+r`, `0<=r<n`, it is

```text
M(m,n,d) = (2^q - 1)(2^n - 1) + 2^q(2^r - 1).
```

The implementation retains the independent chained-product deficit DP and
checks the closed form against the existing exact product-code controls.
