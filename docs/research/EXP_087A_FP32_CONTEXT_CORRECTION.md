# EXP-087A FP32 predicate-semantics correction

## Why the first hosted result is not the final scientific verdict

The first hosted source `eda4310e8ad20cf38ab712c46a9ef7b386882c62` compiled each context predicate from an FP64 mean and FP64 direction, then stored only FP32 mean/basis tensors. The executable program recomputed the sign predicates from the stored FP32 tensors.

Therefore the compiler solved

```text
phi(sign((x - mean64) @ basis64)) -> BF16 residual words
```

while the deployed program executed

```text
phi(sign((x - mean32) @ basis32)) -> BF16 residual words.
```

A near-zero projection can change sign across these two expressions. This explains why one program had nominal feature rank 24 but did not replay its own 24 build outputs. The synthetic GF(2) solver control did not cover this compiler/runtime predicate boundary.

## Frozen correction

The correction changes no scientific capacity parameter:

- same public checkpoint and immutable revision;
- same layer-0 complete SwiGLU boundary;
- same 96 build and 96 evaluation states;
- same balanced four-program partition;
- same 12 predicates selected from the same 64-direction pool;
- same degree-two, 79-feature GF(2) language;
- same FP32 sidecar byte equation;
- same oracle/router and promotion thresholds;
- same prohibition on state keys, target calls, dense fallback, page repair, and prefix tables.

It changes only compiler consistency: candidate directions are generated as before, quantized to the charged FP32 sidecar representation, and all feature selection, GF(2) fitting, and runtime replay use those exact FP32 tensors. A fail-closed invariant checks that returned build bits equal deployed runtime bits.

## Decision policy

The result commit `4a23f49cd6e5354e8616eb930814e11db95cd933` remains preserved as evidence of the discovered control failure but is superseded for mechanism verdict purposes by the corrected hosted run.

The corrected run is still decisive even if build interpolation becomes exact: the frozen target-seeing oracle must generate all 576 BF16 output words exactly on every one of 96 disjoint evaluation states. Failure closes the degree-two finite-context fingerprint under the original stop rule.
