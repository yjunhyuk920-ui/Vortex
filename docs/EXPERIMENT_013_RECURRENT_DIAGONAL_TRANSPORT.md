# Experiment 013 — recurrent Q4 dictionary with diagonal transport

Evidence level: **E2 weight-only compressed representation + causal tree gate**

## Purpose

A three-layer recurrent dictionary closes the projected 405B memory and raw
compute envelope, but directly reusing one target layer at many depth positions
may destroy model behavior. This experiment adds a position-specific transform
without storing a new matrix:

```text
W_layer ~= diag(output_scale) @ D_representative @ diag(input_scale)
```

`D_representative` is an existing target matrix quantized to Q4. The two scale
vectors are fitted by alternating least squares using only target weights. No
prompt activation, token label, gradient, adapter training or checkpoint change
is used.

## Runtime representation

For each original linear operation:

```text
x' = input_scale * x
y  = output_scale * (D_representative @ x')
```

The 405B projection retains:

- target embeddings and LM head in Q4;
- three Q4 representative decoder layers;
- FP16 input/output scale vectors for every original layer position;
- exact FP16 norm vectors;
- 1 GiB runtime workspace.

The scale and norm metadata is projected below 0.1 GiB, keeping the total below
8 GiB.

## TinyLlama protocol

1. Generate an exact continuation from the unchanged model.
2. Select layers `0, 10, 21` and quantize their matrices to Q4.
3. Assign every original depth position to its nearest representative.
4. Fit two diagonal scale vectors for every 2D layer parameter.
5. Materialize the factorized matrices only for functional testing.
6. Keep original one-dimensional norm parameters exact.
7. Measure weight reconstruction error, teacher-forced token ranks,
   autoregressive exact prefix and a causal top-32 tree.

Materializing all adapted matrices in TinyLlama does not change the projected
runtime memory contract; it is numerically equivalent to applying the shared
matrix with input/output scales.

## Gate

A point survives only when:

```text
representative dictionary + metadata + workspace <= 8 GiB
full-depth recurrent compute <= 1.2x native-4B proxy
causal exact path survives the tested tree depth
Q4 target-side serialized lower bound passes
```

Passing remains necessary but insufficient because exact Q6/Q8 verification,
real shared-weight kernels, KV memory and measured wall-clock remain separate
Gates.

## Next extension

If diagonal transport improves behavior but does not preserve the path, add a
small per-layer low-rank residual:

```text
W_layer ~= diag(a) D diag(b) + P_layer Q_layer.T
```

Rank is increased only while the complete 405B resident representation remains
inside the remaining memory headroom.
