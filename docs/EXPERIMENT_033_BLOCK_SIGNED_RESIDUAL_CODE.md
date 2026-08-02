# Experiment 033 — Block Signed Residual Code

## Failure addressed

Experiment 032 showed that partitioned residual norms retained almost the full global Cauchy radius. At block size 128, gate/up radii were still about 99.54% and directional radii about 99.87% of the global bound. Magnitudes alone discard the signed cancellation in `r^T x`.

This experiment stores signed residual projections and bounds only the orthogonal remainder.

## Exact decomposition

For residual block `r_b` and an orthonormal shared basis `U_b`:

```text
r_b = U_b U_b^T r_b + r_b,perp
x_b = U_b U_b^T x_b + x_b,perp
```

Orthogonality gives:

```text
r_b^T x_b
= (r_b^T U_b)(U_b^T x_b)
  + r_b,perp^T x_b,perp
```

Store the signed coefficient:

```text
c_b = r_b^T U_b
```

and the remainder norm:

```text
rho_b = ||r_b,perp||_2
```

At runtime:

```text
center_b = c_b^T (U_b^T x_b)
radius_b = rho_b ||x_b,perp||_2
```

Therefore:

```text
r^T x in [sum_b center_b - sum_b radius_b,
          sum_b center_b + sum_b radius_b]
```

This interval is exact on the stored subspace and sound outside it.

## Causal build/evaluation split

The basis is not built from evaluation traces.

- Build prompts: one English systems prompt and one Korean algorithm prompt.
- Evaluation prompts: one code-generation prompt and one mathematics prompt.

For each layer:

- gate/up share a block basis built from build-prompt MLP inputs;
- down columns use a separate block basis built from build-prompt output duals;
- evaluation activations and duals are disjoint.

The first Gate uses exact float32 coefficients, remainder norms, and bases. This intentionally spends more metadata to isolate whether signed projection itself works before adding metadata quantization.

## Metadata budget

For block count `B=ceil(H/block_size)` and rank `r`:

```text
M_rows = L * I * 3 * B * (r + 1) * 32 / 8
M_basis = L * 2 * H * r * 32 / 8
M_total = M_rows + M_basis
```

Gate/up share one activation basis; down uses one dual basis.

405B configurations:

| Block | Rank | Float32 metadata |
|---:|---:|---:|
| 1024 | 1 | 2.4148 GiB |
| 1024 | 2 | 3.6299 GiB |
| 512 | 1 | 4.8219 GiB |

Metadata Gate: at most 6 GiB. Exact original-neuron refinement remains at most 1.6 GiB/token.

## Real-model protocol

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Hot weight precision: 8-bit.

For every configuration:

1. collect exact one-step cached warm-decode MLP activations and top-two output duals for build prompts;
2. build per-layer block SVD bases only from build traces;
3. compile low-bit matrices, signed residual coefficients, and orthogonal remainder norms once per layer;
4. evaluate disjoint prompt traces;
5. form sound SwiGLU signed intervals;
6. exact-refine widest neuron intervals until each layer receives half the exact margin divided equally across layers;
7. project metadata and exact reads to 405B.

## Promotion conditions

Every disjoint evaluation prompt must satisfy:

```text
unsafe certificates = 0
interval containment failures = 0
metadata <= 6 GiB
exact refinement <= 1.6 GiB/token
```

A pass remains an optimistic fixed-dual E2 result. Runtime dual construction, future-token reuse, cross-layer interval transport, attention, LM head, total hot memory, CUDA scheduling, and wall clock remain open.

## Decision rule

- A pass proves that signed residual structure, unlike norm-only metadata, can make exact decision refinement sparse enough to continue.
- If all configurations remain dense, build-prompt block subspaces do not transfer. Static signed residual codes are rejected and the next Gate must make the codebook online, semantic-state keyed, or directly token-decision keyed.
