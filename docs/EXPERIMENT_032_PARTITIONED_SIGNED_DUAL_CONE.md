# Experiment 032 — Partitioned Residual Signed Dual Cone

## Failure addressed

Experiment 031 produced sound intervals but required 100% exact refinement at 4-bit and 97.93% on average at 8-bit. The dominant looseness came from:

```text
|r_i^T x| <= ||r_i||_2 ||x||_2
```

and one global SiLU Lipschitz constant.

This experiment changes the bound, not the target.

## Partitioned residual bound

Partition the hidden axis into blocks `B_k`. For a quantization residual row `r_i`:

```text
|r_i^T x|
<= sum_k ||r_i[B_k]||_2 ||x[B_k]||_2
<= ||r_i||_2 ||x||_2
```

The first inequality is sound by blockwise Cauchy-Schwarz. The second follows by Cauchy-Schwarz over block norms, so partitioning can only tighten or equal the global radius.

The same bound is used for:

- gate rows against the current activation;
- up rows against the current activation;
- down columns against the current output dual.

## Interval-local SiLU slope

For `f(x)=SiLU(x)`:

```text
f'(x) = sigmoid(x) + x sigmoid(x)(1-sigmoid(x))
```

On interval `[l,u]`:

```text
|f'(x)|
<= sigmoid(u) + max(|l|,|u|)/4
```

because `sigmoid(x)(1-sigmoid(x)) <= 1/4`. The implementation takes the minimum of this interval bound and the global proven bound `1+1/e`.

## Metadata budget

For each MLP neuron and hidden-axis block, store one residual norm for gate, up, and down-column residuals:

```text
N_meta = layers * intermediate * 3 * ceil(hidden / block_size)
M_meta = N_meta * metadata_bits / 8
```

For the 405B target with 8-bit metadata:

| Block size | Hidden blocks | Metadata |
|---:|---:|---:|
| 128 | 128 | about 2.400 GiB |
| 256 | 64 | about 1.200 GiB |
| 512 | 32 | about 0.600 GiB |

The workflow charges metadata and exact refinement separately.

## Real-model gate

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Precision: 8-bit hot approximation.

Block sizes: 128, 256, 512.

Protocol:

1. collect one exact cached warm-decode top-two margin trace for English, Korean, code, and mathematics prompts;
2. store only current-token MLP activations and exact output duals;
3. compile each layer's low-bit matrices and block residual metadata once;
4. reuse the compiled layer across all prompt traces;
5. form sound signed intervals using block-conditioned dot bounds and interval-local SiLU slopes;
6. exact-refine widest neuron intervals until each layer receives at most half the exact margin divided equally across layers;
7. project exact reads and metadata to the 405B target.

## Promotion conditions

Every prompt must satisfy:

```text
unsafe certificates = 0
interval containment failures = 0
exact refinement traffic <= 1.6 GiB/token
metadata <= 2.5 GiB
```

A pass remains an optimistic local-dual result. Runtime dual construction, multi-layer dual drift, attention, LM head, total hot memory, physical scheduling, and wall clock remain unresolved.

## Decision rule

- If block refinement lowers exact reads by orders of magnitude and a point passes, continue to causal multi-layer interval transport.
- If even block size 128 remains dense, reject norm-only residual metadata. The next representation must preserve signed residual projections, not only residual magnitudes.
