# Experiment 032 — Partitioned Residual Signed Dual Cone

## Failure addressed

Experiment 031 produced sound intervals but required 100% exact refinement at 4-bit and 97.93% on average at 8-bit. Its dominant looseness came from global residual-dot Cauchy bounds and one global SiLU slope.

This experiment changes the failed bound, not the target.

## Partitioned residual bound

Partition the hidden axis into blocks `B_k`. For quantization residual row `r_i`:

```text
|r_i^T x|
<= sum_k ||r_i[B_k]||_2 ||x[B_k]||_2
<= ||r_i||_2 ||x||_2
```

The first inequality is blockwise Cauchy-Schwarz. The second is Cauchy-Schwarz over block norms. Partitioning can therefore only tighten or equal the global radius.

The same construction covers gate rows, up rows, and down columns.

## Interval-local SiLU slope

For `f(x)=SiLU(x)`:

```text
f'(x) = sigmoid(x) + x sigmoid(x)(1-sigmoid(x))
```

On `[l,u]`:

```text
|f'(x)| <= sigmoid(u) + max(|l|,|u|)/4
```

because `sigmoid(x)(1-sigmoid(x)) <= 1/4`. The implementation takes the minimum of this interval value and the global proven bound `1+1/e`.

## Sound 8-bit norm metadata

Using float32 block norms while accounting them as 8-bit would be invalid. Each row therefore stores an unsigned 8-bit code and one 16-bit row scale:

```text
scale_i = max_k norm_i,k / 255
code_i,k = ceil(norm_i,k / scale_i)
restored_i,k = nextafter(code_i,k * scale_i, +infinity)
```

Thus every restored norm is guaranteed not to understate the original norm. Round-to-nearest metadata is forbidden because it could invalidate the certificate.

For each MLP neuron, store block norms for gate, up, and down-column residuals plus three row scales:

```text
N_norm  = layers * intermediate * 3 * ceil(hidden / block_size)
N_scale = layers * intermediate * 3

M_meta = N_norm * norm_bits / 8
         + N_scale * scale_bits / 8
```

405B projections with 8-bit norms and 16-bit row scales:

| Block size | Hidden blocks | Metadata |
|---:|---:|---:|
| 128 | 128 | 2.4369 GiB |
| 256 | 64 | 1.2372 GiB |
| 512 | 32 | 0.6373 GiB |

The workflow charges metadata and exact refinement separately.

## Real-model gate

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Hot weight precision: 8-bit.

Block sizes: 128, 256, 512.

Protocol:

1. collect one exact cached warm-decode top-two margin trace for English, Korean, code, and mathematics prompts;
2. store only current-token MLP activations and exact output duals;
3. compile each layer's low-bit matrices and upward-quantized block metadata once;
4. reuse the compiled layer across every prompt trace;
5. form sound signed intervals using block-conditioned dot bounds and interval-local SiLU slopes;
6. exact-refine widest intervals until each layer receives at most half the exact margin divided equally across layers;
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

- A passing point advances to causal multi-layer interval transport.
- If even block size 128 remains dense, norm-only residual metadata is rejected. The next representation must preserve signed residual projections, not only residual magnitudes.
