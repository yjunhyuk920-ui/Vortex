# Next Experiment

## Closed Gate — EXP-063

No exact cached Key or Key-Value duplicate occurred in 147,456 measured layer/head rows. Fully accounted work and bytes exceeded dense execution.

```text
REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY
```

## EXP-064 — Pinned Real-Q4 Exact Output-Row Prototype and Sparse-Delta Gate

### Mechanism

For every registered Q4 dense projection, treat each output row as a linear form. Compile three exact candidates:

1. identical-row groups: one dot product plus output copies;
2. sign-canonical groups: one dot product plus exact sign operations;
3. prototype rows plus exact sparse residuals: `w_r = p_g + delta_r`, so `w_r*x = p_g*x + delta_r*x`.

Bias is never folded away: every output bias addition remains charged. No approximation, thresholding, retraining, or changed quantization is allowed.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and the same FP32-to-Q4 extraction/checksum contract as EXP-057..060. Analyze all 153 two-dimensional tensors and promote only the 144 registered dense projections.

### Accounting

Charge prototype and residual weight bytes, row/prototype mappings, residual column indexes and values, activation reads, prototype dot products, sparse residual multiply-adds, output copies/signs, and all bias additions. Compare against packed Q4 dense row evaluation. Report identical/sign group counts, residual density, operation and query-byte fractions, compilation cost, reconstruction checksum, and 405B storage projection.

### Controls

- exact identical and sign-related rows compress;
- sparse-delta rows reconstruct exactly;
- one changed Q4 nibble prevents false identity;
- dense-random and forced-unique adversaries do not compress;
- selected representation round-trips every scalar;
- no runtime lookup table over activations.

### Promotion Gate

```text
zero checksum/reconstruction/control mismatch
all 144 dense projections represented
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-byte fraction <=10%
p90 query-byte fraction <=25%
best dense/unique adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY
```

### Claim boundary

Phase C weight observation only. Q4 output preservation under a physical kernel, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
