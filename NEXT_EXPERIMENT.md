# Next Experiment

## Closed Gate — EXP-059

All 144 pinned real-Q4 dense projections had full selected shift-displacement rank. Favorable query lower bounds were 100% at p50/p90 and generator storage was 200%.

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

## EXP-060 — Pinned Real-Q4 Exact Zero-Sparsity Streaming Gate

### Mechanism

Measure whether deterministic Q4 matrices contain enough exact zero scalars or completely zero blocks to skip original multiply-adds and weight reads without approximation. Compile and account for:

```text
dense Q4 baseline
scalar CSR
row-wise nonzero-run encoding
BSR 1x4, 1x8, 4x4, 8x8, and 16x16
```

Only exact zeros may be skipped. A nonzero BSR block charges every scalar slot in that block, including internal zeros.

### Pinned population

Use the same TinyStories-1M/3M/8M revisions and all 144 named dense projections. Recompute deterministic row-symmetric Q4 and require exact checksum equality with EXP-057.

### Accounting

- dense operations: `m*n` multiply-add terms;
- CSR/run operations: exact nonzero scalar count;
- BSR operations: scalar slots in nonzero blocks;
- packed Q4 value bytes;
- column/block indexes using the minimum whole-byte width;
- CSR/BSR row pointers;
- run start/length metadata;
- alignment padding and edge blocks;
- all format compile/search work recorded separately.

Select the best format only after all formats are compiled and charged. Report operation and query-byte fractions independently.

### Controls

- highly sparse synthetic matrix must compress below 10%;
- dense-random Q4 matrix must not falsely compress;
- isolated-zero adversary must expose BSR padding waste;
- block-zero positive control must favor its registered BSR shape;
- exact reconstruction from every serialized format;
- row/column permutation changes format statistics but not reconstructed values;
- EXP-057 Q4 checksum agreement.

### Promotion Gate

```text
zero reconstruction/control/checksum mismatch
zero unregistered dense projection
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-byte fraction <=10%
p90 query-byte fraction <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY
```

Phase C observation only. Q4 model-output preservation, physical sparse kernels, actual Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
