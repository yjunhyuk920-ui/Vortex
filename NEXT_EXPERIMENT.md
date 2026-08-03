# Next Experiment

## Closed Gate — EXP-056

Authority: `results/exp_056/summary.json`; workflow `30823042599`; source head `73655fc216340d9bd1d452d779951c28ac1b3d3b`; artifact `8859665874`; ZIP SHA-256 `9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8`.

Exact prototype-residual plans were correct, but p50/p90 logical work was 62.5%/131.25%, p50/p90 bytes 62.115%/169.643%, and dense/unique p50 123.4375%. Decision:

```text
REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY
```

## EXP-057 — Pinned Real-Checkpoint Weight-Structure Extraction Gate

### Why this changes the evidence class

EXP-055 and EXP-056 found exact savings only when weight columns truly repeat or differ by very sparse exact residuals. Continuing with invented matrices would not answer whether public Transformer weights contain that structure. EXP-057 therefore moves from synthetic construction to Phase C observation on unchanged pinned checkpoints.

### Pinned models

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

### Matrix scope

Enumerate every 2-D learned weight tensor used by Transformer linear/embedding projections. Record model, module path, shape, dtype, parameter count, and checksum. Biases and 1-D normalization weights are excluded from column-structure claims but remain in the manifest.

### Representations

1. exact stored floating-point bit patterns;
2. deterministic symmetric per-output-row Q8;
3. deterministic symmetric per-output-row Q4.

Quantization is an execution representation only; checkpoints remain unchanged. Scale, zero handling, clipping, packing, and dequantization error are recorded separately. No quality or output-preservation claim is made by this structural Gate.

### Analyses

For each matrix and representation:

- exact identical and sign-canonical column groups;
- exact group coverage and largest group;
- EXP-055 logical operation/byte fraction;
- EXP-056 frequency/greedy prototype counts 1/2/4/8;
- exact residual scalar/column density;
- best fully accounted logical operation/byte/storage fraction;
- compile search and amortization;
- layer/type/model-size trends.

### Controls

- shuffled-column order control, which must preserve structure counts;
- element-permuted adversary, which should destroy column structure;
- synthetic repeated and sparse-residual positive controls;
- exact reconstruction checks for every compiled plan;
- checksum-pinned model and tensor manifests.

### Early Gate

Promotion to an actual small-model operation-replacement kernel requires all of:

```text
zero reconstruction mismatch
zero unregistered tensors
real-matrix p50 operations <=10%, p90 <=25%
real-matrix p50 bytes <=10%, p90 <=25%
no model-size degradation beyond 25%
projected storage <=1 TiB
compile amortization <=1,000,000 queries
```

Failure decision:

```text
REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY
```

### Claim boundary

Phase C observation at most. It does not execute a replacement Transformer operation and does not test 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, or tokens/sec.
