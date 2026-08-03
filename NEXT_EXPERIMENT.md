# Next Experiment

## Closed Gate — EXP-062

Warm decode contained only 0.0305% exact non-mask zero probabilities in aggregate. Fully accounted whole-model work and bytes exceeded dense execution.

```text
REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY
```

## EXP-063 — Pinned Causal Exact Cached-KV Equivalence Reuse Gate

### Mechanism

During each warm-decode attention step, inspect causally eligible cached vectors per layer and head:

```text
exact K groups   = bit-identical Key vectors across eligible positions
exact KV groups  = bit-identical (Key, Value) vector pairs
```

For an exact K group, compute the query-key score once and copy it to all members. For an exact KV group, copied scores imply bit-identical softmax probabilities; compute the probability-times-Value vector once and reuse the product while retaining source-order output additions. No approximate similarity, clustering, quantization, or reordered reduction is allowed.

### Pinned population

Use unchanged TinyStories-1M/3M/8M revisions, the pinned tokenizer, all six held-out prompt families, and 64-token KV-cached greedy generation. Compare a standard reference against an observation path that returns `past_key_values`; all 1,152 tokens must match.

### Eligibility

- global layers: all causal cache positions;
- local layers: only the registered local window;
- prefill is recorded separately but promotion is based on warm decode;
- tied or repeated tensor storage is not a vector duplicate unless position vectors have identical dtype, shape, and bit pattern.

### Accounting

For each query/head with eligible length `L`, unique Key count `U_K`, unique KV count `U_KV`, and head width `d`:

```text
dense QK multiplications       = d * L
candidate QK multiplications   = d * U_K
score copies                    = L - U_K
dense Value multiplications    = d * L
candidate Value multiplications= d * U_KV
Value additions                = d * L  (unchanged, source order)
cache equivalence scan/hash     = all K and V scalar bits
mapping/index metadata          = fully charged
softmax                         = unchanged
all Linear/MLP work             = unchanged
```

Report K-only, KV-pair, attention-only, and whole-model fractions. A favorable selector may choose no grouping when metadata exceeds savings, but all scan/hash cost remains charged.

### Controls

- injected duplicate K vectors: QK reuse detected;
- injected duplicate KV pairs: QK and product reuse detected;
- one-bit K or V difference prevents the corresponding group;
- positive and negative floating zero are distinct bit patterns for grouping;
- NaN payloads are rejected;
- group construction invariant to stable position enumeration;
- reference and observation tokens identical.

### Promotion Gate

```text
zero token/registration/control mismatch
all six prompt families represented
p50 whole-model warm operation fraction <=10%
p90 whole-model warm operation fraction <=25%
p50 whole-model warm query-byte fraction <=10%
p90 whole-model warm query-byte fraction <=25%
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY
```

### Claim boundary

Phase C observation only. Physical grouped-attention kernels, 405B KV equivalence statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
