# Next Experiment

## Closed Gate — EXP-060

Pinned real-Q4 dense projections contained p50 17.76% exact zeros. Exact row-run streaming retained p50/p90 82.22%/85.06% operations and required 150.93%/200.86% query bytes.

```text
REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY
```

## EXP-061 — Pinned Causal Exact Activation-Sparsity Gate

### Mechanism

For every causal forward pass, an exact-zero input coordinate to a dense projection allows the corresponding weight column to be skipped for every output row. Measure exact IEEE zeros at the input of every registered `torch.nn.Linear`/equivalent learned 2-D projection during:

```text
prompt prefill
first decode token
decode tokens 2..64
```

Causal-attention mask zeros and padding positions are excluded; they are already standard structural sparsity. Only actual projection-input scalar values equal to positive or negative zero count.

### Pinned models and prompts

Use unchanged TinyStories-1M/3M/8M revisions and the pinned GPT-Neo tokenizer from EXP-050. Use the six held-out families: English narrative, Korean, code, mathematics, structured JSON, and identifier boundary. Generate 64 greedy tokens with KV cache for each model/prompt pair.

### Registration

- enumerate every learned 2-D projection module before execution;
- record module name, weight shape/checksum, input feature width, calls, tokens, and phase;
- fail on unhooked or shape-mismatched dense projections;
- deduplicate tied modules only by object identity while preserving named aliases;
- do not count embeddings or causal attention masks as projection-input sparsity.

### Accounting

For input width `n`, output width `m`, and `z` exact-zero input coordinates:

```text
dense operations = m*n
sparse operations = m*(n-z)
weight bytes = Q4 columns for n-z coordinates
activation metadata = nonzero-coordinate indexes + vector row pointer
```

Report operation and query-byte fractions per call, weighted by original dense scalar terms. Charge scanning every activation coordinate to discover zeros as a separate runtime operation count. Selection by prompt, model, module, or token is forbidden; aggregate the full registered population.

### Controls

- injected all-zero vector: zero operation fraction and exact dense fallback equivalence;
- ReLU negative control input: registered exact zeros detected;
- GELU random input: no false zero creation;
- positive-zero and negative-zero counted identically;
- column-skipped mathematical reference equals dense reference for exact-zero coordinates;
- hook registration and call accounting are deterministic;
- greedy committed tokens match an unhooked reference run exactly.

### Promotion Gate

```text
zero output-token mismatch
zero hook/registration/control mismatch
p50 warm-decode operation fraction <=10%
p90 warm-decode operation fraction <=25%
p50 warm-decode query-byte fraction <=10%
p90 warm-decode query-byte fraction <=25%
all six prompt families represented
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY
```

### Claim boundary

Phase C observation only. Actual sparse projection kernels, 405B activation statistics, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
