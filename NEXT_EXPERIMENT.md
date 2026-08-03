# Next Experiment

## Closed Gate — EXP-061

No exact positive or negative zero was observed in 56,448 projection calls. Mandatory zero discovery made warm-decode logical work and bytes exceed dense execution.

```text
REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY
```

## EXP-062 — Pinned Causal Exact Non-Mask Attention-Probability Sparsity Gate

### Mechanism

Request exact attention probabilities during causal generation and measure entries equal to positive or negative zero only after excluding positions that are zero solely because of causal or padding masks. An exact zero probability permits skipping the corresponding Value-vector multiply/add for that query/head without changing output.

### Pinned population

Use unchanged TinyStories-1M/3M/8M revisions, the pinned GPT-Neo tokenizer, six held-out prompt families, prompt prefill, first decode, and 64-token KV-cached generation. Run a standard reference and an `output_attentions=True` observation path; all committed tokens must match.

### Registration

- enumerate every attention layer and head count;
- record query/key/value lengths, unmasked entry population, exact non-mask zero count, phase, token, model, prompt family, and layer;
- fail on missing attention tensors, shape mismatch, NaN, negative probability, or row-sum violation beyond the pinned numerical tolerance;
- causal-mask and padding zeros are excluded from both numerator and eligible population.

### Accounting

For each head/query with key length `L`, charge:

```text
QK score terms               = head_dim * L
softmax terms                = L
Value dense terms            = head_dim * L
Value sparse terms           = head_dim * nonzero_probability_count
probability zero scan        = L
nonzero-key indexes/pointers = exact metadata bytes
```

Total Transformer accounting must also include all unchanged dense-projection and MLP terms from the registered architecture. Report both attention-only and whole-model operation/query-byte fractions. Skipping QK or softmax is forbidden because zero status is known only afterward.

### Controls

- explicit masked logits: mask zeros excluded;
- extreme unmasked logits that underflow: exact zeros detected;
- moderate logits: no false zeros;
- positive/negative zero equivalence;
- dense versus zero-skipped Value accumulation equality in fixed scalar order;
- probability rows finite, nonnegative, and normalized;
- reference and observation generation tokens identical.

### Promotion Gate

```text
zero token/registration/control mismatch
all six families represented
p50 whole-model warm-decode operation fraction <=10%
p90 whole-model warm-decode operation fraction <=25%
p50 whole-model query-byte fraction <=10%
p90 whole-model query-byte fraction <=25%
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY
```

### Claim boundary

Phase C observation only. Physical attention-sparse kernels, 405B attention statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
