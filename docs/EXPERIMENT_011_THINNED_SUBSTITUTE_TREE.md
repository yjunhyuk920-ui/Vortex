# Experiment 011 — 8 GiB target-derived thinned substitute tree

Evidence level: **E2 causal target-derived draft + optimistic target lower bound**

## Motivation

A full 405B Q4 substitute occupies about 189 GiB and cannot reside on an 8 GiB
GPU. SubSpec-style target-derived drafting can still be tested by retaining the
target tokenizer, embeddings, final norm and LM head while selecting only a
small subset of original decoder layers.

For the committed 405B Llama-like specification, untied Q4 IO weights plus a
1 GiB runtime reserve leave room for at most three full decoder layers:

```text
3 retained layers: fits below 8 GiB
4 retained layers: exceeds 8 GiB
```

No adapter, training, distillation or newly learned parameter is allowed.

## Strong optimistic bias

TinyLlama retains three of 22 layers, while the 405B projection retains only
three of 126 layers. The TinyLlama draft is therefore proportionally more than
five times deeper. Draft latency is also charged as zero, and target verification
is charged only at the Q4 lower bound without Q6/Q8 exact refinement.

Failure is consequently a strong rejection signal. Passing is only permission
to implement the complete progressive verifier.

## Layer-selection strategies

```text
front:   first three layers
uniform: first, middle and final layer
edge:    first two layers and final layer
```

All selected modules are copied directly from the unchanged target checkpoint,
renumbered for causal execution, then quantized to full-rank Q4.

## Protocol

1. Generate the exact TinyLlama continuation before pruning.
2. Select three target decoder layers without training.
3. Retain the original embeddings, norm and LM head.
4. Quantize the retained draft to Q4.
5. Build a depth-12, top-32, beam-64 causal tree under 1024 unique nodes.
6. Measure whether the exact target path remains in the retained tree.
7. Apply the same optimistic Q4 405B target-side cost lower bound.

## Promotion rule

A strategy survives only when:

```text
projected 405B substitute memory <= 8 GiB
exact target path survives all 12 causal levels
Q4 target-side serialized lower bound passes
```

A survivor must subsequently pay:

- real substitute draft latency;
- Q6/Q8 exact progressive verification;
- KV and allocator memory;
- measured 8 GiB GPU wall-clock.

No result from this experiment alone can establish the final runtime claim.
