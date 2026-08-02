# Experiment 016 — semantic block Kronecker runtime

Evidence level: **E2 full real-model linear replacement**

## Why global Kronecker rank was insufficient

The global rearranged-SVD candidate mixed unrelated functional axes:

- separate attention heads;
- GQA query and KV groups;
- MLP neuron regions;
- vocabulary rows.

TinyLlama global rank 2 retained less than 4% top-32 exact-token coverage even
though its 405B rank-16 projection easily passed the hardware envelope.

## Semantic block representation

Every original target matrix remains independent, but its fit is additionally
partitioned along architecture-defined boundaries:

```text
q/k/v:    one row block per attention head
O:        one column block per attention head
Gate/Up:  fixed MLP row blocks
Down:     matching MLP column blocks
LM head:  fixed vocabulary row blocks
```

Each block is represented by one or more balanced Kronecker terms. Blocks are
never shared between layers or semantic groups.

## 405B projected allocation

```text
attention projections: 4 terms per head block
MLP projections:       3 terms per 128-neuron block
LM head:               2 terms per 256-token block
FP8 factors
Q4 embedding table
active 256-token KV state
```

Projected resource envelope:

```text
factor bytes/token:       about 1.746 GiB
total traffic/token:      about 1.919 GiB
factorized compute:       about 183 GFLOP/token
total memory:             about 5.355 GiB
projected latency:        about 5.82 ms/token
1.2x native-4B allowance: about 7.45 ms/token
```

Thus memory, traffic and latency all close before quality is considered.

## TinyLlama upper-bound point

TinyLlama uses:

```text
one term per 64-wide attention-head block
one term per 64-neuron MLP block
one term per 64-token vocabulary block
```

Its factor-elements/original-elements density is recorded and compared with the
405B projected density. The Tiny point is deliberately at least as favorable in
factor density as the target point.

## Protocol

1. Generate exact reference logits and continuation.
2. Q4-quantize only the embedding table.
3. Replace all q/k/v/O, gate/up/down and LM-head linear operations with semantic
   block Kronecker modules.
4. Delete the original linear modules from the active graph.
5. Run teacher-forced token-rank evaluation.
6. Run autonomous greedy generation.
7. Build a causal top-32 tree under a fixed node budget.

No prompt activation, label, gradient, fine-tuning, distillation or learned
correction is used to construct the factors.

## Promotion rule

Advance only if:

```text
405B projected Gate passes
Tiny factor density >= projected target density
teacher-forced exact-token top-32 coverage >= 95%
causal tree preserves at least the first exact target token
```

If the point preserves partial but insufficient token information, the same
factor-byte budget is reallocated nonuniformly between attention, MLP and LM
head blocks. If it loses almost all token information, semantic Kronecker is
rejected and the next structured operator family is butterfly/tensor-train.
