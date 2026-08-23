# EXP-104A — Checkpoint-Native Resident Slice p50 Gate

## Question

Can a draft made only from a fully resident, target-derived full-vocabulary head and complete target-width Transformer layers fit in 8 GiB and remain cheap enough to satisfy the same-machine native-4B-Q4 p50 target?

## Three materially different principles screened

1. **Full-vocabulary checkpoint-native resident slice — selected.** Replace an independently trained draft with a full-width slice of the unchanged target, stored only as a draft-side representation. Exact target verification remains unchanged.
2. **Streaming page-shadow self-draft — rejected before implementation.** It tries to create future states while exact pages are transferred, but without a new causal state source it re-enters the already rejected Jacobi/Parareal dependency family.
3. **Deferred mismatch state closure — auxiliary only.** It can move mismatch-token state materialization to the next target sweep, but does not reduce the autoregressive draft scan or increase the accepted prefix.

## Registered 405B inventory

The exact registered non-norm population is recomputed from:

```text
hidden_size       16,384
intermediate_size 53,248
vocab_size        128,256
layers            126
KV heads          8
head dimension    128
```

```text
one layer parameters  3,187,671,040
embedding parameters  2,101,346,304
LM-head parameters    2,101,346,304
registered total    405,849,243,648
```

The input embedding matrix is not resident; exact BF16 embedding rows are charged as an online stream. The untied full LM head is resident because the frozen mechanism performs full-vocabulary drafting.

## Fully charged 8-GiB static ledger

The realistic Q4 representation uses 4-bit payload plus one FP16 scale and one FP16 zero per 128 parameters:

```text
0.5 + 4/128 = 0.53125 bytes/parameter
```

The static ledger also reserves:

```text
exact target KV for 339 tokens  174,956,544 bytes
target weight staging            512 MiB
block activations                 256 MiB
allocator/fragmentation           512 MiB
```

Under this ledger only three complete target layers plus the full Q4 head fit:

```text
resident total at 3 layers  7,717,990,400 bytes
resident total at 4 layers  9,412,829,184 bytes > 8 GiB
```

## Decisive p50 weight-scan floor

A native 4B Q4 baseline has an ideal weight population of:

```text
4B * 4 bits = 2,000,000,000 bytes/token
```

The fixed p50 target permits at most:

```text
1.2 * 2,000,000,000 = 2,400,000,000 bytes/token
```

A nontrivial full-width slice requires the full vocabulary head plus at least one complete target layer. Even an impossible metadata-free Q4 representation requires:

```text
(head + one layer) * 0.5 = 2,644,508,672 bytes/token
ratio to native 4B Q4          = 1.322254336x
```

The realistic group-128 Q4 representation requires:

```text
head                           1,116,340,224 bytes
one complete layer             1,693,450,240 bytes
head + one layer               2,809,790,464 bytes/token
ratio to native 4B Q4          1.404895232x
```

This is the draft alone. It excludes embedding-row traffic, KV/cache traffic, target verification, target sweep, repair, synchronization, and kernel overhead.

Autoregressive acceptance cannot amortize the resident draft scan: every proposed token requires one draft pass. At 100% acceptance the per-committed-token draft scan is still `head + slice`.

Therefore no complete target layer can coexist with a full-vocabulary Q4 head inside the p50 weight-scan allowance.

## Decision

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

The 8-GiB capacity Gate itself passes for a three-layer Q4 slice, but capacity is not latency. The p50 same-machine byte floor rejects the mechanism before public-checkpoint execution.

## Stop rule

Do not reopen this family by changing which complete target layers are selected, increasing configured block length, using another 4-bit group size, or treating resident HBM scans as free.

A new mechanism must do at least one of the following:

1. avoid the full vocabulary-head scan with a causal, fully charged sub-head mechanism;
2. generate more than one useful candidate token per resident weight scan;
3. use a value-changing cross-page/cross-matrix program whose complete online bytes are below the p50 bound.
