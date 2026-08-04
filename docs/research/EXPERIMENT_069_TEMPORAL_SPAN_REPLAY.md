# EXP-069 — Causal Exact Temporal-Span Replay Gate

## Question

Can a dense projection avoid rereading `W` by reconstructing a later exact input/output from exact projection pairs already computed for earlier tokens in the same causal session?

For a fixed projection:

```text
y_t = W x_t
```

If a later input satisfies an exact rational relation

```text
x_t = sum_k c_k x_k
```

then exact-real linearity gives

```text
y_t = sum_k c_k y_k
```

without another weight read. This candidate uses online causal history, not static low rank of `W`, activation zeros, KV equality, or a future-token oracle.

## Exact dyadic certificate

Every captured float32 scalar is decoded as its exact IEEE-754 dyadic rational. For each vector, the exact dyadic value is mapped into three registered odd prime fields:

```text
65521
65519
65497
```

Incremental reduced-row-echelon bases are maintained independently in all three fields. If adding a new vector raises rank under any prime, a rational minor remains nonzero and the vector is certainly outside the exact span of all prior vectors. That arrival therefore requires a full `W*x` pass.

A modular non-increase is not called a replay hit. Only a separately verified exact coefficient witness may receive replay credit. The model population credits exact duplicate witnesses only; all other non-increases remain unverified.

## Population

Pinned unchanged checkpoints:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Required prompt families:

```text
English narrative
code
mathematics
identifier boundary
Korean
structured JSON
```

The run generated up to 96 greedy tokens per case and captured up to 94 warm-decode arrivals for every registered `torch.nn.Linear` projection. Generation stopped on EOS. One TinyStories-8M code case emitted EOS as its first token and therefore had no warm trace, which is permitted by the preregistered `or until termination` rule.

## Favorable necessary accounting

The mandatory lower bound charges only arrivals whose independence is mathematically certified. It grants all of the following for free:

```text
all modular non-increase arrivals
all coefficient discovery
all rank metadata and scheduling
all unverified dependency reconstruction
```

Thus the lower bound is more favorable than any deployable replay executor. If certified-independent calls alone exceed the budget, exact witness search and a physical cache cannot rescue the class.

For audit, a fail-closed path is also reported. It performs a full pass for every call except a verified exact duplicate and charges cached-output reads/copies for duplicate replay.

## Controls

- exact two-dimensional recurrence becomes dependent after basis formation;
- duplicate vectors carry an exact coefficient-1 witness;
- an affine midpoint carries exact rational coefficients;
- a triangular sequence adds one new direction per token;
- random dyadic vectors saturate at the ambient dimension;
- all registered primes agree on the controls;
- frozen EXP-061 projection weight hashes match;
- hooked and unhooked greedy token sequences match.

## Authoritative result

Coverage and correctness:

```text
8 EXP-069 tests passed
3 pinned models
18 model/prompt cases
6 required families
147 registered projections
833 warm projection traces
frozen EXP-061 weight-hash mismatches: 0
output token mismatches: 0
registration mismatches: 0
rank/trace/control mismatches: 0
```

Favorable mandatory lower bound:

```text
p50 weight-read fraction: 100%
p90 weight-read fraction: 100%
p50 operation fraction:   100%
p90 operation fraction:   100%
```

Per-model p50:

```text
TinyStories-1M: 69.2439812633%
TinyStories-3M: 100%
TinyStories-8M: 100%
```

Every required family had a p90 mandatory weight fraction of 100%. Across all 833 traces, no exact duplicate input occurred:

```text
verified exact replay hits: 0
maximum case replay-hit fraction: 0%
fail-closed p50/p90 weight fraction: 100% / 100%
```

Even the favorable basis cache was large relative to one copy of the Q4 projection weights:

```text
p50 basis-cache / Q4 projection population: 391.9746782317%
```

This cache ratio is a measured small-checkpoint ratio, not a direct 405B projection.

## Decision

```text
REJECT_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_AS_CORE
RETAIN_DYADIC_RANK_AUDITOR_AUXILIARY
```

The 3M and 8M warm trajectories kept adding independent directions throughout the observed horizon. The 1M model reached dimension saturation in some narrow projections, but the weighted mandatory lower bound still remained about 69.24%, before any coefficient search, cached-vector traffic, output combination, metadata, or kernel overhead.

Do not reopen this path using numerical tolerances, approximate subspaces, longer traces selected after observing results, cross-prompt future dictionaries, or uncharged replay coefficients/cache traffic.

## Authority

```text
results/exp_069/summary.json
results/exp_069/raw/projection_rows.jsonl
results/exp_069/raw/case_rows.jsonl
results/exp_069/raw/registration_rows.jsonl
results/exp_069/raw/control_rows.jsonl
results/exp_069/checksums.sha256
workflow 30922174380
artifact 8897596252
artifact ZIP SHA-256 81e73226e5369a4fb876d3d855f1d1dc69e0a182a7e584b858d4a111a0724247
```

## Claim boundary

Evidence level E1. Exact float32 dyadic modular rank and the resulting mandatory-full-pass lower bound are measured. Complete exact model witness search, bitwise floating-point replay, a physical cache kernel, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain **NOT TESTED**.
