# Experiment 010 — causal Q4 speculative-tree cost lower bound

Evidence level: **E2 causal draft construction + optimistic target-side lower bound**

## Why this gate is mandatory

The progressive-precision experiments showed that Q4 preserves the exact token
inside top-32 on authoritative exact prefixes. That does not make Q4 a causal
candidate generator:

```text
Q4 teacher-forced top-32 coverage: 100%
Q4 autoregressive exact prefix:    0 tokens
```

A deployable runtime must construct future candidates without receiving target
future tokens. This experiment therefore builds an actual causal top-k beam tree
from the Q4 model's own prefixes.

## Strong optimistic assumptions

To avoid rejecting the idea because of an implementation detail, the resource
model gives the candidate mechanism impossible advantages:

- Q4 draft generation costs zero time;
- Q4 draft generation consumes zero VRAM;
- target-side verification is charged only one Q4 full-rank pass;
- Q6/Q8 precision needed to make verification exact is omitted;
- the Q4 weight stream is transferred once per tree;
- all retained tree nodes are evaluated in parallel.

This is not an exact verifier budget. It is a necessary target-side cost lower
bound. Passing only permits the full progressive verifier to be measured next;
failing rejects every more expensive exact verifier for the same tree.

## Candidate protocol

For each point:

1. generate the exact TinyLlama greedy continuation before quantization;
2. quantize all Linear and Embedding weights to full-rank Q4;
3. construct a causal beam tree from Q4's own prefixes;
4. retain the globally highest-scoring branches under a node budget;
5. measure the longest exact target prefix still present in any branch;
6. project one 405B Q4 lower-bound pass over all retained nodes;
7. divide target-side transfer and compute time by the exact contiguous prefix
   that could be committed if a later exact verifier accepted that branch.

The exact continuation is used only for diagnostics. It never influences tree
construction or pruning.

## Resource law

A tree with `N` nodes and depth `D` can commit at most `D` contiguous tokens:

```text
T_lower_bound = transfer(Q4 405B once) + compute_Q4(N nodes)
T/token       = T_lower_bound / committed_prefix
committed_prefix <= D
```

The current 24 GiB/s host-link projection requires thousands of committed tokens
per Q4 stream to approach the 1.2x native-4B envelope. Wide shallow trees may
improve path coverage but increase verification compute without increasing the
maximum possible commitment depth.

## Sweep

```text
top-k  beam width  depth  node budget
8      16          12     256
16     32          12     512
32     64          12     1024
```

## Decision

A point survives this lower-bound gate only if the causal tree preserves the
complete tested exact path and the conservative serialized Q4 lower bound
passes. A survivor must still pass Q6/Q8 exact progressive verification. Failure
prevents teacher-forced precision evidence from being presented as an executable
decode path.

Related external mechanisms used as architectural references:

- SubSpec, NeurIPS 2025, arXiv:2509.18344 — target-derived quantized substitute
  drafts for offloaded models;
- SpecExec, arXiv:2406.02532 — large speculative trees for offloaded consumer
  inference;
- Lookahead Decoding, arXiv:2402.02057 — exact training-free parallel decoding.
