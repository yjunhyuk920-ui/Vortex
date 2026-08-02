# Experiment 001 — exact-span Atlas warm-decode rejection

Evidence level: **E1 real-operation falsification**

## Question

Can a rank-32 exact-span `U/WU` capsule, built on one prompt, avoid cold reads during warm decode of a disjoint prompt while preserving exact greedy tokens?

## Model and replacement

```text
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
managed operations: self_attn.o_proj + mlp.down_proj
managed modules: 44
build prompt: English explanation task
evaluation prompt: Korean algorithm task
max rank per projection: 32
generated tokens: 8
warm decode steps: 7
```

The wrapper replaced the actual `nn.Linear` operations during generation. On a span miss it used the original exact weight, so token correctness remained authoritative.

## Result

```text
exact greedy token match: true
prefill fast vectors: 44 / 1,980
warm decode fast vectors: 0 / 308
warm decode exact fallback: 7 reads for every one of 44 modules
rank growth during disjoint evaluation: 23 for every module
```

Logical warm-decode repair:

```text
managed exact bytes read: 9,688,842,240
managed model bytes:       1,384,120,320
full model bytes:          4,400,193,536
full-model repair fraction: 2.2019127479
E = tokens / repair fraction: 3.1790542140
```

Gate 0 requires:

```text
E >= 491.2991599793
```

Shortfall:

```text
491.2991599793 / 3.1790542140 = 154.5426x
```

## Decision

```text
REJECT: exact-span Atlas as the steady-state warm-decode mechanism
```

The capsule reached its rank limit during disjoint prefill. Every subsequent decode activation was outside the cached exact span, so all managed projection weights were read for every token.

Increasing the exact-span rank is not the response: model-wide capsule memory grows linearly with rank and violates the 8 GiB envelope before it reaches the dimensions needed for broad unseen-prompt coverage.

## What remains open

This experiment rejects exact-span membership as the normal-path gate. It does not yet reject:

- approximate capsule execution;
- final-token rather than hidden-state correctness;
- selective exact repair of a small layer/tile subset;
- weight-stationary repair shared by a block of proposed positions.

The next experiment is deliberately optimistic: project all managed operations through the rank-32 capsule, then use an oracle to find the smallest exact layer suffix that restores the original token sequence. If even the oracle cannot reach the Gate 0 repair efficiency, the current approximate capsule architecture is rejected before implementing a runtime repair selector.

## Reproduction

Committed summary:

```text
results/tinyllama_1_1b_exact_span_warm_decode.json
```

Workflow run:

```text
30734726402
```

Artifact digest:

```text
sha256:ae594f61d1b6114a3784138c1eab9d288412177377e0584ddb64a4049ed981e0
```
