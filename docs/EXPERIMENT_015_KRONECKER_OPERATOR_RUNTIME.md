# Experiment 015 — executable Kronecker operator runtime

Evidence level: **E2 real-model full linear-operation replacement**

## Why this family survives Gate 0

All earlier resident dictionary candidates reduced stored checkpoint size but
still read a full-size matrix at every one of 126 autoregressive layer
positions. Their single-stream HBM lower bound remained about 188 GiB/token.

This candidate changes the executed operator itself:

```text
W ~= sum(k=1..R) A_k tensor_product B_k
```

No full target matrix or full representative matrix is read during decode. Each
term is evaluated as:

```text
Y_k = A_k @ reshape(x) @ B_k.T
Y   = sum_k Y_k
```

Balanced tensor dimensions avoid the degenerate `1 x N` case that would reduce
each term to an ordinary rank-one matrix. A single Kronecker term can therefore
have high ordinary matrix rank.

## 405B rank-64 projection

For the committed Llama-like 405B dimensions:

```text
FP8 Kronecker factor reads: about 2.02 GiB/token
factorized linear compute:  about 658 GFLOP/token
active-256 attention:       about 2.11 GFLOP/token
HBM lower bound @300 GiB/s: about 6.74 ms/token
compute @160 TOPS:          about 4.13 ms/token
allowed 1.2x 4B envelope:   about 7.45 ms/token
```

Memory includes:

- FP8 factors for every decoder linear and LM head;
- Q4 embedding storage;
- active-token KV state;
- exact normalization vectors;
- 1.5 GiB workspace;
- 1 GiB allocator reserve.

The complete projection remains below 8 GiB. Active-token attention quality and
real kernel efficiency remain separate Gates.

## Automatic factor construction

For a matrix with dimensions:

```text
out = o1 * o2
in  = i1 * i2
```

weights are rearranged into:

```text
R[(o1, i1), (o2, i2)]
```

A deterministic randomized SVD of `R` produces the Kronecker factors. The build
uses checkpoint weights only. It is not gradient training, activation
calibration, distillation or an externally learned adapter.

## TinyLlama frontier

TinyLlama ranks are scaled to approximately equal factor-parameter density:

```text
Tiny rank 2 -> projected 405B rank 16
Tiny rank 4 -> projected 405B rank 32
Tiny rank 8 -> projected 405B rank 64
```

For every point:

1. generate exact reference tokens and logits;
2. Q4-quantize the embedding table;
3. replace every `nn.Linear`, including `lm_head`, with executable Kronecker
   factors;
4. discard the original linear modules;
5. run teacher-forced exact-prefix evaluation;
6. run autonomous greedy generation;
7. build a causal top-32 candidate tree.

## Promotion rule

A point advances only when:

```text
projected 405B memory/traffic/latency Gate passes
teacher-forced exact-token top-32 coverage >= 95%
causal tree preserves at least the first exact token
```

The final runtime claim still requires:

- held-out prompts and tasks;
- longer causal path survival;
- active-KV correctness;
- factorized CUDA kernels;
- measured 8 GiB peak VRAM and same-machine 4B baseline;
- quality evaluation beyond token agreement.
