# Experiment 008 — exact weight-stationary fixed-point blocks

Evidence level: **E1 real-model execution frontier**

## Why the architecture changes

The previous Gate 0 family compressed selected O/down operators into low-rank
session capsules and repaired missing information with exact tiles. Real-model
experiments rejected that family:

- the hot representation lost the exact top-1 token immediately;
- nonuniform rank allocation did not recover unseen continuation decisions;
- causal rolling refresh required exact anchors thousands of tokens apart to fit
  405B traffic, while accuracy already failed at the first token.

The next candidate does not approximate target weights at all.

## Candidate

```text
original 16-bit target weights outside VRAM
        ↓ exact sub-operator tiled stream
K draft positions evaluated together
        ↓ deterministic causal Jacobi update
unchanged prefix = exact greedy certificate
        ↓
commit only certified tokens
```

One target-model pass streams each exact matrix tile once and evaluates all `K`
draft positions as matrix-matrix operations. The target model and tokenizer are
not modified, quantized, trained, distilled, or sparsified.

## Exactness contract

For deterministic greedy decoding, let `F` be the causal parallel token update.
A prefix unchanged between `y` and `F(y)` is the unique autoregressive greedy
prefix:

1. position zero is fixed by the prompt logit;
2. each later unchanged position depends only on the already certified prefix.

No reference continuation is required for the runtime certificate. The exact
reference is used only as an experiment diagnostic.

## Memory contract

The official target specification is 405,849,243,648 parameters at 16-bit,
approximately 756 GiB. A complete layer is too large for safe double buffering,
so the runtime uses exact sub-operator tiles. The Gate 0 memory model charges:

```text
2 × resident exact operator tile
+ full target KV state
+ K-position activation workspace
+ fixed kernel/allocator workspace
<= 8 GiB
```

The default resident tile is 1.5 GiB, giving a 3 GiB double buffer without
changing any weight value.

## Resource correction

Earlier Gate 0 work compared raw target FLOPs directly with a 4B decode FLOP
count. That is too pessimistic for a batched block because native single-token
4B decode is usually memory-bandwidth bound while the target block can use
high-throughput tensor-core GEMM.

This experiment therefore reports a roofline time bound:

```text
T_transfer/pass = original_target_weight_bytes / host_to_device_bandwidth
T_compute/pass  = target_block_FLOPs / effective_tensor_TFLOPS
T_ideal/pass    = max(T_transfer, T_compute)
T_serial/pass   = T_transfer + T_compute
T/token         = passes × T/pass / certified_tokens
```

Both ideal-overlap and conservative serialized bounds must be visible. Passing
the conservative bound and the 8 GiB bound is required for promotion. The
report also emits the minimum tensor throughput and host-link bandwidth needed
at full block commitment, making a hardware impossibility explicit rather than
hiding it in a proxy FLOP ratio.

## Diagnostics for the next stage

Every Jacobi trajectory is also scanned for exact-reference n-grams. This is not
a runtime certificate. It answers a different question: if fixed-point
convergence is too slow, do the trajectories nevertheless contain long correct
fragments that a target-pass verifier could reuse through Lookahead-style
multi-candidate verification and rejection recycling?

## Sweep

- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- block sizes: 16, 32, 64
- initializers: exact prompt-next repetition and prompt-tail repetition
- maximum iterations: 16, 24, 32 respectively
- target projection: 405B dense, original 16-bit weights, exact 1.5 GiB tiles
- device target: 8 GiB
- baseline: native 4B, 4-bit resident decode

## Promotion rule

Advance the fixed-point block decoder only when:

```text
certified prefix matches exact greedy reference
and peak device memory <= 8 GiB
and conservative serialized roofline <= 1.2 × native-4B time
```

If fixed-point certification fails but a long trajectory n-gram is observed,
advance to a weight-stationary multi-trajectory verifier. If both signals are
small, reject training-free Jacobi initialization for this target and change the
draft construction mechanism.
