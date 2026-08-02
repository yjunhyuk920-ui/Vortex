# Experiment 009 — full-rank progressive precision

Evidence level: **E1 real-model information-retention frontier**

## Fixed objective

The target remains an arbitrary existing Hugging Face dense model, including a
405B-class 16-bit checkpoint, on an 8 GiB GPU with no user training or manual
model-specific rewrite, while preserving declared model quality and approaching
native 4B Q4 warm-decode latency.

## Why this family is different

The rejected Atlas families reduced an activation/operator domain to a small
rank. TinyLlama experiments showed that unseen continuation decisions left that
span immediately. Increasing, mixing, or reallocating rank did not repair the
lost token information.

This experiment removes no direction:

```text
W16 = Qb(W16) + Rb
```

- `Qb` is a full-rank, per-output-row coarse representation used by the hot path.
- `Rb` denotes the exact remaining source precision retained outside the hot
  representation.
- the checkpoint file is unchanged;
- no row, column, neuron, layer, or singular direction is removed;
- no retraining, calibration dataset, or model-specific parameter tuning is
  used.

The initial experiment fake-quantizes `Linear` and `Embedding` weights only to
measure information retention portably. It is not yet an exact residual
certificate and does not claim the target is solved.

## Protocol

For each precision `b ∈ {2,3,4,6,8}`:

1. Load the same TinyLlama 1.1B checkpoint in float32.
2. Generate 32 exact greedy continuation tokens.
3. Evaluate exact logits on the authoritative exact prefix.
4. Replace every unique Linear/Embedding weight by a per-row symmetric
   full-rank `b`-bit fake-quantized value.
5. Re-evaluate the same authoritative prefix.
6. Measure exact-token rank, top-K coverage, logit error, and the first
   divergence without allowing autoregressive error propagation.
7. Separately run the coarse model autoregressively to measure cascade damage.
8. Project the corresponding 405B coarse stream onto 64–4096 token blocks.

Tied embedding/LM-head storage is transformed once. Quantization is processed
in bounded row chunks to avoid materializing a second full model.

## Decision metrics

```text
teacher-forced exact top-1 match
exact-token coverage at K = 1,2,4,8,16,32,64,128,256
autoregressive exact prefix and match rate
first-divergence exact-token rank
full-rank hot stream GiB
minimum block required by transfer and low-bit compute rooflines
```

A precision becomes a hot-path candidate only when:

```text
teacher-forced top-1 >= 90%
top-32 coverage >= 99%
and some tested block can meet the ideal 1.2× native-4B roofline
```

This is deliberately not sufficient for exact execution. A promoted precision
must next receive a causal residual certificate or a progressive refinement
mechanism that reads additional exact bitplanes only for uncertain decisions.

## Falsification rule

If even 8-bit full-rank execution does not preserve exact tokens in a small
candidate set, the problem is not low-rank information loss alone and this
family is rejected.

If 4–8 bit paths retain high top-K coverage but not top-1, the next experiment
will test multi-precision consensus and token/layer-local residual refinement.
If 2–4 bit top-1 is already high, prioritize a streamed low-bit hot kernel plus
exact residual repair budgeting.
