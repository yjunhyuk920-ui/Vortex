# EXP-092A — One-Sweep Block-Span Correction Closure Gate

## Status

```text
FROZEN BEFORE PUBLIC-CHECKPOINT RESULT
Phase: C cheapest decisive oracle correction Gate
Evidence ceiling: E2/E3 small-real-checkpoint rank lower bound
TARGET-W / target hardware / 4B-class latency: NOT TESTED
```

## 1. Why this changes the missing information source

EXP-091A showed that unchanged-target Jacobi relaxation advances the exact prefix only after additional full checkpoint sweeps. At `K=128`, two sweeps already miss the registered logical weight-traffic target.

EXP-092A does not change the seed or add another sweep. It asks whether the first guessed-block sweep already computed a sufficient **linear image basis** to correct the entire block after the true tokens become known.

For one checkpoint projection `W` and its `K` guessed input rows `X_g`, the first sweep caches

```text
Y_g = X_g W^T.
```

If the true incremental-reference input block satisfies

```text
rowspan(X_t) subseteq rowspan(X_g),
```

then an oracle coefficient matrix `C` exists with

```text
X_t = C X_g
Y_t = C Y_g,
```

so the projection's true block image can be reconstructed without rereading `W`. Elementwise nonlinearities, attention normalization, coefficient solving/application, routing, and native-order repair are granted free in this Gate.

The Gate also grants up to eight additional checkpoint-static basis directions per operator input. Thus the frozen necessary condition is

```text
rank([X_g; X_t]) - rank(X_g) <= 8.
```

This is not EXP-086A's copy/secant/quadratic temporal predictor language. It grants the complete `K=128` guessed block as an arbitrary row basis and asks only whether the exact corrected block leaves that span by more than the fixed side basis.

## 2. Public checkpoint and complete boundaries

```text
DEV-W: HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
reference: official transformers 4.46.3 LlamaForCausalLM
ABI: eager BF16 CPU
block length: 128
seed: prompt_suffix_cycle_16, frozen from EXP-091A
layers: 0, 15, 29
```

For each selected complete decoder layer, the runner captures all dominant linear-operator input roles:

```text
attention_qkv_input
attention_o_input
mlp_gate_up_input
mlp_down_input
```

The first, middle, and last layers are a cheapest falsification population. A pass would authorize an all-layer Gate; it is not itself a complete-model acceptance.

## 3. Causal and oracle ordering

For every build and untouched holdout prompt:

1. prefill the exact committed prefix and obtain the exact boundary token;
2. construct the frozen checkpoint-independent seed;
3. execute and hash the guessed block while capturing operator inputs;
4. only after the guessed capture is complete, run the ordinary exact incremental AR continuation while capturing every selected operator input at each of its 128 steps;
5. verify that every operator produced exactly 128 incremental reference rows and that the official token/cache trace is complete;
6. compare guessed complete-block inputs with the stacked true incremental-reference inputs.

The exact incremental block and coefficients are oracle information. They may decide whether the mechanism deserves a causal constructor, but they are not credited as online inputs.

## 4. Exact dyadic rank lower bound

Every captured tensor is BF16. For each selected coordinate, the runner converts all guessed and true words to a common exact dyadic integer scale. No floating tolerance or SVD threshold is used.

The frozen coordinate restriction selects 192 columns deterministically from the operator key. For primes

```text
1,000,003
1,000,033
1,000,037
```

the runner first requires a prime to certify full guessed-row rank:

```text
rank_p(X_g) = K = 128.
```

This equality proves `rank_Q(X_g)=K`, because modular rank cannot exceed rational rank and `X_g` has only `K` rows. For every such certifying prime the runner computes

```text
r_p = rank_p([X_g; X_t]) - K.
```

The certified lower bound is

```text
r_lb = max_{p : rank_p(X_g)=K} r_p.
```

Coordinate restriction and modular reduction cannot make the union rank exceed its full rational rank, while the guessed rank is already exactly `K`. Therefore

```text
r_lb <= rank_Q([X_g;X_t]) - rank_Q(X_g).
```

A prime for which the guessed rank drops is diagnostic only and is not used to certify the lower bound.

If `r_lb > 8`, the full exact correction necessarily needs more than the frozen basis budget. No full-coordinate rank calculation, coefficient solver, backend, or kernel can repair that rejection.

## 5. Resource equation

The favorable correction language gives each of the four operator-input roles at every 405B layer `b` static BF16 basis directions.

With Llama-405B dimensions

```text
hidden = 16,384
intermediate = 53,248
layers = 126
```

the sidecar is

```text
B_side(b)
 = 2 * b * 126 * (3*16,384 + 53,248) bytes.
```

At the frozen budget `b=8`:

```text
B_side(8) = 206,438,400 bytes
extra projection fraction = 8/128 = 6.25%
block projection multiplier = 1.0625x.
```

If the measured lower bound is `r`, even the oracle correction multiplier is at least

```text
1 + r/128.
```

This remains separate from the unresolved base block arithmetic and physical target-machine throughput.

## 6. Frozen decisions

### Rejection

```text
REJECT_ONE_SWEEP_BLOCK_SPAN_CORRECTION_AS_405B_CORE
```

when any untouched holdout operator has a certified lower bound above eight directions.

### Survival

```text
SURVIVES_BLOCK_SPAN_LOWER_BOUND_REQUIRES_FULL_EXACT_COEFFICIENT_GATE
```

only when every integrity control passes and no holdout lower bound exceeds eight. Survival is not promotion: full-coordinate exact ranks, build-derived static bases, causal coefficient construction, native-order equality, and all layers would remain mandatory.

### Invalid

```text
INVALID_BLOCK_SPAN_CORRECTION_CONTROL_FAILURE
```

when any frozen integrity control fails.

## 7. Integrity controls

- official checkpoint and runtime pins match;
- guessed captures complete before ordinary AR target generation;
- every selected incremental-reference operator supplies exactly 128 rows from the official AR decode;
- build and holdout prompts are distinct;
- all 72 layer/role/case reports are present;
- identical-block positive control has rank increment zero;
- an injected 12-direction control certifies exactly 12;
- deterministic coordinate selection is stable and large enough to decide the eight-direction threshold;
- non-finite BF16 values fail closed.

## 8. Stop rule

On rejection, do not sweep:

- seed, block length, prompt subset, layer subset, operator role;
- side-basis width around eight;
- coordinate count, coordinate order, prime, or rank implementation;
- approximate SVD tolerance or low-rank fitting;
- build-selected activation dictionaries that remain linear spans.

Reopening requires a materially different correction source: a nonlinear finite-word symbolic transducer, an exact branch program, or a sound token certificate that does not assert that corrected operator inputs lie in a small linear extension of the guessed-block span.

## 9. Claim boundary

A rejection closes only this one-sweep linear-span correction language. A survival would authorize only an all-layer exact coefficient Gate. TARGET-W, 8-GiB allocation, SSD/PCIe/HBM traffic, CUDA/SASS, base block arithmetic, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
