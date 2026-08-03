# Experiment 042 — End-to-End Llama Final-Decision Routing Bound

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and fixed objective

This is an E1 formal/executable information certificate. It is not a physical 405B runtime, does not measure GPU wall clock, and does not by itself prove per-token host traffic.

The fixed project objective remains an arbitrary unmodified Hugging Face dense transformer, one 8 GiB GPU, no user training or checkpoint-specific adapter authoring, original-model decisions preserved, and warm decode within 1.2x of a native 4B Q4 baseline.

Experiment 041 proved a 9.5978 GiB metadata-aware exact top-1 lower bound for an independently callable Llama-shaped operator collection. It deliberately did not prove that those operator bits survive actual Transformer composition to the final vocabulary winner. Experiment 042 closes that routing gap with a concrete Llama-style model family.

## Exact architecture family

Every model in the family uses only standard bias-free Llama-style components:

- token embeddings;
- RMSNorm;
- causal grouped-query self-attention;
- residual connections;
- SwiGLU MLP;
- final RMSNorm;
- linear LM head.

Q and K are fixed to zero in the loader attention layers. Therefore RoPE has no effect, causal softmax is an exact uniform average over the legal prefix, and the construction remains compatible with the ordinary Llama attention equations. Loader V/O projections copy control coordinates in chunks no larger than the GQA KV projection dimension. All later attention projections are zero but still execute the same architectural path.

The legal prompt has four tokens:

```text
[layer/group selector token,
 payload-coordinate token,
 output-selector token,
 query/carrier token]
```

The first loader layers move the three selected one-hot controls into the final query position. The remaining variable layers carry independently selectable Q4 codes in `up_proj`.

## Variable Q4 code layout

Let:

```text
L = variable decoder layers
A = selectable neuron groups per layer
B = neurons per group
Q = payload coordinates
```

Each layer uses `A * B` active intermediate neurons. For group `a`, the gate row reads only the layer/group selector. For neuron `b`, the up row contains one independently selectable signed Q4 coefficient for every payload coordinate `j`.

A query `(layer, group, payload, output)` produces:

```text
gate = fixed positive constant
up   = selected signed Q4 level
SwiGLU output = SiLU(gate) * up
```

The down projection writes neuron `b` to code-output coordinate `b`. Unselected layers and groups have gate exactly zero, so `SiLU(0)=0` makes their contribution exactly zero.

Each coefficient takes one of the 16 signed Q4 levels:

```text
{-8, -7, ..., -1, 0, 1, ..., 7}
```

Thus every selected `up_proj` coefficient carries four independent bits.

## Final vocabulary decoder

For each output coordinate `b`, the fixed LM head owns 16 answer rows. Let

```text
c_k = SiLU(gate_constant) * signed_q4_level(k)
```

The row for code `k` computes, up to the common final RMSNorm scale:

```text
baseline(output_selector_b)
+ 2 * c_k * code_output_b
- c_k^2 * carrier
```

At the exact code output `code_output_b = c_t`, this equals a common constant minus `(c_t-c_k)^2`. Therefore answer row `t` is the unique winner. A large output-selector baseline suppresses answer groups for every unselected `b`.

The final next-token winner directly recovers the selected Q4 code. No internal exact tensor is exposed to the evaluator; only legal token IDs and final vocabulary logits are used.

## Function counting theorem

The number of independently selectable Q4 coefficients is:

```text
C = L * A * B * Q
```

The complete legal-prompt decision signature recovers every coefficient, so the family contains:

```text
16^C = 2^(4C)
```

distinct final-token decision functions. Any checkpoint-specific representation that preserves all exact final winners for this family must distinguish every function and therefore requires at least:

```text
4 * L * A * B * Q bits
```

This is metadata-aware. The representation is not assumed to store raw weights.

## GQA loader accounting

The final query position needs these hidden coordinates:

```text
L*A       layer/group selectors
Q         payload selectors
B         output selectors
B         code outputs
1         carrier
```

The source and destination control coordinates are the same hidden dimensions. A loader layer copies at most:

```text
KV_DIM = num_key_value_heads * head_dim
```

controls through V/O. Therefore:

```text
control_coordinates = L*A + Q + B
loader_layers = ceil(control_coordinates / KV_DIM)
variable_layers = total_layers - loader_layers
```

The construction charges those loader layers; they are not available for variable Q4 storage.

## Executable micro-certificate

The checked micro-model uses:

```text
hidden size: 8
attention heads: 4
KV heads: 1
head dimension: 2
KV dimension: 2
total layers: 4
loader layers: 2
variable layers: 2
A = 1
B = 1
Q = 1
```

There are two independently selectable Q4 coefficients and therefore `16^2 = 256` possible checkpoints/functions. The workflow exhaustively instantiates every checkpoint, evaluates both legal layer queries, and requires:

- 256 distinct final-token winner signatures;
- exact recovery of both Q4 codes;
- strictly positive top-1 margin for every query;
- nonzero causal GQA control loading;
- exact zero contribution from an unselected variable layer.

## Llama-405B-shaped projection

The projection uses the established target envelope:

```text
hidden size: 16,384
intermediate size: 53,248
total decoder layers: 126
KV projection dimension: 1,024
vocabulary limit: 128,256
```

An exhaustive integer search under hidden, intermediate, GQA-loader, and vocabulary constraints yields:

```text
loader layers: 15
variable layers: 111
A groups/layer: 31
B neurons/group: 1,717
Q payload coordinates: 9,508
active intermediate neurons/layer: 53,227
control coordinates: 14,666
vocabulary rows: 42,139
```

The variable coefficient count is:

```text
C = 111 * 31 * 1,717 * 9,508
  = 56,175,137,076 coefficients
```

At four bits per coefficient:

```text
metadata lower bound = 224,700,548,304 bits
                     = 26.1585866455 GiB
```

This proves that a complete exact final-decision representation for the constructed family cannot fit entirely inside an 8 GiB resident checkpoint-information allowance.

## Promotion criteria

The certificate advances only if all of the following pass:

1. every executable micro-checkpoint maps to a unique final-token signature;
2. the signature exactly equals the encoded Q4 table;
3. every winner margin is positive;
4. causal GQA loaders are nonzero and respect KV-dimension chunking;
5. unselected variable layers contribute exactly zero;
6. the target projection satisfies hidden, intermediate, loader, and vocabulary limits;
7. the projected complete metadata lower bound exceeds 8 GiB.

## Scope and prohibited overclaims

A passing result proves an end-to-end final-token **total information-size** lower bound for the constructed Llama-style Q4 family. It closes an all-resident exact decision-table path.

It does **not** prove that every token must transfer or inspect all 26.16 GiB. Each legal query selects one coefficient, so an external host-resident indexed table could in principle answer with a sparse random access. The theorem therefore does not yet close:

```text
host-indexed exact metadata
per-token cell-probe complexity
PCIe/CPU latency
metadata construction cost
real autoregressive access locality
```

It also does not perform a real 405B run, measure wall clock, or prove that every released 405B checkpoint has maximum information complexity.

The theorem is a worst-case universality result for representation size: a runtime claiming support for arbitrary exact-decision checkpoints must distinguish this family somewhere in its resident plus external state. The next proof obligation is a cell-probe/communication lower bound for the autoregressive query sequence, or a constructive charged host-indexed runtime.

Do not report metadata size as per-token traffic. The physical 405B/8-GiB/4B-speed runtime objective remains unsolved.
