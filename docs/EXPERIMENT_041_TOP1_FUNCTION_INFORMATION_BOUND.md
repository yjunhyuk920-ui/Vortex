# Experiment 041 — Metadata-Aware Exact Top-1 Function Information Bound

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1 formal/executable information gate. It does not run a real 405B model and does not prove end-to-end Transformer wall clock.

Experiment 040 established:

- an arbitrary Q4 405B exact operator output requires 188.98828125 GiB of worst-case checkpoint information;
- every unrepresented weight coordinate can adversarially change exact top-1 in the coordinate-query model;
- a general metadata-aware top-1-only bit lower bound remained open.

Experiment 041 closes part of that gap by constructing an injective family of exact top-1 decision functions. The representation may be arbitrary checkpoint-specific metadata; distinct decision functions still require distinct metadata states.

## Selector/payload classifier family

Consider an `m x d` dense classifier:

```text
logits(x) = W x
winner(x) = argmax_i logits_i(x)
```

Choose:

```text
p = min(floor(m/2), floor(d/2))
q = d - p
```

Use the first `p` input coordinates as selector coordinates and the remaining `q` as payload coordinates. Use `p` disjoint output-row pairs:

```text
pair r = rows (2r, 2r+1)
```

For every pair `r` and payload coordinate `j`, encode one independent bit `a[r,j]`.

Set a selector margin `M > 1`:

```text
W[2r,   r] = M
W[2r+1, r] = M
```

For payload column `p+j`:

```text
if a[r,j] = 0:
    W[2r,   p+j] = 1
    W[2r+1, p+j] = 0
else:
    W[2r,   p+j] = 0
    W[2r+1, p+j] = 1
```

All unused coordinates are zero. Query:

```text
x_(r,j) = e_r + e_(p+j)
```

The selected pair scores `M+1` and `M`; every unselected row scores at most `1`. Therefore the unique top-1 winner is:

```text
2r     when a[r,j] = 0
2r + 1 when a[r,j] = 1
```

Every bit table produces a distinct exact top-1 decision function.

## Information theorem

The family contains:

```text
K = p q independent bits
2^K distinct top-1 functions
```

Any exact checkpoint-specific metadata representation for this family must distinguish every function and therefore requires at least:

```text
I_top1 >= K bits
```

This is metadata-aware for the constructed family. It does not assume the metadata stores raw coordinates or weights.

For an even square `H x H` classifier:

```text
p = H/2
q = H/2
K = H^2/4 = N/4 bits
```

## Llama-405B operator-shape projection

Use the repository target shape:

```text
hidden H = 16,384
intermediate I = 53,248
KV projection output = 1,024
layers L = 126
vocabulary V = 128,256
```

For one matrix shape `(rows, columns)`:

```text
K(rows, columns)
= p (columns - p)
p = min(floor(rows/2), floor(columns/2))
```

Per matrix:

| Matrix | Shape | K bits | MiB |
|---|---:|---:|---:|
| Q | 16,384 x 16,384 | 67,108,864 | 8.0000 |
| K | 1,024 x 16,384 | 8,126,464 | 0.96875 |
| V | 1,024 x 16,384 | 8,126,464 | 0.96875 |
| O | 16,384 x 16,384 | 67,108,864 | 8.0000 |
| gate | 53,248 x 16,384 | 67,108,864 | 8.0000 |
| up | 53,248 x 16,384 | 67,108,864 | 8.0000 |
| down | 16,384 x 53,248 | 369,098,752 | 44.0000 |

Per decoder layer:

```text
K_layer = 653,787,136 bits
        = 77.9375 MiB
```

For 126 independently callable decoder-layer operator collections:

```text
K_layers = 82,377,179,136 bits
         = 9.5899658203125 GiB
```

Adding a directly callable LM head contributes another 8 MiB:

```text
9.5977783203125 GiB
```

This exceeds 8 GiB even though the family encodes only one exact decision bit per selector/payload query, not full Q4 weights.

## Critical scope separation

The experiment must report three different statements:

1. **Direct dense classifier theorem** — proven when exhaustive enumeration verifies all `2^K` bit tables map to distinct top-1 signatures.
2. **Independently callable operator collection projection** — the metadata lower bounds add because the interface permits separate queries to each operator.
3. **End-to-end Transformer final-token theorem** — not proven merely by summing internal operator bounds.

A full Transformer theorem needs an explicit Llama-like routing construction showing that final token decisions can independently expose each layer/operator bit family. Until that construction exists, the 9.59 GiB number is an operator-collection bound, not an end-to-end language-model bound.

## Executable gate

Implement:

- family-shape accounting `K=p(d-p)`;
- deterministic bit-table encoding into weights;
- selector/payload query generation;
- exact top-1 signature extraction;
- exhaustive enumeration for small shapes where `2^K` is tractable;
- verification that the signature exactly equals the encoded bit table;
- verification that every function signature is unique;
- 405B attention/MLP shape projection;
- strict JSON scope fields that forbid interpreting the operator-collection result as a full Transformer theorem.

## Promotion criteria

The certificate passes when:

- every query has one unique winner;
- decoded winner bits equal the encoded table for every enumerated checkpoint;
- distinct bit tables have distinct full decision signatures;
- observed function count equals `2^K` for every test shape;
- metadata lower bound equals `K` bits;
- 405B operator-shape constants reproduce the documented values;
- operator-collection metadata exceeds 8 GiB;
- `full_transformer_top1_bound_proven` remains false.

## Interpretation after a pass

A passing certificate proves:

> Arbitrary checkpoint-specific metadata cannot compress the exact top-1 function of the constructed dense classifier family below `K=p(d-p)` bits. For the independently callable collection of Llama-405B-shaped attention and MLP operators, the additive lower bound is about 9.59 GiB.

It does not yet prove:

> Every end-to-end 405B Transformer final-token decision function requires more than 8 GiB.

The next proof obligation is to embed independently addressable selector/payload families through a Llama-like residual/attention/MLP composition so final token winners expose the layerwise bits without direct internal-operator access.
