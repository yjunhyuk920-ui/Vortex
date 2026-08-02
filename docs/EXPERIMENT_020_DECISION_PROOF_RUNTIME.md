# Experiment 020 — Decision-Proof Runtime

Evidence level: **E1 primitive / E2 pretrained LM-head gate**

## Motivation

Previous candidates tried to make every target operation cheap enough to trust
without verification. At the 405B/8 GiB envelope, representations that closed
executed-byte traffic lost token decisions, while accurate representations read
too many bytes.

Decision-Proof Runtime changes the contract:

```text
small hot operator proposes a decision
        ↓
compact residual metadata proves whether omitted information can change it
        ↓
commit only a proven exact decision
        ↓
otherwise refine the proof or read selected residual blocks
```

The first gate isolates the LM head. It uses exact hidden states, a Q4 LM-head
proposal, and row-by-column-block norms of the omitted residual.

## Sound certificate

Let

```text
W = W_hot + R
```

For output row `r`, activation blocks `x_b`, and residual row blocks `R[r,b]`:

```text
|R[r] x| <= sum_b ||R[r,b]||_2 ||x_b||_2
```

For candidate `c`, define:

```text
lower(c) = hot_logit(c) - residual_bound(c)
upper(j) = hot_logit(j) + residual_bound(j)
```

If

```text
lower(c) > max_{j != c} upper(j)
```

then `c` is the exact argmax. The proof requires no residual weight values at
runtime. An accepted wrong token is a correctness bug and fails the gate.

## Metadata projection

For the 405B-class LM head:

```text
rows:          128,256
columns:        16,384
column block:      256
norm entries:  8,208,384
FP16 metadata: < 0.02 GiB
```

The metadata is small enough to remain resident. This does not yet solve the
internal Transformer projections; it tests whether exact output-sensitive
certification is strong enough to justify extending a backward proof through
the model.

## Real-model protocol

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- exact greedy continuation: 32 tokens
- hidden states: exact authoritative prefix
- hot LM head: per-row Q4
- proof metadata: residual row norms at multiple column block sizes
- required invariant: zero unsafe certificates

## Promotion rule

Advance to model-internal bilinear residual proofs only when:

```text
unsafe certificates == 0
certificate rate >= 50%
exact hidden/reference alignment == 100%
```

A low certificate rate rejects only the tested metadata granularity. The next
refinement is a hierarchical norm tree or signed support-function metadata,
not an unverified confidence threshold.
