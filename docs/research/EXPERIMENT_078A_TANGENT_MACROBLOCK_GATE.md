# EXP-078A -- Frozen Tangent Macroblock Lifetime and Construction Gate

## Status

Complete. The authoritative E1 result rejects the registered frozen-anchor
reuse path. Source commit `cc368190031d87db92f7a476dd741c18224239c1`;
evidence commit `fe6081917c65b2392f8760d72a2b0e17a4461982`.

## Question

Can the exact input-conditioned MLP mapping induced by one causal anchor token
remain valid long enough on later tokens to amortize its construction and make
the active MLP behave like a 1B-class hot path?

This is throwaway prototype code answering that question. It is not a production
runtime, speed kernel, or target-hardware test.

## Mechanism

For one bias-free SwiGLU path at anchor activation `a`:

```text
c(a) = SiLU(W_gate a)
M(a) = W_down diag(c(a)) W_up
MLP(a) = M(a) a
```

`M(a)` is an exact hidden-by-hidden linear mapping for the anchor. EXP-078A
reuses the same `c(a)` at later causal activations `x_t`, which is algebraically
the same candidate `M(a) x_t`. It does not select channels, replay an old output,
or read future target tokens.

The unchanged target executes the exact prompt and supplies the last prompt
position as the causal anchor. The candidate then reuses all 24 layer anchors
for the seven registered EXP-076 conditioning positions. Candidate recurrent and
KV state is allowed to diverge causally after the exact prefix.

## Why this is not EXP-058, EXP-069, or EXP-077A

- EXP-058 decomposed fixed checkpoint matrices; EXP-078A constructs a transient
  input-conditioned map.
- EXP-069 replayed exact prior activation spans; EXP-078A applies a prior local
  computation law to a new activation.
- EXP-077A omitted individually scored channels; EXP-078A retains the complete
  anchor-conditioned MLP contribution in one composed map.

Only this frozen-coefficient interface is reopened. Approximate temporal
subspaces, future-selected dictionaries, fraction sweeps, and activation-norm
channel rescue remain prohibited.

## Fully charged E0 equation

For hidden width `H`, intermediate width `I`, and `E` active expert paths:

```text
exact active MLP MACs/token       = 3 E H I
hot combined macro MACs/token     = H^2
direct macro materialization MACs = E H^2 I
hot fraction                      = H / (3 E I)
build token-equivalents           = H / 3
charged cycle fraction(L)         = (1 + H/3 + L*hot_fraction) / (L+1)
```

The leading `1` charges one exact anchor token. The direct constructor is an
explicit implementation cost, not a universal algebraic lower bound. A cheaper
constructor would be a new mechanism and must be preregistered with all probes,
metadata, error, and fallback costs.

Registered shapes:

```text
Qwen3.5-0.8B              H=1024, I=3584, E=1
Qwen3.5-122B-A10B screen  H=3072, I=1024, E=9
```

The surrogate's nine paths are eight routed plus one shared expert combined into
one hidden-by-hidden map. This remains Qwen-specific screening and does not
replace the arbitrary dense-405B objective.

## Registered cost and quality Gates

P50 allows at most `0.12` of exact active-MLP work and the p95-tail screen allows
at most `0.15`. Integer minimum reuse spans are derived before the model result.

The seven reuse positions exclude the exact anchor logit. Evaluation promotion
requires all of:

- zero baseline mismatch on all 192 registered EXP-076 target decisions;
- held-out top-1 agreement at least 99%;
- every required family at least 95% top-1;
- held-out mean target-to-candidate KL at most 0.02;
- held-out p95 KL at most 0.05;
- population p50 and p05 valid-prefix lifetime reaching the fully charged
  integer requirements.

A valid prefix ends before the first top-1 mismatch or token KL above `0.05`.
If every case remains valid through position seven, the trace is right-censored
and the result is only `INCONCLUSIVE_TANGENT_LIFETIME_EXTEND_TRACE`. It cannot
promote from seven observations when the cost equation requires more.

## Decisions

```text
PROMOTE_TO_CHARGED_TANGENT_MACROBLOCK_CONSTRUCTION_GATE
INCONCLUSIVE_TANGENT_LIFETIME_EXTEND_TRACE
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
INVALID_TANGENT_MACROBLOCK_CONTROL_FAILURE
```

Scientific rejection stops this exact frozen-anchor reuse form. Do not rescue it
with prompt/layer selection, a longer trace after an observed early failure,
postselected ranks, or uncharged sentinel/fallback work. A delta-updated operator
would be a different dependency and requires a fresh E0 constructor equation.

## Evidence ceiling

Phase C small unchanged checkpoint, causal favorable lifetime observation at E1.
The reference does not physically materialize or accelerate `M(a)`, implement a
sentinel or exact repair, measure hardware, execute 35B/122B/405B, or establish
E2-E7. Structurally valid conditions may be established; large-model performance
remains unverified.

## Authoritative result

The unchanged target replayed all 24 cases and 192 registered decisions with
zero mismatch. On the disjoint 18-case evaluation population, the frozen exact
anchor operator matched `0/126` later-token top-1 decisions. Every case failed
at the first reuse position, so valid-prefix p05/p50/p95 was `0/0/0`.

```text
held-out top-1 agreement             0.0000%
held-out mean KL                    14.898423
held-out p95 KL                     25.335417
MLP relative-L2 p50/p95              0.354193 / 0.511419
0.8B required p50/p05 lifetime       13,821 / 6,249
122B screen required p50/p05         115,299 / 26,354
```

All six prompt families had zero top-1 agreement. The failure occurs before
construction cost, sentinel cost, rank approximation, fallback, and physical
traffic could make the route worse.

Decision:

```text
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
```

This closes only reuse of a complete, unchanged prior-token MLP operator. It
does not reject an operator whose coefficients are causally updated by a new
cheap delta mechanism. Such a mechanism must explain how the delta is computed
without the omitted full gate/weight work and must freeze its construction,
sentinel, repair, cache, and cold-traffic equations before implementation.

Authority: `results/exp_078a/summary.json`; deterministic core SHA-256
`c743ae14748effaad3a034def7d8d92e67fa09abf65eeb931bef3e8a26368667`.
