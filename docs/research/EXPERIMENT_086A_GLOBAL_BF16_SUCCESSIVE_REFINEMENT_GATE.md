# EXP-086A — Global BF16 Successive-Refinement Oracle Gate

## Status

```text
Date preregistered: 2026-08-19 Asia/Seoul
Status: SOURCE AND GATE FROZEN BEFORE OFFICIAL-CHECKPOINT EXECUTION
Phase: A/B plus small-public-checkpoint favorable-oracle observation
Evidence ceiling: E1
```

## Question

Can a checkpoint-static, globally significance-ordered BF16 representation avoid
reading most low-significance weight information for an actual causal SwiGLU
query while preserving the complete unchanged BF16 MLP output bit-for-bit?

EXP-085A partitioned the complete SwiGLU by intermediate-neuron macro-pages.
Even an impossible oracle had to read all 192 pages on every registered
activation. The failure indicates that page-local absolute enclosures destroy
cross-page signed cancellation.

EXP-086A changes the refinement unit. It never selects an independent neuron
page. Every weight is represented as:

```text
sign | exponent | mantissa bit 6 | ... | mantissa bit 0
```

A query first sees sign/exponent and a global most-significant mantissa prefix.
Lower mantissa information is progressively added. Candidate output is accepted
only when the complete SwiGLU output is bitwise identical to the unchanged BF16
reference.

## Mechanism fingerprint

```text
global_bf16_sign_exponent_mantissa_successive_refinement
+ one global gate/up precision per causal activation
+ independently refinable down-projection output rows
+ zero-order entropy as a favorable information lower estimate
```

The row-adaptive down arm is deliberately more favorable than a physical stream:
it may choose the minimum down mantissa prefix separately for each output row
after observing the unchanged reference output. Selection, indexing, entropy
coder state, random access, decompression, kernel work, and all non-MLP model
traffic are free.

A failure rejects this fixed MSB-prefix information source before a packed
kernel is built. It does not reject a context-modelled code, nonlinear
cross-weight generator, or different causal execution dependency.

## Exact BF16 prefix

For `k in 0..7`, `Prefix_k(w)` preserves the sign bit, eight exponent bits, and
`k` most-significant mantissa bits, clearing the remaining bits.

```text
Prefix_7(w) = w bit-for-bit
```

There is no rounding, retraining, stochastic quantization, or weight update.

For one causal activation `x`:

```text
g_k = Prefix_k(W_gate) x
u_k = Prefix_k(W_up) x
z_k = SiLU(g_k) * u_k
y_{k,r} = Prefix_r(W_down) z_k
```

Reference:

```text
y* = W_down [SiLU(W_gate x) * (W_up x)]
```

A candidate is exact only when every BF16 output bit equals `y*`.

## Favorable oracles

### O1 — global uniform precision

Choose one `k` for gate/up and one `r` for all down rows. Among exact pairs,
select the pair with minimum measured zero-order prefix entropy.

### O2 — global gate/up plus row-adaptive down

Choose one gate/up precision `k`. For each down output row `j`, choose the first
`r_j` whose complete candidate row equals `y*_j`. Select `k` and `{r_j}` with
minimum measured zero-order prefix entropy.

O2 is authoritative and non-deployable because it uses the reference output.

## Frozen public population

```text
checkpoint  HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader      transformers.LlamaForCausalLM.from_pretrained
dtype       BF16
attention   eager
layer       0
families    English, Korean, code, structured JSON, mathematics, repeated pattern
positions   four causal positions per family
population  24 actual checkpoint activations
```

No future generated target token is read by an earlier query.

## Information accounting

Raw prefix length is `9+k` bits per weight. The runner also measures empirical
zero-order entropy of each prefix symbol population, granting ideal entropy
coding with zero codebook, framing, address, random-access, and decode overhead.

Let `rho(x)` be O2's prefix-entropy fraction relative to the full BF16
zero-order entropy of the three MLP matrices. Granting every non-MLP parameter
and every other runtime cost for free:

```text
A50(x) = ceil((MLP_parameter_share * rho(x)) / (1.2 * 4 / 405))
A95(x) = ceil((MLP_parameter_share * rho(x)) / (1.5 * 4 / 405))
```

This is a necessary traffic screen only. Standard dense `F.linear` still
executes all scalar MACs.

## Pre-registered Gate

Integrity:

```text
Prefix_7 identity mismatch = 0
full-precision candidate/reference mismatch = 0
O2 exact output mismatch = 0
checkpoint/runtime/config identity mismatch = 0
```

Promotion to a separate packed-kernel Gate requires:

```text
O2 entropy fraction p50 <= 20%
O2 entropy fraction p95 <= 25%
O2 minimum perfect acceptance p50 <= 16
O2 minimum perfect acceptance p95 <= 32
```

The information thresholds require at least a 5x/4x reduction before physical
costs. They are frozen because current strict speculative evidence is on the
order of tens rather than hundreds of accepted tokens.

Pass:

```text
PROMOTE_GLOBAL_BF16_SUCCESSIVE_REFINEMENT_TO_PACKED_KERNEL_GATE
```

Failure:

```text
REJECT_GLOBAL_BF16_MANTISSA_PREFIX_AS_COLD_QUERY_CORE
```

Control failure:

```text
INVALID_GLOBAL_BF16_SUCCESSIVE_REFINEMENT_CONTROL_FAILURE
```

## Stop rule

After rejection, do not tune mantissa ordering, prompts, layer, dtype,
thresholds, or row policy. Do not call a different bitplane order a new
mechanism. Reopening requires a cross-weight context model, nonlinear exact
generator with charged residual, a non-prefix causal code, or a different
state/time execution dependency.

## Claim boundary

A pass authorizes only a packed bit-serial or fused decompression/compute Gate.
It does not establish:

```text
standard-ISA operation saving
complete Transformer-layer replacement
successor-state equality
128-step generation
405B execution
8 GiB peak VRAM
target SSD/PCIe traffic
same-machine native-4B p50/p95
E2-E7
```

## Reproduction

```bash
python -m pytest -q tests/exp_086a
python experiments/exp_086a/run_experiment.py \
  --config experiments/exp_086a/config.json \
  --output-dir results/exp_086a/local
```
