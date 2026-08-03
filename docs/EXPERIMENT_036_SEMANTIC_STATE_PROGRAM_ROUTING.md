# Experiment 036 — Semantic-State Program Routing

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

The objective remains unchanged: execute an arbitrary unmodified 405B-class dense Hugging Face model on one 8 GiB VRAM GPU, preserve original-model quality, require no user training or adapter work, and approach the p50 warm-decode wall clock of a native 4B Q4 model on the same machine.

This experiment is an E1/E2 routing-capacity gate. It does not claim end-to-end execution.

## Why this gate exists

PRs #31–#34 established two facts:

1. signed residual cancellation is real and materially tightens local MLP decision bounds;
2. one static build-prompt program does not transfer: even the best block1024/rank2 code retained about 69.4% activation-perpendicular energy, 81.9% dual-perpendicular energy, and more than 90% exact refinement.

The next question is therefore not another static rank or block sweep. It is whether a small causal semantic-state router can select one of several precompiled signed residual programs so that held-out decode states fall close to the selected activation and output-dual subspaces for multiple consecutive tokens.

Automatic first-run program-bank generation is allowed by `AGENTS.md`. The user does not create, train, or tune the bank.

## Candidate comparison before implementation

### Candidate A — precompiled semantic-state program bank

At first-run format generation, collect model-generated calibration traces and compile `K` signed residual programs. At warm decode token `t`, a causal signature from the previously completed token selects one program before the current MLP weight reads.

Only the active program must reside in VRAM. Other programs may remain in host memory or storage. A program switch transfers one program image.

For block size `b`, rank `r`, coefficient precision `c`, remainder precision `n`, scale precision `s`, and basis precision `u`:

```text
N_blocks = ceil(H / b)
N_rows   = 3 * L * I

M_coeff  = N_rows * N_blocks * r * c / 8
M_rem    = N_rows * N_blocks * n / 8
M_scale  = N_rows * 2 * s / 8
M_basis  = L * 2 * H * r * u / 8
M_prog   = M_coeff + M_rem + M_scale + M_basis
```

The two row scales are one signed-coefficient scale and one upward-safe remainder-norm scale per matrix row. Gate and up share the activation basis; down columns use the dual basis.

For the 405B target, `b=1024`, `c=4`, `n=8`, `s=16`, and `u=8`:

| Rank | Active program size |
|---:|---:|
| 2 | 0.6825 GiB |
| 4 | 0.9901 GiB |
| 8 | 1.6054 GiB |

If `p_switch` is the fraction of decode steps that load a different program:

```text
B_switch/token = p_switch * M_prog
mean run length approximately 1 / p_switch
```

This gate uses a conservative switching allowance of 0.4 GiB/token. Thus the required mean run length is approximately:

```text
A_switch >= M_prog / 0.4 GiB
```

or about 1.71, 2.48, and 4.01 tokens for ranks 2, 4, and 8.

This is only program-transfer traffic. Exact refinement must still satisfy the separate partial MLP traffic gate in a later full residual-code experiment.

### Candidate B — one exact interaction creates a multi-token decision program

An optimistic one-token exact 405B interaction at 4-bit-equivalent source traffic and dense arithmetic costs at least:

```text
B_exact = 405,849,243,648 * 4 / 8 = 188.9883 GiB
C_exact = 2 * 405,849,243,648     = 811.6985 GFLOP
```

Using the existing Cascade Capsule hot estimates and the flagship envelope:

```text
B_total_limit = 2.4 GiB/token
C_total_limit = 9.6 GFLOP/token
B_hot         = 1.2808 GiB/token
C_hot         = 6.3123 GFLOP/token
```

one exact interaction must be reused for at least:

```text
A_traffic >= 188.9883 / (2.4 - 1.2808) = 168.86 tokens
A_compute >= 811.6985 / (9.6 - 6.3123) = 246.89 tokens
```

A standard speculative block forward does not satisfy this compute equation: it evaluates the 405B network once per candidate position, so dense arithmetic remains about 811.7 GFLOP per verified token even when weight traffic is shared.

Candidate B therefore requires a genuinely predictive decision program that commits roughly 247 future tokens from one exact interaction, not ordinary batched verification. No current evidence supports that reuse horizon. Candidate A has the cheaper decisive falsification and is selected first.

## Causal routing contract

The routing signature for decode step `t` is derived only from the final hidden state of the previously completed token `t-1`:

```text
z_t = normalize(R h_final,t-1)
```

`R` is a deterministic resident random projection. It does not use the current token's exact MLP activation, current output dual, future logits, or teacher token.

Spherical farthest-point initialization followed by deterministic Lloyd updates produces `K` build-state centroids. The selected program is:

```text
k_t = argmax_k dot(z_t, centroid_k)
```

The experiment may use exact activations and exact top-one-versus-runner-up output duals to evaluate optimistic subspace coverage after routing. Those exact values are diagnostic oracles and are not available to the router.

## Program-capacity measurement

For every routed state `k`, decoder layer `l`, and hidden-axis block `B_j`, construct activation and dual bases from only build traces assigned to state `k`:

```text
U_x[k,l,j] = top-r right singular vectors of build activations
U_q[k,l,j] = top-r right singular vectors of build output duals
```

For a held-out vector `v`, measure:

```text
rho(v,U) = ||v - U U^T v||_2 / max(||v||_2, epsilon)
```

The gate records activation and dual perpendicular ratios by token, layer, prompt, state, and configuration.

## Disjoint trace protocol

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

- Build prompts and evaluation prompts are disjoint.
- Each prompt produces multiple exact greedy warm-decode steps.
- At each step the router sees only the previous token's final hidden state.
- Exact current MLP inputs and exact current margin duals are retained only for coverage measurement.
- Configurations sweep state count and basis rank while keeping block size 1024 and signature dimension 64.

Initial configurations:

```text
K=4,  rank=2
K=8,  rank=2
K=8,  rank=4
K=16, rank=4
K=16, rank=8
```

## Promotion thresholds

A configuration advances to full signed-residual compilation only if all are true on held-out prompts:

```text
active program metadata <= 2.0 GiB
projected program-switch traffic <= 0.4 GiB/token
mean activation perpendicular ratio <= 0.10
mean dual perpendicular ratio <= 0.10
p95 activation perpendicular ratio <= 0.20
p95 dual perpendicular ratio <= 0.20
no non-finite values or empty routed evaluation states
```

These thresholds are deliberately far tighter than PR #33 because a representation that leaves most state energy perpendicular cannot plausibly reduce exact refinement from above 90% to the required sub-percent range.

A pass is not target completion. It only justifies compiling the full signed residual code and rerunning the exact-refinement certificate with quantized program metadata and actual switch traffic.

## Rejection rule

Reject the tested semantic-state program bank if every memory-compatible configuration fails either:

- the held-out activation/dual coverage thresholds; or
- the program-switch traffic threshold.

Do not respond by increasing static rank, block count, or state count without charging active-program size, switching traffic, bank-generation storage, and causal routing behavior.

## Files and evidence

Planned durable outputs:

```text
vortex_runtime/semantic_program_routing.py
vortex_runtime/multistep_decision_trace.py
scripts/run_semantic_program_routing.py
tests/test_semantic_program_routing.py
experiments/semantic_program_prompts.json
results/tinyllama_1_1b_semantic_program_routing.json
.github/workflows/semantic-program-routing.yml
```

The workflow must commit the raw JSON and post a compact PR decision report.