# Experiment 037 — Prompt-Compiled Hankel Decision Program

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Execute an arbitrary unmodified 405B-class dense Hugging Face model on one 8 GiB VRAM GPU, preserve original-model quality, require no user-authored training or adapters, and reach warm-decode wall clock comparable to a native 4B Q4 model.

This experiment is an E1/E2 predictive-capacity gate. It does not provide a sound end-to-end certificate unless the 247-token horizon gate passes and a later error-bound experiment succeeds.

## Failure addressed

PR #36 rejected precompiled semantic-state program banks:

- the best tested state/rank point left 42.28% activation and 53.91% dual perpendicular energy on average;
- p95 perpendicular energy remained approximately 99.5%/99.8%;
- the selected program changed every 1.36 tokens on average;
- program switching alone cost 0.726 GiB/token.

The missing mechanism is multi-token dynamic evolution. The new program is prompt-specific and updates its state every token instead of selecting another static program.

## Core hypothesis

An exact prompt prefill already produces a sequence of final hidden states:

```text
h_1, h_2, ..., h_n
```

The runtime automatically compiles this prompt trajectory into a low-rank controlled recurrence. No future continuation state or token is used during compilation.

Center and project prompt states:

```text
h_t = mean_h + U z_t + epsilon_t
z_t = U^T (h_t - mean_h)
```

Project the input-token embedding into a separate control basis:

```text
v_t = V^T E[token_t]
```

For Hankel order `p`, the linear program is:

```text
z_(t+1) = Theta^T [z_t, z_(t-1), ..., z_(t-p+1), v_(t+1), 1]
```

The lifted bilinear program adds dynamic terms:

```text
z_t squared
z_t elementwise-multiplied by pad_or_truncate(v_(t+1), rank)
```

These terms model state curvature and token-conditioned transition without evaluating any Transformer layer during warm decode.

## Projected decision head

The program never reconstructs the full hidden vector for token selection. During compilation, project the original LM head onto the prompt basis:

```text
W_program = W_lm U
b_program = W_lm mean_h + b_lm
logits_t  = W_program z_t + b_program
```

The original embedding table is also projected once:

```text
T_control = E V
```

At warm decode the program performs only one recurrence update and one `vocab x rank` projected decision head.

## Causal rollout

The exact prefill supplies:

- the final prompt reduced-state history;
- the exact first generated token from the prompt's final logits.

The program then autonomously predicts every later hidden state and token. It receives only its own previously selected tokens. Exact continuation tokens and hidden states are evaluation oracles.

Teacher-forced rollout is recorded separately by feeding exact future tokens to isolate transition error from autonomous token divergence.

## 405B program budget

For hidden size `H`, vocabulary `V`, state rank `r`, control rank `q`, order `p`, feature dimension `F`, and 8-bit projected tables with 16-bit row scales:

```text
M_state_basis = H * r bytes
M_mean        = 2 * H bytes
M_control     = V * (q + 2) bytes
M_lm_program  = V * (r + 2) bytes
M_theta       = 2 * F * r bytes
M_history     = 2 * p * r bytes
```

Program compute per token is approximately:

```text
C_hot = 2 * V * r + 2 * F * r FLOP
```

Prompt-specific compilation projects the embedding and LM-head tables:

```text
C_build = 2 * V * H * (r + q) FLOP
```

This does not reread or re-evaluate the 405B transformer body. Exact prompt prefill is treated as the normal prefill obligation, not hidden warm-decode work.

For the 405B target (`H=16384`, `V=128256`):

| r | q | p | lift | program | hot compute | build compute | minimum build reuse |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 8 | 1 | linear | 0.00254 GiB | 0.00205 GFLOP/token | 67.24 GFLOP | 7.01 tokens |
| 16 | 8 | 2 | linear | 0.00362 GiB | 0.00411 GFLOP/token | 100.86 GFLOP | 10.52 tokens |
| 16 | 16 | 2 | full | 0.00458 GiB | 0.00411 GFLOP/token | 134.49 GFLOP | 14.02 tokens |
| 32 | 16 | 2 | full | 0.00674 GiB | 0.00822 GFLOP/token | 201.73 GFLOP | 21.04 tokens |
| 32 | 16 | 4 | full | 0.00674 GiB | 0.00822 GFLOP/token | 201.73 GFLOP | 21.04 tokens |
| 64 | 16 | 2 | full | 0.01108 GiB | 0.01645 GFLOP/token | 336.22 GFLOP | 35.09 tokens |

Minimum build reuse uses:

```text
A_build = C_build / (9.6 - C_hot)
```

The program is symbolically memory- and compute-compatible. Predictive horizon is the decisive unknown.

## Why the promotion horizon remains 247 tokens

A generic exact 405B interaction requires approximately 246.89 tokens of compute amortization under the existing flagship envelope. Although this prompt compiler performs less prompt-specific build work than a full exact interaction, the project retains the stronger 247-token autonomous agreement gate. Passing a weaker horizon would not justify abandoning exact-target execution.

## Real-model protocol

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

For every prompt independently:

1. run exact prefill and retain only final-layer hidden states and prompt token IDs;
2. compile each recurrence using only the prompt trajectory;
3. generate 256 exact greedy continuation decisions and exact final hidden states for evaluation;
4. anchor the program with the exact first token from prefill;
5. autonomously generate the next 256 decisions;
6. also run teacher-forced recurrence using exact continuation controls;
7. compare exact token prefix, token agreement, top-32 inclusion, hidden relative error, hidden cosine error, and degeneration diagnostics.

Prompts are long and domain-diverse so the regression has sufficient samples. Exact continuation is temporally disjoint from the prompt build trajectory.

## Configurations

```text
rank=8,  control=8,  order=1, lift=linear
rank=16, control=8,  order=2, lift=linear
rank=16, control=16, order=2, lift=full
rank=32, control=16, order=2, lift=full
rank=32, control=16, order=4, lift=full
rank=64, control=16, order=2, lift=full
```

All fits use ridge regularization selected in advance. No continuation-based hyperparameter tuning is allowed.

## Promotion thresholds

A configuration advances only if every evaluation prompt satisfies:

```text
autonomous exact prefix >= 247 predicted tokens
teacher-forced exact top-1 rate >= 99%
teacher-forced exact top-32 rate = 100%
p95 hidden cosine error <= 0.05
program memory <= 0.25 GiB
C_hot + C_build / 247 <= 9.6 GFLOP/token
no EOS before token 247
maximum identical-token run <= 8
exact continuation unique-token fraction >= 0.10
```

The degeneration conditions prevent a trivial EOS or repeated-token loop from appearing to satisfy the horizon.

## Rejection rule

Reject prompt-compiled low-rank Hankel/bilinear decision programs if no configuration reaches the 247-token autonomous prefix on every prompt.

A failure must not be answered by only increasing rank or recurrence order. Any next candidate must add a sound repair/certificate mechanism or a fundamentally different nonlocal state representation.

## Planned evidence

```text
vortex_runtime/hankel_decision_program.py
vortex_runtime/final_hidden_trace.py
scripts/run_prompt_hankel_decision_program.py
tests/test_hankel_decision_program.py
experiments/hankel_decision_prompts.json
results/tinyllama_1_1b_hankel_<prompt>.json
results/tinyllama_1_1b_hankel_frontier.json
.github/workflows/prompt-hankel-decision-program.yml
```
