# EXP-086A — Causal Functional Microprogram Sharing Gate

## Status before execution

```text
Phase: A/B/C small-real-checkpoint oracle falsification
Evidence ceiling: E1
Checkpoint: HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2
Scientific result: PENDING
```

## Question

Can one automatically compiled, non-state-keyed library of small programs replace the complete nonlinear SwiGLU function on many **unseen causal states** with exact BF16 output?

The compiled function is

```text
F(x) = W_down[SiLU(W_gate x) * (W_up x)].
```

EXP-085A rejected page-separable additive refinement even with an exact-contribution oracle. EXP-086A crosses every gate/up/down page before runtime: each program approximates the complete function directly as

```text
P_j(x) = b_j + U_j(V_j^T(x-a_j)).
```

`P_j` is not a decomposition of any fixed checkpoint matrix. It is a low-rank executable map of the **whole nonlinear function** over a causal-state region. Four programs are generated automatically from build traces. No program contains an exact state key, prefix, KV digest, stored output table, target-forward call, or dense-matmul opcode.

## Frozen population

- six prompt families;
- one build and one disjoint evaluation prompt per family;
- 16 causal positions per prompt;
- 96 build states and 96 evaluation states;
- layer 0 complete MLP input/output pairs from the official BF16 runtime;
- four deterministic input-space clusters;
- rank cap 32;
- FP32 stored program coefficients;
- no post-result cluster, rank, prompt, precision, or layer sweep.

## Compiler

For one build cluster, center the paired matrices `X` and `Y=F(X)`. With

```text
X_c = U S V^T,
```

the minimum-norm retained-rank program is

```text
P(x) = y_bar + ((x-x_bar) V_r) [S_r^-1 U_r^T Y_c].
```

The full retained rank is at most the number of cluster states minus one. FP64 factors are kept only for the numerical positive control. The candidate stores and queries FP32 factors, then rounds its output to BF16.

## Routing arms

### O0 — impossible favorable oracle

Evaluate all four programs and choose the one matching the largest number of exact target output words. This consumes the target output and is not deployable. It is the strongest early falsification of program existence and reuse.

### R0 — causal nearest-centroid router

Normalize the current input and choose the nearest frozen build centroid. It consumes no target output or future token.

Neither arm may select by exact state digest or complete prefix.

## Resource equations

For target hidden width `H`, rank `r`, program count `K`, layers `L`, and scalar bytes `s`, the sidecar is

```text
M = K L s [3H + 2Hr].
```

The three `H` terms are input center, output center, and router centroid. Online coefficient work per layer is

```text
C_router  = H + 3KH,
C_program = 2Hr + H.
```

The frozen 405B projection uses `H=16,384`, `I=53,248`, `L=126`, `K=4`, `r=32`, and FP32 sidecar scalars.

A complete-model success is not claimed because attention and the LM head remain unchanged. The Gate reports that uncompiled fraction explicitly.

## Promotion and rejection

Integrity requires the FP64 build router to reproduce every build BF16 output vector. Resource promotion requires sidecar `<=4 GiB` and compiled-MLP work `<=1.185185%` of the original MLP coefficient work.

The oracle promotion Gate requires:

```text
FP32 evaluation vector exact fraction = 100%
minimum evaluation states per used program >=4
```

If the oracle passes but the causal router fails, only a router Gate is authorized. If both pass, a complete real-MLP replacement Gate is authorized.

Decisions:

```text
INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE
REJECT_CAUSAL_FUNCTIONAL_MICROPROGRAM_RESOURCE_GATE
REJECT_LOW_RANK_FUNCTIONAL_MICROPROGRAM_LIBRARY_AT_ORACLE_GATE
PROMOTE_FUNCTIONAL_MICROPROGRAM_LIBRARY_TO_CAUSAL_ROUTER_GATE
PROMOTE_FUNCTIONAL_MICROPROGRAM_TO_COMPLETE_MLP_REPLACEMENT_GATE
```

## Stop rule

Oracle rejection closes this affine functional-program language at the frozen resource point. Do not rescue it by sweeping clusters, rank, prompts, layer, coefficient precision, or router. Reopening requires a non-affine or whole-transition program language with a new exact information source.

## Claim boundary

Even a pass is E1 small-checkpoint evidence. It does not establish a complete MLP replacement, attention compilation, successor-state equality, 405B execution, 8-GiB residency, CUDA, storage/H2D behavior, or 4B-class latency.
