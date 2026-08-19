# EXP-085A — Joint Exact SwiGLU Compiler Gate

Status before execution: **PREREGISTERED / NOT YET RUN**

## 1. Question

Can a complete real SwiGLU MLP be compiled as one nonlinear function rather than three independently executed matrix products, while preserving the unchanged BF16 output and reducing the fully charged target-equivalent work toward the 405B/4B gate?

The fused function is

```text
F_W(x) = W_down [ SiLU(W_gate x) * (W_up x) ].
```

The v1 compiler partitions only the intermediate-neuron axis into exact macro-pages. One page contains the corresponding `gate_proj` rows, `up_proj` rows, and `down_proj` columns, so its query primitive is the complete nonlinear contribution

```text
f_page(x) = W_down[:, page]
              [ SiLU(W_gate[page, :] x) * (W_up[page, :] x) ].

F_W(x) = sum_page f_page(x).
```

This is not channel dropping. A token/state may use a partial sum only when a deterministic outward BF16 rounding certificate proves that every unread page cannot change any MLP output coordinate. An unresolved query executes the unchanged native MLP and is counted as fallback.

## 2. Mechanism fingerprint

```text
page_separable_fused_swiglu_exact_rounding_refinement
```

Distinctive properties:

- the compiler unit crosses `gate_proj -> SiLU -> multiply -> down_proj`;
- gate, up, and down weights for one intermediate page are materialized together;
- the online state is an exact partial MLP output plus an unread uncertainty set;
- no future generated token, learned router, changed checkpoint, tolerance, or quality threshold is used;
- failure closes only additive page-separable macrofunction refinement, not every globally coupled nonlinear compiler.

## 3. Why this is not a renamed closed mechanism

- EXP-077A selected nonzero channels approximately and failed its quality gate. EXP-085A never commits an approximation.
- EXP-078A reused one frozen prior-token tangent operator. EXP-085A recomputes each selected macro-page from the current exact activation.
- EXP-079A propagated a fixed linear pilot and residual balls. EXP-085A evaluates exact nonlinear page contributions.
- EXP-083A/B selected one residual page inside a projection and propagated uncertainty to the final token. EXP-085A closes or rejects the uncertainty at the complete MLP output boundary.
- The session-basis rounding experiment used `WQ` linear images and failed after SwiGLU. EXP-085A does not approximate the intermediate activation by a linear basis.

## 4. Frozen checkpoints and runtime

```text
DEV-W       HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader      transformers.LlamaForCausalLM.from_pretrained
runtime     torch 2.5.1+cpu / transformers 4.46.3 / safetensors 0.4.5
dtype       BF16
attention   eager
layer       0
page width  8 intermediate channels
```

TARGET-W remains:

```text
meta-llama/Meta-Llama-3.1-405B-Instruct
f9801cba95a53242b3cc928a4a418d12571d1c5f
```

TARGET-W execution, tensor payload, and hardware remain `NOT TESTED`.

## 5. Frozen causal population

Six prompt families are fixed in `experiments/exp_085a/config.json`:

```text
English explanation
Korean reasoning
code generation
structured JSON
mathematics
repeated pattern
```

For every prompt, capture the actual layer-0 MLP input at prefill-last and the next three greedy decode transitions. The registered population is therefore 24 unchanged-checkpoint causal activations. No prompt or position may be removed after observing results.

## 6. Numerical exactness contract

### Gate/up pages

Pages partition only the output-row axis of `gate_proj` and `up_proj`; they do not split either dot-product reduction axis. Every page output must bit-match the corresponding unchanged full projection row.

### Down accumulation

For BF16 values, each BF16 product is exactly representable in FP32. Let

```text
s_j = sum_i W_down[j,i] z_i
A_j = sum_i |W_down[j,i] z_i|.
```

The compiler uses the conservative FP32 accumulation envelope

```text
|fl32(s_j) - s_j| <= gamma_n A_j,
gamma_n = n*u / (1-n*u),  u=2^-24.
```

For a partial exact-real center `c_j` and unread radius `r_j`, an output is certified only when both interval endpoints

```text
c_j - r_j - gamma_n A_j
c_j + r_j + gamma_n A_j
```

round to the same BF16 value and that value equals the unchanged native MLP output. Any open coordinate forces continued refinement or native fallback. False acceptance is a control failure.

## 7. Two registered selectors

### O1 — favorable oracle ceiling

The oracle receives every exact page contribution and every exact absolute contribution bound for free. At each step it may inspect every remaining page and choose the page maximizing newly certified output coordinates. This is non-deployable and can only help the mechanism.

### S1 — sound metadata selector

The deployable reference stores, for every page and output row,

```text
B_page,j = sum_i_in_page |W_down[j,i]| ||W_gate[i,:]||_2 ||W_up[i,:]||_2.
```

For current input `x`,

```text
|f_page(x)_j| <= B_page,j ||x||_2^2
```

because `|SiLU(a)| <= |a|`. Pages are read in descending total bound. The full bound table, page metadata, scan work, and cold payload are charged. The target operation ledger grants one scalar operation per metadata entry and only three output-scalar updates per materialized page; this is deliberately favorable.

## 8. Target resource equations

The registered Llama-405B inventory contains

```text
total parameters                  405,849,243,648
one gate/up/down projection       109,924,319,232
all MLP projections               329,772,957,696
MLP parameter share               81.254996...%
```

If `k` of `P` equal macro-pages are read, the deliberately favorable whole-model weight fraction is

```text
(k/P) * MLP_parameter_share.
```

The final gates are

```text
p50 <= 1.2*4/405 = 1.185185185...%
p95 <= 1.5*4/405 = 1.481481481...%.
```

At TARGET-W shape `H=16,384`, `I=53,248`, 126 layers, page width 8, S1 hot metadata is projected exactly as

```text
126 * (I/8) * H * 8 bytes + page records.
```

It must fit inside the complete 8 GiB allowance; no per-layer reuse of the same bytes is assumed.

## 9. Preregistered promotion gate

All conditions must pass:

```text
official DEV-W load succeeds
gate/up row-split mismatches = 0
metadata bound violations = 0
false accepts = 0
candidate/reference mismatches = 0
O1 fallback rate = 0
S1 fallback rate = 0
O1 p50 whole-model fraction <= 1.185185185%
O1 p95 whole-model fraction <= 1.481481481%
S1 p50 whole-model fraction <= 1.185185185%
S1 p95 whole-model fraction <= 1.481481481%
O1 p50/p95 operation fraction <= 1.185185185% / 1.481481481%
S1 p50/p95 operation fraction <= 1.185185185% / 1.481481481%
projected S1 hot metadata <= 8 GiB
```

Pass:

```text
PROMOTE_JOINT_EXACT_SWIGLU_COMPILER_TO_COMPLETE_LAYER_GATE
```

Scientific rejection:

```text
REJECT_PAGE_SEPARABLE_JOINT_EXACT_SWIGLU_REFINEMENT_AS_CORE
```

Control failure:

```text
INVALID_JOINT_EXACT_SWIGLU_COMPILER_CONTROL_FAILURE
```

Infrastructure failure:

```text
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

## 10. Stop rule

On scientific rejection, do not rescue this fingerprint with a page-width, prompt, layer, page-order, margin, tolerance, dtype, or selector sweep. A continuation must share information across multiple macro-pages through a genuinely nonseparable nonlinear code or change the execution dependency. Another page scheduler is the same mechanism.

## 11. Claim boundary

A pass would authorize only a complete real-MLP replacement Gate. It would not establish a complete layer, successor-state equality, 128-step decode, CUDA, physical I/O, 405B execution, 8 GiB residency, or 4B-class latency.

Required statement regardless of result:

> Structurally valid conditions were established. Large-model performance remains unverified.
