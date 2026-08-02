# Experiment 031 — Signed Dual Cone for SwiGLU

## Hypothesis

Previous exact-neuron oracles attempted to preserve the full MLP output vector. The current token decision only depends on a small number of logit-margin directions. A signed directional interval may therefore require far fewer exact original neurons than vector-output reconstruction.

This experiment tests the most optimistic local version of that claim on real TinyLlama warm-decode steps.

## Local operator

For one SwiGLU MLP:

```text
g_i = w^g_i x
u_i = w^u_i x
a_i = SiLU(g_i) u_i
y = D a
```

For a fixed output dual `q`, neuron `i` contributes the scalar:

```text
c_i = a_i (d_i^T q)
q^T y = sum_i c_i
```

The workflow obtains `q` from the exact top-one versus runner-up decode margin. This is an optimistic oracle input and is not yet a runtime selector.

## Sound low-bit interval

Let low-bit gate/up/down approximations produce centers:

```text
ĝ_i, û_i, â_i, ŝ_i
```

and certified radii:

```text
alpha_g_i = ||w^g_i - ŵ^g_i||_2 ||x||_2
alpha_u_i = ||w^u_i - ŵ^u_i||_2 ||x||_2
beta_i    = ||d_i - d̂_i||_2 ||q||_2
```

A global SiLU Lipschitz constant is used:

```text
L_silu = 1 + 1/e
```

because `|x sigmoid(x)(1-sigmoid(x))| <= 1/e`.

The activation-product radius is:

```text
alpha_i = L_silu alpha_g_i (|û_i| + alpha_u_i)
          + |SiLU(ĝ_i)| alpha_u_i
```

Thus:

```text
a_i in [â_i - alpha_i, â_i + alpha_i]
s_i in [ŝ_i - beta_i, ŝ_i + beta_i]
```

The implementation forms the exact interval hull of the four endpoint products rather than only a symmetric error radius. Summing per-neuron intervals gives a sound local interval for `q^T y`.

A simpler valid bound, retained for interpretation, is:

```text
|c_i - â_i ŝ_i|
<= |â_i| beta_i + |ŝ_i| alpha_i + alpha_i beta_i
```

## Exact refinement

All exact original-neuron reads have equal weight cost: one gate row, one up row, and one down column. The runtime prototype refines neurons in descending interval width.

After refining neuron `i`, its approximate interval is replaced by the exact scalar `c_i`. Refinement stops when:

1. the local scalar sign is certified; or
2. the remaining uncertainty is below the layer's allocated share of the exact top-two margin.

The margin-share experiment assigns:

```text
tau_layer = margin_share * exact_top2_margin / number_of_layers
```

with `margin_share = 0.5`.

## 405B exact-read accounting

For selected fraction `f`, the projected exact MLP refinement traffic is:

```text
B_exact(f)
= layers * ceil(intermediate * f)
  * (gate_row + up_row + down_column)
  * source_bits / 8

= L * ceil(I f) * 3H * source_bits / 8
```

The partial Gate is:

```text
B_exact <= 1.6 GiB/token
```

This does not include a complete hot-path solution. Promotion only means the exact residual reads are sparse enough to justify implementing multi-layer dual-interval transport.

## Real-model protocol

Model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

For each English, Korean, code, and mathematics prompt:

1. run exact prefill without gradients;
2. take one greedy token;
3. execute one cached warm-decode step with gradients;
4. form the exact top-one versus runner-up margin;
5. capture every current-token MLP input and output dual;
6. build 4-bit or 8-bit signed intervals;
7. measure sign-only and equal-margin-share exact refinement counts;
8. verify that every exact local scalar lies inside its interval.

## Promotion conditions

A precision point advances only if all prompts satisfy:

```text
unsafe certificates = 0
interval failures = 0
all layer margin-share targets close
projected 405B exact refinement traffic <= 1.6 GiB/token
```

A pass remains E1/E2 optimistic evidence. Exact-dual availability, dual drift through later nonlinear layers, attention, LM-head proof, hot-state memory, CUDA scheduling, and end-to-end wall clock remain open.

## Rejection meaning

If 8-bit intervals still require a large exact fraction, then signed decision projection does not solve the density problem and the family is rejected.

If 8-bit passes but 4-bit fails, the next gate is residual-aware mixed-precision hot storage allocated by interval width per byte.
