# EXP-088A — Cross-Layer Template and Recurrence Code Gate

## Question

Can the complete ordered MLP weight stack be generated exactly from a fixed number of full-shape templates or low-order layer recurrences plus a low-entropy residual?

EXP-087A rejected aligned context **within** one layer. EXP-088A changes the source to cross-layer structure over every official SmolLM2 decoder MLP layer.

## Frozen population

```text
checkpoint       HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2
runtime          official LlamaForCausalLM, BF16/eager
layers           every decoder layer
roles            gate_proj, up_proj, transposed down_proj
forwards         zero
template groups  four contiguous depth groups
```

## Exact generators

For each role independently, the favorable compiler chooses among:

1. `W_l = W_(l-1) XOR R_l`;
2. `W_l = W_(l-1) + R_l mod 2^16`;
3. `W_l = W_(l-1) XOR W_(l-2) XOR R_l`;
4. `W_l = 2W_(l-1)-W_(l-2)+R_l mod 2^16`;
5. one full-shape bit-majority template plus exact XOR residuals;
6. four contiguous full-shape bit-majority templates plus exact XOR residuals.

Every anchor/template and residual stream receives ideal zero-order entropy coding. Decoder, probability tables, framing, addresses, arithmetic coder, and query kernel are free. Each generator must reconstruct every BF16 word exactly.

## Favorable oracle

Gate, up, and down roles may choose different best generators after observing the complete checkpoint. This can only improve the candidate and makes rejection stronger.

## Positive control

A 32-layer synthetic stack made from four repeated templates crosses the 20% information Gate. The language is therefore capable of promotion when its claimed structure exists.

## Gates

```text
all exact reconstructions
model-total information fraction <=20%
effective layer fraction p50 <=20%
effective layer fraction p95 <=25%
```

Decisions:

```text
INVALID_CROSS_LAYER_TEMPLATE_RECONSTRUCTION_FAILURE
REJECT_CROSS_LAYER_TEMPLATE_AND_RECURRENCE_CODE_AS_COLD_QUERY_CORE
PROMOTE_CROSS_LAYER_TEMPLATE_CODE_TO_FUSED_QUERY_GATE
```

## Stop rule

After rejection, do not sweep template count, layer grouping, recurrence order, residual operator, dtype, or thresholds. Reopening requires a generator whose program size is sublinear in both layer count and matrix area, or a causal execution dependency that avoids reconstructing the exact weight field.

## Claim boundary

This is a no-forward checkpoint-description Gate. Even a pass would not establish reduced dense arithmetic, complete operation replacement, exact successor state, CUDA performance, 405B execution, 8-GiB residency, or native-4B-class latency.
