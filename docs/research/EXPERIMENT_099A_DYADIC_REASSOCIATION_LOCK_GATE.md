# EXP-099A — Dyadic Reassociation / Native-Order Lock Gate

## Motivation

EXP-098A rejected Q4 prediction plus exact BF16 row repair: even with a perfect
match oracle, untouched-holdout original-BF16 repair remained about 99%. That
failure says a low-bit approximate value is too far from the official BF16
projection; it does not answer a different question required by exact fast
matrix multiplication.

Fast bilinear algorithms change algebraic association. For BF16 inputs, the
mathematical real dot product can be computed by another exact arithmetic
program, but the official ABI may differ because its FP32 accumulation order is
not associative. EXP-099A asks whether this ABI difference itself is sparse at
the final BF16 word boundary.

## Execution principle

For each original projection row and current activation:

```text
s_real = sum_j W_bf16[j] * x_bf16[j]
b      = round_BF16(s_real)
y_ref  = official BF16 projection word
```

The experiment evaluates `s_real` in FP64 and encloses FP64 reduction error by a
conservative gamma bound. It accepts `b` as the exact-real BF16 word only when
the complete interval lies strictly inside one BF16 rounding cell. An impossible
oracle then reveals whether that rigorously determined exact-real word equals
the official word.

Only rows that fail either condition require native-order repair. The favorable
repair fraction is weighted by the original row dot-product width.

This is the missing ABI screen for a future explicit rectangular fast-matrix
multiplication executor. It does not itself implement that algorithm.

## Secondary diagnostic

A much wider generic FP32 dot-product error enclosure is also tested. This is
not used for promotion because the target GPU reduction tree is not frozen in
this CPU Gate. A row whose entire generic FP32 envelope fits the same BF16 cell
is nevertheless a strong order-independent lock signal.

## Frozen public-checkpoint population

- `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- official BF16 eager path
- complete layers `0,15,29`
- all dense projection roles `q/k/v/o`, `gate/up/down`
- 3 build + 3 untouched holdout prompts
- 4 exact greedy causal calls per prompt
- 504 projection observations
- every native replay must match the hooked official BF16 word exactly

## First 10x Gate

Promotion requires:

```text
holdout native-order repair p50 <= 10%
holdout native-order repair p95 <= 10%
every projection role repair p95 <= 25%
zero integrity failures
```

Decision labels:

```text
PROMOTE_DYADIC_REASSOCIATION_LOCK_TO_EXPLICIT_RECTANGULAR_FMM_GATE
REJECT_REASSOCIATED_EXACT_DOT_NATIVE_ORDER_REPAIR_AS_10X_CORE
INVALID_DYADIC_REASSOCIATION_LOCK_CONTROL_FAILURE
```

## Relation to EXP-080A

EXP-080A rejected standard recursive Strassen: even at K=16,384 its constructive
arithmetic fraction was about 36.11%, although the non-constructive omega oracle
was far below the target. Its stop rule explicitly leaves an explicit exact
algorithm that closes the constant gap plus a new causal block source open.

EXP-099A does not reopen standard Strassen or enlarge K. It tests a different
necessary condition: whether a future explicit rectangular bilinear scheme can
reassociate exact dyadic products without forcing almost every row back through
the official native reduction order.

## Three-principle frontier

This round compares three premise flips:

1. **Dyadic reassociation lock** — change the arithmetic primitive rather than
   omit weights; use explicit rectangular bilinear multiplication and sparse
   native-order repair. EXP-099A is its cheapest Gate.
2. **Backward margin dual witness** — start from a candidate winner and prove the
   margin through a sparse exact backward witness rather than materializing the
   complete forward state. Existing absolute/spectral proof balls are excluded;
   this remains a later candidate only if it provides a new correlated witness.
3. **Session dynamic-span saturation** — cache exact projection images of an
   online activation basis and stop consulting weights after new states enter
   the span. Existing EXP-092A/086A evidence makes this lower priority because
   128-state blocks already exhibit large independent-rank growth and the
   favorable state-axis operation fraction stayed near 20%, above the first 10x
   Gate.

The dyadic reassociation route is selected because it is universal for arbitrary
Dense weights and its first ABI question is directly falsifiable without a
405B download or another predictor family.

## If it promotes

The next round must search/construct an **explicit**, small-coefficient,
rectangular bilinear scheme for actual Llama projection aspect ratios and a
causally available block. Every multiply, pre/post addition, packing operation,
workspace byte, candidate node `N`, committed token `A`, weight byte, and native
repair row is charged. A non-constructive matrix-multiplication exponent is not
evidence.

## Stop rule

On failure do not tune layers, prompts, positions, BF16 cell tolerance, or Gate
thresholds. A continuation must change the arithmetic representation itself or
introduce a new correlated decision proof. On promotion do not claim 405B
feasibility until an explicit scheme and causal block source close both
`r*(N/A)` and the transport equation.

## Claim boundary

This is an E1 small-real-checkpoint arithmetic-ABI Gate. No 405B checkpoint,
8-GiB physical allocation, CUDA/SASS fast bilinear kernel, causal long block, or
same-machine 4B latency is tested here.
