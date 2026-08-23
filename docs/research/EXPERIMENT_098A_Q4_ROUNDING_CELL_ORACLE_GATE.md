# EXP-098A — Q4 Predictor / BF16 Rounding-Cell Oracle Gate

## Why this experiment exists

EXP-097A removed the strongest apparent residual clue: the same target-seeing
minimum-L1 margin LP certified untouched targets even when its K=128 bank came
from another prompt or when the bank's vocabulary columns were permuted. The
residual capacity result remains mathematically true, but it is no longer
model-specific evidence for reducing the dense arithmetic fraction `r`.

The next core round therefore changes the verification unit instead of fitting a
better token-space residual program.

## Hidden premise being reversed

The ordinary exact path performs a complete BF16 dense projection and only then
uses the result. EXP-098A asks whether a much cheaper checkpoint-derived Q4
projection can propose the output first, while the original BF16 projection is
used only to repair output coordinates whose final BF16 word would differ.

For one official projection with input `x`:

```text
y* = BF16(W_bf16 x)
yq = BF16(Q4(W_bf16) x)
```

For this first Gate an impossible perfect oracle reveals exactly which output
coordinates already satisfy:

```text
word_bf16(yq_i) == word_bf16(y*_i)
```

Only mismatched coordinates are charged an original BF16 row dot product.
Therefore the favorable original-BF16 arithmetic fraction is

```text
r_oracle =
  sum_i 1[word(yq_i) != word(y*_i)] * input_width_i
  --------------------------------------------------
  sum_i output_rows_i * input_width_i
```

This is an upper-bound feasibility Gate. The Q4 proposal work, selector proof,
attention/nonlinear work, storage, and kernel overhead are not credited toward
final performance; a pass only authorizes the next sound certificate Gate.

## Why this is not the rejected ROUNDLOCK experiment

F-066 used an Atlas-plus-one-page predictor and found that a free exact-word
oracle still left 99.9023% of the frozen down-projection coordinates requiring
repair. That result prohibits tuning the same predictor, but its stop rule
explicitly leaves a materially different predictor or globally coded repair
source open.

EXP-098A changes the predictor information source to the complete rowwise
symmetric Q4 projection of each original matrix. It also tests all seven dense
projection roles over complete layers rather than one Atlas row. If this much
stronger favorable predictor still leaves more than 10% original BF16 work,
there is no reason to build a sound rounding-cell certificate for this family.

## Public-checkpoint Gate

Frozen DEV-W:

```text
HuggingFaceTB/SmolLM2-135M
revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
transformers 4.46.3
BF16 eager official path
```

Population:

```text
layers      0, 15, 29
roles       q/k/v/o, gate/up/down
prompts     3 build + 3 untouched holdout
positions   4 exact greedy causal calls per prompt
rows        6 * 4 * 3 * 7 = 504 projection observations
```

The hook first replays every official projection with the exact original module
and requires byte-identical BF16 output. Only then is the Q4 predictor compared.

## First 10x promotion Gate

A necessary signal for this predictor/repair family is:

```text
untouched-holdout weighted original-BF16 repair fraction p50 <= 0.10
untouched-holdout weighted original-BF16 repair fraction p95 <= 0.10
no individual projection role has holdout repair p95 > 0.25
zero exact-replay integrity mismatch
```

Decisions:

```text
SURVIVES_Q4_ORACLE_ROW_LOCK_REQUIRES_SOUND_ROUNDING_CELL_GATE
REJECT_Q4_ROUNDING_CELL_ROW_REPAIR_AS_10X_CORE
INVALID_Q4_ROUNDING_CELL_ORACLE_CONTROL_FAILURE
```

## If it survives

The next experiment must remove the perfect oracle. For every proposed BF16
word it must compute a sound interval for the original BF16 dot product from
checkpoint-derived quantization residual information and prove that the entire
interval lies inside the same BF16 rounding cell as `yq_i`. Only proven rows may
skip the original BF16 dot product. Q4 bytes/MACs, interval work, metadata,
repairs, fallback, native reduction order, exact successor state, and the
speculative-offload dual roofline must all be charged.

## Stop rule

On rejection do not sweep Q4 bit range, layers, prompts, positions, equality
tolerance, or the 10%/25% thresholds. Reopening requires a materially different
predictor information source or a repair unit that shares exact original work
across many output rows. On survival, do not interpret oracle equality as a
runtime until the sound rounding-cell proof and complete cost ledger pass.

## Claim boundary

This Gate is E1 small-real-checkpoint evidence only. It does not execute 405B,
validate 8-GiB residency, construct a sound skip certificate, implement a packed
Q4+BF16 repair kernel, preserve the full Transformer successor state under that
kernel, or establish same-machine 4B-class p50/p95.
