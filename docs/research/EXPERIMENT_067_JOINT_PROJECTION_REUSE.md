# EXP-067 — Pinned Real-Q4 Joint Multi-Projection Exact-Reuse Gate

## Question

Do projections evaluated from the same activation contain enough exact shared arithmetic to approach the fixed 405B/8 GiB/4B-class target without changing or retraining the model?

## Execution class

EXP-067 examines complete attention Q/K/V groups and, where present, MLP Gate/Up groups. It does not decompose a single matrix.

For every deterministic Q4 integer row, it measures exact reusable classes under:

```text
row equality
row negation
primitive integer proportionality
exact repeated contiguous row blocks
```

A primitive class may compute one integer dot product and apply an exact per-row multiplier and quantization scale. Hash candidates are reconstructed and byte-verified. Zero rows are counted separately.

## Shared input factor lower bound

Any exact shared transform

```text
z = Bx
W_i x = A_i z
```

requires `rank(B)` to be at least the rank of the vertically stacked projection group. When any group member has full column rank under EXP-058, the stacked rank equals the full input width and no low-width shared input transform exists.

## Population

Pinned unchanged TinyStories-1M/3M/8M revisions. Every 2D Q4 checksum must match the frozen EXP-057 evidence. The preregistered population expects 24 complete Q/K/V groups and no Gate/Up groups because these GPT-Neo checkpoints do not contain gated MLP pairs.

## Favorable accounting

Operations charge one full dot product per primitive class plus one multiplier/scale correction per nonzero output row. Storage charges primitive Q4 rows, original row scales, row-to-class maps, multipliers, and metadata. This deliberately favors survival and does not credit a physical kernel.

## Controls

- duplicate/sign/integer-multiple synthetic rows collapse exactly;
- one-nibble mutation breaks the class;
- random dense rows have at most 1% reusable rows;
- repeated block controls are detected;
- Q4 checksums and EXP-058 rank evidence match;
- malformed groups fail closed.

## Promotion Gate

```text
zero checksum/rank/control/hash mismatch
24/24 complete Q/K/V groups
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 storage fraction <=10%
p90 storage fraction <=25%
p50 common-right rank fraction <=10%
p90 common-right rank fraction <=25%
random reusable-row fraction <=1%
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Before survival, joint kernels, model-wide integration, approximate clustering, learned adapters, unbounded transform searches, and arbitrary linear-circuit synthesis are prohibited.

## Claim boundary

Phase A/B/C weight observation, evidence ceiling E1. Bitwise floating-point replay, a physical joint kernel, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED.
