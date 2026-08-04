# EXP-067 — Pinned Real-Q4 Joint Multi-Projection Exact-Reuse Gate

## Status

Closed at evidence level E1.

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Machine-readable authority:

```text
results/exp_067/summary.json
results/exp_067/evidence_manifest.json
results/exp_067/checksums.sha256
workflow 30915776126
artifact 8894994162
artifact ZIP SHA-256 e4ce3078b338cbdc410fdcd8265704d7d3eb49f606438ba0cf89f9824597ce05
```

## Question

Do projections evaluated from the same activation contain enough exact shared arithmetic to approach the fixed 405B/8 GiB/4B-class target without changing or retraining the model?

## Method

EXP-067 examined all complete attention Q/K/V groups in the pinned TinyStories-1M/3M/8M Q4 population. These GPT-Neo checkpoints contain no gated MLP Gate/Up pairs.

For every Q4 integer row it built byte-verified classes under exact equality, negation, and primitive integer proportionality. One class is allowed one integer dot product plus per-row multiplier and scale corrections. It also checked exact repeated row blocks at sizes 4, 8, and 16.

For an exact shared input transform

```text
z = Bx
W_i x = A_i z
```

EXP-058 full-column-rank evidence gives a lower bound on `rank(B)`. If any group member has full column rank, the vertically stacked group also has full input-width rank.

## Correctness and coverage

```text
EXP-067 tests: 6 passed
2D tensors checksum-verified: 153
complete Q/K/V groups: 24/24
incomplete groups: 0
analyzed Q/K/V rows: 10,752
checksum mismatches: 0
rank-evidence mismatches: 0
hash-collision mismatches: 0
control failures: 0
```

Synthetic duplicate/sign/integer-multiple rows collapsed correctly. A one-nibble mutation reduced reuse. Random dense rows had 0% reuse.

## Authoritative result

```text
exact reusable rows: 0 / 10,752
maximum group reusable-row fraction: 0%
operation p50: 100%
operation p90: 100%
storage p50: 107.4142156863%
storage p90: 114.1203703704%
common-right rank p50: 100% of input width
common-right rank p90: 100% of input width
```

There was no exact equality/sign/proportional dot-product reuse in any measured Q/K/V group. The favorable operation count therefore remained identical to dense execution, while maps and multipliers increased storage. The exact shared input-transform width was never below the original input width.

No joint replay or physical kernel can rescue this registered mechanism because the required common arithmetic is absent before implementation overhead is charged.

## Scientific closure

Closed as primary core for the measured population:

- exact equal-row reuse across jointly evaluated projections;
- exact sign reuse;
- exact primitive integer-proportional reuse;
- exact repeated registered row blocks;
- exact common low-width right factors for Q/K/V.

Retained auxiliary infrastructure:

- projection-group discovery;
- exact canonical row certifier;
- collision verification;
- common-right-rank lower-bound accounting.

The next primary Gate must use demand-dependent computation or another new information source, rather than static exact common subexpressions in the weights.

## Claim boundary

Not tested:

```text
bitwise floating-point replay
physical joint kernel
actual Transformer operation replacement
405B execution
8 GiB peak VRAM
CUDA, PCIe, SSD, TTFT, tokens/second
```
