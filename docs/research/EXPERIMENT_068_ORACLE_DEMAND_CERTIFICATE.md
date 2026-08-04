# EXP-068 — Oracle Global Demand-Certificate Necessary Gate

## Question

Can an exact activation-conditioned lazy executor leave almost all target weights unread while still proving the same greedy token?

EXP-068 changes execution class again. It does not search for static weight structure. It asks whether the current hidden state and final-token margin permit exact fail-closed commitment after reading only a small subset of weights.

## Necessary-condition reduction

Before bounding every nonlinear Transformer path, EXP-068 isolates a condition that every registered global demand certificate must satisfy: the final output head must distinguish the winning token from every competitor.

The Gate is deliberately more favorable than any deployable executor:

```text
all preceding Transformer operations and weight reads are free
the complete winning output-head row is free
all precomputed norm/bound/order metadata is free
every competitor receives its own independently optimal reveal order
coordinate-sized tiles are permitted
```

Only the minimum competitor-row weight entries that remain mathematically necessary are charged. If this output-head-only lower bound already exceeds the whole-model target budget, adding the omitted Transformer work cannot rescue the execution class.

## Exact-real bound

For winning token `y`, competitor `i`, hidden coordinate `j`, output-head weights `w`, and hidden state `h`, define

```text
d_ij = (w_yj - w_ij) * h_j
```

After revealing coordinates `S`, exact commitment against competitor `i` requires

```text
bias_y - bias_i + sum_{j in S} d_ij
>
sum_{j not in S} |d_ij|
```

Equivalently, revealed coordinates must collect more than

```text
sum_j |d_ij| - (bias_y - bias_i)
```

of gain

```text
2 * max(d_ij, 0)
```

The oracle sorts gains independently for each competitor. This impossible per-competitor freedom minimizes charged work. Competitor rows contain disjoint weight entries, so the sum of their independent minima is a sound lower bound for any coordinate/tile implementation using these absolute unread bounds. The winner row is not charged.

The arithmetic is evaluated in exact real interpretation of the captured float32 values using float64 accumulation. The standard PyTorch model winner and direct output-head replay winner must match. Bitwise floating-point certification is not claimed.

## Population

Pinned unchanged checkpoints:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Held-out families:

```text
English narrative
code
mathematics
identifier boundary
Korean
structured JSON
```

Every two-dimensional source tensor is byte-hashed and compared with frozen EXP-058 source evidence. Required coverage is 18 model/prompt cases and all 153 registered 2D tensors.

## Accounting

The denominator is the frozen per-token dense weight population:

```text
all registered dense projections + output head
```

The numerator is only the necessary competitor-row output-head entries. Each entry is charged as one weight read and one multiply contribution. The following real costs are excluded to favor survival:

```text
all preceding dense layers
attention and nonlinear operations
the winner output-head row
bound metadata reads
bound aggregation
scheduler work
fallback work
```

## Controls

- a sparse large-margin head certifies after one coordinate;
- a late-flip construction requires every decisive positive coordinate;
- leaving a flip-capable coordinate unread prevents commitment;
- bounded random cases compare the derived lower bound with exact subset enumeration;
- non-finite inputs fail closed;
- full-model and direct-head greedy winners match.

## Promotion Gate

```text
zero source/reference/control/bound mismatch
3 pinned models
18 prompt cases
6 required families
153 two-dimensional tensors
p50 favorable whole-model weight fraction lower bound <=10%
p90 favorable whole-model weight fraction lower bound <=25%
p50 favorable whole-model operation fraction lower bound <=10%
p90 favorable whole-model operation fraction lower bound <=25%
no family p90 >25%
no largest-model degradation >25%
```

Passing this necessary Gate authorizes only the full-network downstream-bound stage. It does not authorize a scheduler or kernel.

## Authoritative result

Coverage and integrity:

```text
6 EXP-068 tests passed
repository validation passed
3 pinned models
18 prompt cases
6 required families
153/153 source tensor hashes matched
full-model/direct-head winner mismatches: 0
bound violations: 0
control failures: 0
```

Favorable output-head-only lower bound against the whole-model dense baseline:

```text
p50 weight-read fraction: 13.7696858262%   FAIL against 10%
p90 weight-read fraction: 19.2524013315%   PASS against 25%
p50 operation fraction:   13.7696858262%   FAIL against 10%
p90 operation fraction:   19.2524013315%   PASS against 25%
minimum case:              10.1376199755%
maximum case:              21.0055007890%
```

Per-family p90 lower bounds:

```text
code                 21.0055007890%
identifier boundary  19.2524013315%
Korean               17.8296751339%
English narrative    17.7761143419%
mathematics           17.7563341076%
structured JSON       13.3003515009%
```

The largest checkpoint had the lowest model p50, so the model-trend Gate passed. That does not rescue the failed global p50 target.

## Decision

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

The final output head alone exceeds the whole-model p50 budget after every preceding Transformer operation, every preceding weight read, the complete winning row, all metadata, and an independently optimal order for every competitor are excluded from cost. Full-network propagation and deployment overhead can only add work.

Therefore the registered exact norm/absolute-unread demand family is closed as a primary core. It may not be reopened through tile-size tuning, ordering search, a scheduler implementation, or a kernel implementation without a genuinely stronger exact theorem or new information source.

## Evidence authority

```text
results/exp_068/summary.json
results/exp_068/raw/case_rows.jsonl
results/exp_068/raw/control_rows.jsonl
results/exp_068/raw/model_rows.jsonl
results/exp_068/raw/tensor_rows.jsonl
results/exp_068/raw/snapshot_rows.jsonl
results/exp_068/evidence_manifest.json
workflow 30918865952
artifact 8896230736
artifact ZIP SHA-256 ff0f4398c0d162142d3e71d6864a3990704a14bf59e007182c9dce72c913835f
```

## Claim boundary

Phase A/B/C small-model oracle-bound evidence, ceiling E1. Bitwise floating-point certificates, a deployable scheduler, actual skipped Transformer operations, physical kernels, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED.
