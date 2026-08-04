# Next Experiment

## Closed Gate — EXP-069

EXP-069 tested whether exact projection inputs from earlier causal tokens span later inputs, allowing later `W*x` outputs to be reconstructed from cached exact `(x_k, W*x_k)` pairs without rereading `W`.

Every captured float32 scalar was interpreted as its exact dyadic rational and mapped into three odd prime fields. A rank increase under any prime certifies that the new input is outside the exact rational span of all previous inputs and therefore requires a full projection pass.

Authoritative coverage and correctness:

```text
3 pinned models
18 model/prompt cases
6 required families
147 registered projections
833 warm projection traces
8 EXP-069 tests passed
frozen EXP-061 weight-hash mismatches: 0
output token mismatches: 0
registration mismatches: 0
rank/trace/control mismatches: 0
```

Favorable mandatory lower bound:

```text
p50 weight-read fraction: 100%
p90 weight-read fraction: 100%
p50 operation fraction:   100%
p90 operation fraction:   100%
TinyStories-1M p50:        69.2439812633%
TinyStories-3M p50:        100%
TinyStories-8M p50:        100%
verified exact replay hits: 0
p50 basis cache / Q4 projection population: 391.9746782317%
```

The mandatory lower bound grants every modular non-increase, coefficient search, rank metadata, and unverified reconstruction for free. Certified-independent arrivals alone still consume the full budget at p50 and p90.

Decision:

```text
REJECT_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_AS_CORE
RETAIN_DYADIC_RANK_AUDITOR_AUXILIARY
```

Exact temporal span reuse is closed as a primary core. It may not be reopened using numerical tolerances, approximate subspaces, longer traces selected after observation, future/cross-prompt dictionaries, or uncharged coefficient/cache work.

Authority:

```text
results/exp_069/summary.json
results/exp_069/raw/projection_rows.jsonl
results/exp_069/raw/case_rows.jsonl
results/exp_069/raw/control_rows.jsonl
workflow 30922174380
artifact 8897596252
artifact ZIP SHA-256 81e73226e5369a4fb876d3d855f1d1dc69e0a182a7e584b858d4a111a0724247
```

## EXP-070 — Exact Q4 Local-Pattern Table Circuit Gate

### Execution-class change

EXP-070 returns to static exact arithmetic, but at a granularity not covered by whole-row/whole-column reuse, low rank, Kronecker/TT structure, or prototype residuals.

For a Q4 integer projection `y = W x`, partition the input columns into short blocks. Within one block, many output rows may contain the same short coefficient pattern:

```text
W[row, block] = p
```

For the current activation block `x_block`, compute each distinct partial dot product once:

```text
v_p = p · x_block
```

Every row carrying pattern `p` gathers the same `v_p`. This is the exact finite-alphabet table method often called a Four-Russians-style linear circuit. It can reuse partial arithmetic even when no complete rows or columns are equal.

### Why this class is allowed

Potential upside:

- Q4 has a small coefficient alphabet, so short patterns must repeat;
- one partial dot product can serve many output rows;
- the transform is exact in the registered Q4 integer domain;
- no training, activation approximation, future token, or model modification is required.

Reasons for low prior probability:

- pattern IDs may require almost as many bits as the original Q4 block;
- dictionary coefficients and row-routing metadata must also be read/stored;
- output assembly adds one gather/accumulation per row and block;
- wider blocks reduce assembly work but rapidly become unique;
- GPU Q4 kernels already exploit packed low-bit arithmetic, so scalar arithmetic savings may not become physical speed.

### Registered block families

Evaluate only the bounded widths:

```text
2, 3, 4, 6, 8, 12, 16 columns
```

For non-divisible widths, the final short block is charged at its actual width. Evaluate these deterministic column orders only:

```text
natural order
bit-reversal order where the width permits it
lexicographic column-signature order
```

No unbounded partition search, learned permutation, or post-result width addition is permitted.

### Exact plan and accounting

For every block:

1. group output rows by the exact Q4 coefficient tuple;
2. reconstruct every row block from its dictionary pattern ID and verify equality;
3. compute one partial dot product per distinct nonzero pattern;
4. gather and accumulate one partial result per output row;
5. charge the exact pattern dictionary, row IDs, block offsets, and routing metadata.

Report three separate quantities:

```text
operation fraction
query-byte fraction
static representation fraction
```

Favorable operation accounting may treat multiplication by `0`, `+1`, and `-1` at their exact minimal costs, but it must charge all other coefficient work and output assembly. Query bytes must include every dictionary coefficient and pattern ID needed by the token. Static storage and query traffic may not be conflated.

### Population

Use the unchanged pinned real-Q4 dense projections and checksums from EXP-057/058:

```text
3 TinyStories checkpoints
144 dense projections
all registered matrix roles and model sizes
```

No selected-tensor-only result may promote the candidate.

### Controls

- repeated local patterns reconstruct exactly and fall below the Gate;
- forced-unique patterns do not appear compressible;
- dense-random Q4 matrices retain high pattern entropy;
- one-nibble mutation changes the expected dictionary class;
- natural and permuted plans reconstruct the identical Q4 matrix;
- dictionary/hash collisions are checked by full tuple equality;
- all dictionary, ID, offset, gather, and accumulation costs are charged.

### Promotion Gate

```text
zero checksum/reconstruction/control/collision mismatch
100% registered dense-projection coverage
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-byte fraction <=10%
p90 query-byte fraction <=25%
p50 static representation fraction <=10%
p90 static representation fraction <=25%
no required model role with p90 >25% in either cost axis
no largest-model degradation >25%
```

Passing authorizes only an exact floating-point replay-order and physical table-kernel Gate. It does not authorize a 405B or 8 GiB claim.

### Failure decision

```text
REJECT_EXACT_Q4_LOCAL_PATTERN_TABLE_AS_CORE
RETAIN_BLOCK_PATTERN_ANALYZER_AUXILIARY
```

On failure, the registered short-block table/dictionary family is closed. It may not be rescued by reporting arithmetic without bytes, hiding row-routing costs, choosing widths after observation, or counting repeated synthetic controls as real-model evidence.

### Stop rule

Before survival, prohibit:

```text
CUDA table kernels
model-wide packed-table conversion
learned column permutations
approximate pattern merging
405B implementation work
```

### Claim boundary

Phase A/B/C real-Q4 structural evidence, ceiling E1. Floating-point replay order, a physical lookup kernel, actual Transformer replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain **NOT TESTED**.
