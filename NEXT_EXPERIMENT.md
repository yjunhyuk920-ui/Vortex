# Next Experiment

## Closed Gate — EXP-054

Authority: `results/exp_054/summary.json`; workflow `30816333096`; source head `2c63da85050afcedad6a00698a6f8fddd3bc99d2`; artifact `8856906303`; ZIP SHA-256 `0dc642f306cea99ce01095758a5f49151092d530efb94d36985553e408596edf`.

24 operators were compiled in natural and weight-magnitude orders: 48 completed diagrams, zero ceiling/fallback, zero mismatches across 9,013,248 validations, and zero truth-table representations. Selected global p50/p90 path fractions were 35%/95%. Dense-random growth was 1.6872587x per added input bit, maximum projected storage was 202.2479 TiB, and maximum order-search amortization was 1,185,055 queries. Late-bit controls reached 5–12.5% paths, but dense, low-rank, and sparse families failed the universal Gate.

Decision:

```text
REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY
```

## EXP-055 — Exact Column-Signature Popcount Aggregation Gate

### Mechanism change

For a binary activation vector and signed modular multi-class linear decision, represent each input column as the exact vector of class weights. Compile identical column signatures into groups and compute one activation count per group:

```text
score_vector = bias_vector + sum_g popcount(active bits in group g) * column_signature_g
```

Optional exact-negated grouping may use a canonical signature plus signed count only when modular equality is proved. Runtime states are not enumerated.

### Conditions

```text
G0 independent signed modular top-1 reference
G1 exact identical-column grouping
G2 exact sign-canonical grouping with proved modular reconstruction
G3 scalar and packed group-popcount evaluator
G4 sparse/repeated/low-rank structured controls
G5 dense-random and unique-column adversaries
G6 exact complete-domain validation
```

### Registered domains

Use binary inputs n=8/12/16/20/32/64, classes C=2/4/8, accumulator widths 8/12/16, and structured repeated-column, sparse, low-rank, dense-random, and forced-unique families.

### Accounting

```text
source columns and scalar weights
group count and signature bytes
group membership/index bytes
input bits scanned or popcount words
group popcount operations
scaled vector-add operations
p50/p90 logical scalar-operation fraction
p50/p90 logical bytes touched
compile time and storage
405B source-parameter projection
exact mismatch and fallback
```

Baseline is the exact dense scalar operation `C*n` signed conditional additions plus source weight reads. Bit scanning/popcount and all grouped vector operations are charged.

### Early rejection Gate

```text
exact mismatch >0
runtime state table used as representation
p50 operation fraction >10%
p90 operation fraction >25%
p50 byte fraction >10%
p90 byte fraction >25%
dense-random or unique-column p50 fraction >25%
projected grouped storage >1 TiB
non-degrading savings fail as n/classes grow
compile cost cannot be amortized within 1,000,000 queries
```

Failure decision:

```text
REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY
```

### Promotion boundary

Synthetic success still requires real checkpoint weight-column extraction and operation replacement, exact token/logit agreement, p90 fully-accounted target fraction <=1.185185%, non-degrading scaling, 8 GiB closure, and Phase-D evidence.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement exact column grouping and signed modular grouped evaluator;
2. add repeated, sign-related, sparse, low-rank, dense-random, and unique-column generators;
3. exhaustively validate small domains and use deterministic larger-domain controls;
4. measure grouped operations/bytes and storage projections;
5. freeze raw groups, manifests, checksums, scaling, and decision.
