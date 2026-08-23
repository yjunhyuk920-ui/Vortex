# EXP-104A latest result — Checkpoint-Native Resident Slice p50 Gate

Authoritative decision:

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

Deterministic core:

```text
dcdce92c76ca84154d12e87cbfd904e1fdfe637cdbd0c8ea51d55e13605a6f93
```

Integrity failures:

```text
[]
```

## Key results

| Representation | Head bytes | One layer bytes | Head + one layer | Ratio to native 4B Q4 | Max full layers under 8 GiB ledger |
|---|---:|---:|---:|---:|---:|
| BF16 | 4,202,692,608 | 6,375,342,080 | 10,578,034,688 | 5.289017344x | 0 |
| Q8 g128 | 2,167,013,376 | 3,287,285,760 | 5,454,299,136 | 2.727149568x | 1 |
| ideal Q4, no metadata | 1,050,673,152 | 1,593,835,520 | 2,644,508,672 | 1.322254336x | 3 |
| Q4 g128 scale+zero | 1,116,340,224 | 1,693,450,240 | 2,809,790,464 | 1.404895232x | 3 |

The p50 target is `1.2x` the same-machine native-4B-Q4 baseline, or `2,400,000,000` ideal weight bytes per committed token. Even metadata-free Q4 head plus one complete layer exceeds that floor. The realistic Q4 draft alone is already `1.404895232x` native-4B weight bytes before target verification or any other online cost.

## Local validation

```text
7 focused tests passed
registered parameter population reproduced exactly
byte-identical deterministic rerun
Python compile PASS
SHA-256 evidence ledger PASS
GitHub Actions not run
```

## Claim boundary

This is a target-scale E0/E1 static and executable cost-ledger rejection. The public checkpoint slice was not executed because the same-machine p50 byte floor is already decisive. Complete 405B execution, physical 8-GiB allocation, target CUDA/SASS, and final p50/p95 latency remain `NOT TESTED`.
