# VORTEX Research State

## Fixed target

Arbitrary public unmodified dense 405B-class checkpoint; executor-only; one 8-GiB GPU; exact output and successor state; same-machine native-4B-Q4 warm latency `p50<=1.2x`, `p95<=1.5x`.

## Operating mode

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Duplicate GitHub Actions execution is not required unless explicitly requested.

## Latest authoritative result — EXP-107A

```text
REJECT_CURRENT_680_EXPLICIT_FMM_CATALOG_COMPOSITIONS_AS_10X_CORE
REJECT_REGISTERED_NAIVE_RANK46_APA_INTERPOLATION_EXACTIFICATION
```

The first EXP-107A branch-sharing batch was entirely `NO_GO`; it shared weight reads but not the value-changing arithmetic/state of distinct branches. The second batch attacked arithmetic directly.

For the pinned 680-entry 2026 small-format FMM catalog, granting the best normalized exponent `2.792481250` to every target aspect ratio with unit constant and no transforms/additions/ABI/byte costs still leaves `12.515087006%` model-weighted multiplication at `K=16384`. The first-core line requires `10%`, equivalent to normalized exponent `2.7700683089152998`.

Smirnov rank-46 APA has zero-overhead algebraic headroom, but the registered 22-evaluation tensor-power interpolation plan raises the model indicator above `200%`; only that naive exactification is rejected.

## Active survivor — EXP-108A

```text
FINITE_STRASSEN_CALCULUS_DIRECT_SUM_EXTRACTION_CHEAP_KILL_ONLY
```

Derive an explicit finite exact target-size schedule from nontrivial direct-sum asymptotic-rank speedups. An asymptotic exponent statement is insufficient. Before backend work the schedule must cross `10%` after all arithmetic, exactification, coefficient, transform, native-ABI and workspace costs.

The causal `A`, `N/A`, exact verification and successor-state source remains an independent mandatory Gate after arithmetic survival.

## Not tested

- complete 405B target execution;
- physical complete 8-GiB allocation;
- target CUDA/SASS and storage/PCIe/HBM behavior;
- same-machine native-4B-Q4 p50/p95 acceptance.
