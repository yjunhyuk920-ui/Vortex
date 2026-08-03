# Next Experiment

## Closed Gate — EXP-052

Authority: `results/exp_052/summary.json`; workflow `30811429049`; source head `d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`.

1,152 exact warm states and 36 leave-one-family-out rows produced zero wrong hits and zero build/evaluation leakage, but P0 prefix and S0 KV-state held-out hit rates were 0% in every family. Fallback was 100%, natural exact reuse median/max was 1/1, and p90 fully-accounted target fraction was 6.0 (600%). Same-state replay was 100% exact and required at least 85 repetitions. Under 8 GiB hot index plus 1 TiB cold advice, combined coverage of 2^48 independent states was 6.357828752356909e-7, leaving fallback 0.9999993642171248.

Decision:

```text
REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY
```

## EXP-053 — Automatic Bit-Exact Decision-Circuit Compiler Gate

### Mechanism change

Compile a bounded quantized target operator directly from immutable weights and exact arithmetic semantics into a reduced Boolean/arithmetic decision circuit. Runtime states may not be stored as the representation.

### Conditions

```text
Q0 independent bit-exact arithmetic reference
Q1 weight-derived bit-vector/AIG compiler
Q2 structural hashing and exact reduction
Q3 structured sparse/low-rank controls
Q4 adversarial random dense and late-bit operators
Q5 exact circuit query and exact fallback
```

### Initial domains

```text
input bits 8, 12, 16, 20
output classes 2, 4, 8
accumulator widths 8, 12, 16
structured and dense-random operator families
```

### Contract

- no training, future generated token, or state truth table as the compiled representation;
- compiler input is weights/config/arithmetic semantics only;
- exhaustive small-domain enumeration is validation only;
- bit-exact equality is mandatory;
- compile time, nodes, bytes, reduction, query touches/bytes, and fallback are charged;
- structured success cannot erase adversarial/random scaling failure.

### Early rejection Gate

```text
exact mismatch >0
hidden truth-table representation
p50 query node/byte fraction >10%
p90 query node/byte fraction >25%
1 TiB projection exceeded before target scale
adversarial node growth doubling exponent >1.5 per added input bit
compile cost not amortizable under measured reuse
random dense cases require near-full original arithmetic
```

Promotion still requires real small-checkpoint operation replacement and p90 fully-accounted fraction <=1.185185%.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement bit-vector arithmetic and an exact AIG evaluator;
2. compile dense linear top-1 decisions from weights without state enumeration;
3. add structural hashing and reduction;
4. use exhaustive small-domain evaluation only for equivalence validation;
5. measure structured versus adversarial node/query scaling;
6. freeze circuits, checksums, scaling fits, and decision.
