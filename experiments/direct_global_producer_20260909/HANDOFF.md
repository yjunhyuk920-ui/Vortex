# Handoff — direct global finite-word producer frontier

## Current authority

Read, in order:

```text
PREREGISTRATION.md
REPORT.md
obligations.json
results/e0_direct_global_producer_gate_v5/summary.json
```

Canonical summary SHA-256:

```text
a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe
```

## What changed

Three direct-producer principles were frozen and all executed.

1. Global reconstructive coding fails before implementation optimization: 8 GiB
   can replace only 4.255098% of arbitrary Q4 source bits and dense arithmetic
   remains 100%.
2. Exact mathematical sum does not isolate a cheap rounding problem.  A balanced
   FP32 zero-sum gadget encodes arbitrary binary MatVec information purely in
   rounding error.
3. A new global nonlinear vector-route theorem permits cross-matrix advice and
   adaptive cells without per-matrix division, but its registered 578,619-word
   lower bound is 516.87x below the favorable target allowance.

An actual arbitrary-GF(2) producer was also constructed using deterministic
Gauss-Jordan row operations and exact reverse replay.  It is exact but fails the
90% operation Gate: `16385/32768` at width 16,384 for its fixed adversary, with
447.97 MiB of row-op metadata lower bound for that one matrix.

## Do not reopen by renaming

- reconstruct-all erasure/network coding with the same 8-GiB information source;
- exact-sum plus an unspecified/free rounding witness;
- Gauss-Jordan/LU row-operation replay as though the operation stream were free;
- the old finite-semiring/M4RM table/full-scan routes;
- per-matrix division of global nonlinear advice.

## Remaining constructive object

```text
Compile(arbitrary native checkpoint) -> implicit global nonlinear state G
Address(G,current causal state) -> subdense paid addresses
Decode(words) -> exact native ordered dense effects + successor state
```

The missing innovation has to make this direct query genuinely smaller than
both source reconstruction and a quadratic static program.  The new global
route theorem is a guardrail, not the algorithm.

O1--O5 remain OPEN; O6 is PARTIAL.  405B/CUDA/<=8GiB/PCIe/SSD/native4BQ4
p50/p95/TTFT remain NOT TESTED.
