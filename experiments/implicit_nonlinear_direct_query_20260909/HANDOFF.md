# Handoff — implicit nonlinear direct-query frontier

## Current authority

Read in order:

```text
PREREGISTRATION.md
REPORT.md
obligations.json
VALIDATION.md
results/e0_implicit_direct_query_gate_v3/summary.json
```

Canonical summary SHA-256:

```text
78644fd5ef4b3e0131bf78867e8986718ed02faeaf37b4228fd1c37909a7af29
```

## What was actually constructed

1. **P1 query-side image frame**: an exact finite producer for every GF(2)
   matrix. The current right-factor block bits directly select compiled output
   images; there is no checkpoint-dependent runtime instruction stream.
2. **P2 value/state coalescing**: an exact finite ordered-leaf executor for the
   declared deterministic finite-word ABI, including a stronger online grouping
   by `(accumulator,weight)` state.
3. **P3 safe gauge controls**: exact proofs/controls that arbitrary
   meet-preserving Boolean bijections are coordinate permutations and that
   independent invertible linear product isotopies are one aligned permutation.

## Why none is the core

```text
P1 registered transformed storage   ~4,817.82 GiB
P1 registered query payload         ~4.699 GiB
P1 / registered 8/675 line          ~8.4357x

P2 static best operation fraction   16385/32768 ~= 50.003%
P2 high-distinctness fraction       1.0
P2 dynamic state adversary          1.0 update fraction

P3 safe permutation dense support   unchanged
```

The P1 general counting bound is only for complete one-sided **linear** image
frames. The P2 adversary is only for identical descriptor/state transition
sharing. The P3 theorem is only for encodings that preserve the coordinatewise
product without paying a richer transformed nonlinear operator.

## Do not reopen by renaming

- complete one-sided image dictionaries with larger blocks;
- per-weight duplicate-product caching as though row accumulation were free;
- `(accumulator,weight)` duplicate transition sharing as a universal escape;
- coordinate/product-preserving gauges claimed to sparsify arbitrary dense maps;
- prior reconstruct-all coding, exact-sum/free-rounding-witness, Gauss-Jordan
  row-program replay, Boolean-semiring lifts, static Lupanov/Steiner programs or
  full-scan finite-semiring tables.

## Remaining constructive object

```text
Compile(arbitrary unchanged native checkpoint) -> compact/nonlocal representation G
Address(G,current causal state/right factors)   -> subdense paid probes
Decode(words,state)                            -> exact ordered native dense effects
                                                -> exact logits/KV/RNG successor
```

The next core direction must be materially different from all three principles
above and must explicitly construct every selector/decoder/transformed nonlinear
operator/state update rather than name it.

O1--O5 remain OPEN; O6 is PARTIAL. 405B/CUDA/<=8GiB/PCIe/SSD/HBM/native4BQ4
p50/p95/TTFT remain NOT TESTED.
