# EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

## Question

Can unchanged real Q4 dense Transformer matrices be represented by an exact classical Tensor-Train / Matrix-Product-Operator with sufficiently small bond dimensions to plausibly approach the fixed 405B/8 GiB/4B-class runtime target?

## Efficiency scope

This is the final authorized bounded screen in the classical exact single-matrix tensor-factorization direction after EXP-065. It is intentionally limited to exact unfolding-rank certificates. No MPO reconstruction, contraction kernel, backend integration, or broad mode-order rescue search is permitted before this Gate passes.

## Precommitted mode family

For each matrix dimension, factor it into primes and one deterministic coarsening whose mode product is at most 16 where possible. Combine row and column schedules with only these deterministic variants:

1. paired forward;
2. paired with reversed column schedule;
3. row modes followed by column modes;
4. column modes followed by row modes;
5. alternating row/column singleton modes.

This bounded family is fixed before observing EXP-066 results. Failure may not be rescued by an unbounded ordering or radix sweep.

## Exact certificate

Given mode pairs `(m_k,n_k)`, reshape the integer Q4 matrix into an interleaved tensor with physical dimensions `d_k=m_k*n_k`. For every internal cut `k`, certify

```text
R_k = rank(unfold(W, product_{i<=k} d_i, product_{i>k} d_i))
```

independently under primes 251 and 257. Each witness stores pivot rows, pivot columns, and a verified nonzero modular minor determinant. The maximum certified prime-field rank is a rigorous lower bound on the exact integer/rational bond rank.

## Favorable accounting

With `R_0=R_L=1`, charge the classical dense-core slot lower bound

```text
S_core = sum_k R_{k-1} * m_k * n_k * R_k
```

using 4-bit core slots. Also charge row scales, biases, mode/rank metadata, input/output reads, and a favorable intermediate allowance. Construction, coefficient widening, sparse-core indexing, and many contraction-index operations are omitted, deliberately favoring survival.

## Population

Unchanged pinned checkpoints:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Required coverage: all 153 two-dimensional tensors and all 144 registered dense projections, with exact agreement against frozen EXP-057 Q4 checksums.

## Controls

- exact rank-one MPO has unit bond ranks;
- one-nibble mutation raises at least one bond rank;
- reshape/interleave round trip is exact;
- dense-random controls produce high certified bond rank;
- every selected witness validates under both primes;
- malformed shapes and insufficient prime sets fail closed.

## Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 favorable operation fraction <=10%
p90 favorable operation fraction <=25%
p50 favorable storage fraction <=10%
p90 favorable storage fraction <=25%
dense-random p50 favorable operation fraction <=25%
projected static storage <=1 TiB
largest-model degradation <=25%
```

A surviving lower bound is still insufficient for runtime promotion. Exact integer MPO reconstruction and actual operation replacement would require a new Gate.

## Failure decision

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

On failure, exact classical single-matrix tensor factorization is closed as the primary direction for this measured real-Q4 population. The next core candidate must change execution class and pass the research-efficiency E0 triage.

## Claim boundary

Phase A/B/C-observation, evidence ceiling E1. Exact MPO cores, Q4 model-output preservation, physical kernels, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, E6, and E7 remain NOT TESTED.
