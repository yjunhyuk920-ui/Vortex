# EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

## Question

Can unchanged real Q4 dense Transformer matrices be represented by an exact classical Tensor-Train / Matrix-Product-Operator with sufficiently small bond dimensions to plausibly approach the fixed 405B/8 GiB/4B-class runtime target?

## Efficiency scope

This is the final authorized bounded screen in the classical exact single-matrix tensor-factorization direction after EXP-065. No MPO reconstruction, contraction kernel, backend integration, broad mode-order rescue search, checkpoint download, or repeated real-weight modular elimination is permitted before this Gate passes.

## Frozen evidence reuse

EXP-065 already evaluated every ordered nontrivial factorization

```text
m = m1 * m2
n = n1 * n2
```

of the same deterministic Q4 matrices under primes 251 and 257. It froze 6,108 validated plan rows with source checksums and zero witness mismatch.

For TT/MPO mode pairs `(m_k,n_k)`, the cut after mode `k` has factors

```text
m1 = product_{i<=k} m_i
m2 = product_{i>k}  m_i
n1 = product_{i<=k} n_i
n2 = product_{i>k}  n_i
```

and its prefix/suffix unfolding is byte-for-byte the EXP-065 Kronecker rearrangement

```text
W.reshape(m1,m2,n1,n2).transpose(0,2,1,3)
```

Therefore EXP-066 verifies frozen EXP-065 checksums and reuses each matching rank lower bound. A cut containing a unit factor was intentionally absent from EXP-065; it receives rank lower bound one, which is universally valid and deliberately favors the TT/MPO candidate. Any missing nontrivial mapping fails the correctness Gate.

## Precommitted mode family

For each matrix dimension, factor it into primes and one deterministic coarsening whose mode product is at most 16 where possible. Combine row and column schedules with only these deterministic variants:

1. paired forward;
2. paired with reversed column schedule;
3. row modes followed by column modes;
4. column modes followed by row modes;
5. alternating row/column singleton modes.

This bounded family was fixed before observing EXP-066 results. Failure may not be rescued by an unbounded ordering or radix sweep.

## Favorable accounting

With `R_0=R_L=1`, charge the classical dense-core slot lower bound

```text
S_core = sum_k R_{k-1} * m_k * n_k * R_k
```

using 4-bit core slots. Also charge row scales, biases, mode/rank metadata, input/output reads, and a favorable intermediate allowance. Construction, coefficient widening, sparse-core indexing, and many contraction-index operations are omitted, deliberately favoring survival.

The reused bond ranks are lower bounds. Combining them is sound because every classical TT core size is monotone in adjacent bond dimensions.

## Population

Frozen unchanged checkpoints represented by EXP-065 evidence:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Required coverage: all 153 two-dimensional tensors and all 144 registered dense projections. Source Q4 checksums and EXP-065 evidence-file checksums must match exactly.

## Controls

- every tested TT cut unfolding equals the EXP-065 rearrangement exactly;
- frozen input files match committed checksums;
- duplicate EXP-065 keys cannot conflict;
- exact rank-one MPO controls retain unit bond ranks;
- dense-random controls produce unfavorable classical MPO accounting;
- any missing nontrivial EXP-065 mapping fails closed;
- unit-boundary cuts are explicitly labeled as favorable rank-one lower bounds.

## Promotion Gate

```text
zero source-checksum/mapping/witness/control mismatch
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

Phase A/B/C-derived-from-frozen-real-Q4-evidence, evidence ceiling E1. EXP-066 performs no new real-model execution and no new real-weight modular-rank measurement. Exact MPO cores, Q4 model-output preservation, physical kernels, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, E6, and E7 remain NOT TESTED.
