# EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

## Status

Closed at evidence level E1.

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

Machine-readable authority:

```text
results/exp_066/summary.json
results/exp_066/evidence_manifest.json
results/exp_066/checksums.sha256
workflow 30913788506
artifact 8894166935
artifact ZIP SHA-256 4bf87298e47d4800be1c17a80fb0ccd03e7c064118d42f31f45a186e181d157a
```

## Question

Can unchanged real-Q4 dense Transformer matrices be represented by an exact classical Tensor-Train / Matrix-Product-Operator with sufficiently small necessary bond ranks to survive the fixed operation and storage Gates before any core reconstruction or kernel work?

## Bounded method

EXP-066 did not download or execute the models again and did not repeat real-weight modular elimination.

It reused two checksum-verified evidence sets over the same 153 deterministic Q4 matrices:

- EXP-065: 6,108 ordered nontrivial Kronecker-rearrangement rank rows;
- EXP-058: full integer/rational matrix-rank certificates for all 153 matrices.

For mode pairs `(m_k,n_k)`, an internal TT cut has row/column prefix and suffix products `(m1,m2,n1,n2)`. Its interleaved unfolding becomes the corresponding EXP-065 Kronecker rearrangement after independent row and column permutations. Rank is invariant under those permutations.

Cuts equal to `W` or `W^T` reuse the EXP-058 full-matrix rank. The exact adjacent inequalities

```text
R_k <= d_k R_{k-1}
R_{k-1} <= d_k R_k
```

then propagate lower bounds through neighboring bonds. Every remaining unresolved unit-boundary rank is assigned only a favorable lower bound; resolving it can increase, but cannot reduce, the accounting.

The preregistered family contained prime/coarsened schedules with physical mode size at most 16 and five deterministic order variants. No rescue sweep was allowed.

## Correctness and controls

```text
EXP-066 tests: 14 passed
2D tensors: 153
registered dense projections: 144
EXP-065 mapped cuts: 4,937
EXP-058 full-rank rows reused: 153
source checksum mismatches: 0
source witness mismatches: 0
missing nontrivial mappings: 0
control failures: 0
```

The dense-random control is correctly oriented as an incompressibility control: its p50 favorable operation fraction must remain at or above 25%. Measured p50 was 105%, so the control passed.

## Favorable population lower bounds

```text
operation p50: 3.8941375969%
operation p90: 6.7788461538%
query-byte p50: 2.9983836207%
query-byte p90: 6.1697345890%
storage p50: 11.0523897059%
storage p90: 22.9882812500%
aggregate storage fraction: 7.0691890618%
```

The operation Gate passed. The storage Gate failed because the preregistered p50 limit was 10%:

```text
11.0523897059% > 10%
```

This is already a favorable lower bound. Additional exact unit-boundary ranks, core construction overhead, coefficient widening, sparse indexing, and physical contraction costs can only make it worse. Therefore exact core reconstruction and kernel work are not authorized.

## Projection boundary

Applying the measured aggregate small-checkpoint fraction to a 405B Q4 parameter stream gives:

```text
projected lower-bound storage: 14,315,107,850 bytes
8 GiB:                         8,589,934,592 bytes
ratio:                         1.6665x
```

This is a projection from the TinyStories population, not a direct 405B measurement or a universal impossibility proof.

## Scientific closure

The result closes the preregistered bounded exact classical single-matrix TT/MPO family as the primary execution core for the measured real-Q4 population. It also closes rescue by unbounded mode-order expansion, Tensor Ring, Hierarchical Tucker, or adjacent relabelings unless a genuinely new measured mechanism reopens the class.

Retained auxiliary infrastructure:

- TT/MPO mode enumeration;
- permutation-equivalence controls;
- frozen-rank evidence reuse;
- exact adjacent-rank propagation;
- favorable storage/operation accounting.

The next primary research Gate must change execution class and pass bounded E0 triage.

## Claim boundary

Not tested:

```text
exact MPO core reconstruction
Q4 output preservation through an MPO runtime
physical contraction kernel
actual Transformer operation replacement
405B execution
8 GiB peak VRAM
CUDA, PCIe, SSD, TTFT, tokens/second
```
