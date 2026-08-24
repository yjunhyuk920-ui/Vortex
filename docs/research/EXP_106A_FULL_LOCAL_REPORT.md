# EXP-106A full local research report

## Final decision

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

Deterministic core: `9399769c602659bc2c52c0bb7fe2435aef679063bde85fb34523d01d58ef081b`  
Integrity failures: `[]`

## What was tested

A training-free, depth-complete checkpoint surrogate with a fixed width-`m`
linear state was given every concrete component required for an executable
causal draft: Q4 g128 all-layer operators, embedding bridge, full narrow head,
KV, workspace, target staging, allocator reserve, and raw-BF16 target
verification amortization. Registered widths ranged from 128 to the full 16,384.

## Main result

Resource-feasible widths exist only in the noninjective regime. The largest
width that both fits 8 GiB and scans below the p50 limit is 1,664, but it leaves
a 14,720-dimensional kernel and needs an actual accepted segment of 16,902 to
amortize the target sweep. Width 1,792 already scans 2.701788376 GB/token and
cannot meet p50 even with infinite acceptance. Full width scans
214.493272576 GB/token and occupies 202.377974 GiB including the registered
reserves.

The exact dense encoder collision and full-depth omitted-channel control show
that an arbitrary legal target can distinguish states the narrow draft must
identify. The dense Hadamard control rules out an SVD/rotation rescue as a
uniform guarantee: all singular values are equal, and every rank-deficient
approximation has full relative operator-norm error.

A one-bit-per-parameter all-layer transducer requires
47.247070312 GiB, before nonlinear/carry metadata, and is also
outside both resource floors.

## Validation

```text
8 focused tests passed
Python compile PASS
byte-identical deterministic rerun
SHA-256 ledger PASS
GitHub Actions not run
```

## Boundary

No public checkpoint weight run was performed because weights were unavailable
locally and the arbitrary-checkpoint finite-word Gate was decisive. This closes
only the registered fixed-width linear/low-rank/coordinate surrogate and
resident bitplane families as universal cores. It does not close empirical
restricted drafts or a new nonlinear cold-backed exact decoder.
