# EXP-106A latest result — Depth-Complete Width-Thin Surrogate

Authoritative decision:

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

- deterministic core: `9399769c602659bc2c52c0bb7fe2435aef679063bde85fb34523d01d58ef081b`
- integrity failures: `[]`
- local focused tests: `8 passed`
- deterministic rerun: byte-identical
- GitHub Actions: not run by policy

## Resource/behavior dichotomy

| width | peak resident | draft scan/token | required actual A p50 | result |
|---:|---:|---:|---:|---|
| 128 | 1.465912 GiB | 0.026286 GB | 342 | resource pass, nonzero kernel |
| 512 | 1.698543 GiB | 0.249908 GB | 378 | resource pass, nonzero kernel |
| 1,024 | 2.347819 GiB | 0.912178 GB | 546 | resource pass, nonzero kernel |
| 1,536 | 3.381136 GiB | 1.986807 GB | 1965 | resource pass, nonzero kernel |
| 1,664 | 3.729347 GiB | 2.351975 GB | 16902 | last p50-scan-feasible width; kernel 14,720 |
| 1,792 | 4.063258 GiB | 2.701788 GB | impossible | p50 scan fail |
| 16,384 | 202.377974 GiB | 214.493273 GB | impossible | injective width; resource fail |

All resource-feasible widths are strictly narrower than the target state and
therefore noninjective. The exact dense encoder control produced byte-identical
narrow codes for distinct states that the target readout distinguished. The
full-depth coordinate control matched 512/512 when the decision channel was
retained and 0/512 when an equally weighted decision channel was omitted.

An exact dense Hadamard operator also confirmed the flat-spectrum case: every
rank-deficient linear approximation has relative operator-norm error 1, so SVD
or a different linear bridge cannot provide a uniform arbitrary-checkpoint
rescue.

## Bit-sliced arm

One checkpoint bit per parameter costs `47.247070312 GiB`,
`5.905883789x` the GPU budget and
`21.137981440x` the p50 byte budget per
autoregressive token. It is rejected before backend work.

## Claim boundary

This is an executable finite-word and fully charged target-scale E1 Gate. No
public checkpoint weights were run because local weights were unavailable and
the arbitrary-checkpoint control was already decisive. Complete 405B execution,
physical 8 GiB, target CUDA/SASS, storage/H2D throughput, and final same-machine
p50/p95 remain `NOT_TESTED`.
