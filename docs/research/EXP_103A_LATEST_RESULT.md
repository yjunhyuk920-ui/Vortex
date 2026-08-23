# EXP-103A latest result — Hierarchical Native-Rounding Page Certificate Gate

- decision: `REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE`
- deterministic core: `d912ff35377f34d76926c9ca1a887a92c51eb0f69899b8621340650b67b7c363`
- integrity failures: `[]`
- randomized soundness: `53676` certified pages / `200000` cases / `0` mismatches

## Traffic floor

| Q | metadata fraction | maximum full-page read fraction | required page skip |
|---:|---:|---:|---:|
| 32 | 1.562500000% | 0.000000000% | 100.000000000% |
| 64 | 0.781250000% | 0.403935185% | 99.596064815% |
| 128 | 0.390625000% | 0.794560185% | 99.205439815% |
| 256 | 0.195312500% | 0.989872685% | 99.010127315% |
| 512 | 0.097656250% | 1.087528935% | 98.912471065% |
| 1024 | 0.048828125% | 1.136357060% | 98.863642940% |

## Executed finite-word controls

- all-ones BF16 dense row, width 16,384: `100%` pages read and `100%` terms executed at every registered Q; exact result match `True`.
- 64 BF16 Gaussian dense rows, width 4,096: page-read p50/p95 `100% / 100%` at every registered Q; exact result match for every row.
- tiny-product positive control: pages are skipped and exact result remains identical, proving the certificate is operational rather than permanently false.
- deterministic rerun: byte-identical `result.json`.

## Meaning

Native rounding can certify genuine no-op pages, but one byte of exponent metadata per page already consumes part of the 1.185185% traffic allowance. The remaining allowance requires roughly 98.9–99.6% of full pages to be skipped. A legal all-ones BF16 dense row forces every page to be read, and realistic Gaussian controls also read every page. Therefore this mechanism cannot be the universal arbitrary-checkpoint 10× core.

## Claim boundary

This is an executable finite-word E1 Gate. Public-checkpoint execution, target 405B, physical 8 GiB, CUDA/SASS, and same-machine native-4B-Q4 latency remain `NOT_TESTED`; no model run was authorized because the universal byte Gate was already decisive.
