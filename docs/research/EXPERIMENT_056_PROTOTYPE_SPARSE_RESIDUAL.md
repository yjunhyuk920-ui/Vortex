# EXP-056 — Exact Prototype Plus Sparse-Residual Dictionary Gate

## Authority

- workflow `30823042599`
- source head `73655fc216340d9bd1d452d779951c28ac1b3d3b`
- workflow merge `df19bf0dee5e7f42a10378d5bca70d5513697982`
- artifact `8859665874` (245256 bytes)
- artifact ZIP SHA-256 `9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8`
- config SHA-256 `f75819b0cc6a741fac464d0e2adec2cb9b83612e7a17eed4b95a0cec5c03f151`

## MEASURED

- 56 cases, 448 plans, 7 families;
- 1,161,216 scalar validations; 702,464 exhaustive and 458,752 sampled;
- score/top-1/packed mismatches: 0/0/0;
- runtime truth tables: 0;
- p50/p90 operations: 62.5%/131.25%;
- p50/p90 bytes: 62.115%/169.643%;
- dense/unique p50: 123.4375%;
- repeated n=64: 7.8125%; sparse prototype residual n=64: 10.9375%; sign clusters n=64: 15.625%;
- maximum projected logical 405B-Q4 storage: 0.6791 TiB;
- 24 cases had no positive amortization.

## Decision

```text
REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY
```

Exact prototype dictionaries remain auxiliary. The next evidence must measure real pinned checkpoint matrices before more synthetic representation work.
