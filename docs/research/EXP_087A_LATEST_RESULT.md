# EXP-087A latest result

```text
source SHA             eda4310e8ad20cf38ab712c46a9ef7b386882c62
decision               REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_BUILD_GATE
evidence                E1
deterministic core      03680d9211b76a432c82176e8fb1542f52c6af15e1a3c21da47865c990b31512
```

## Actual public-checkpoint result

```text
build states                         96
evaluation states                    96
build program exact                  [True, False, True, True]
evaluation oracle vectors exact      0 / 96
evaluation oracle vector fraction    0.0
evaluation oracle row-exact p50      0.0017361111240461469
evaluation oracle mismatched rows p50 575.0
evaluation router vectors exact      0 / 96
evaluation router vector fraction    0.0
```

## Target-shape favorable ledger

```text
feature count                         79
sidecar bytes                         1783627776
sidecar GiB                           1.6611328125
compiled MLP operation fraction       0.00036154114283048187
whole-model fraction if MLP-only      0.18774338904543011
minimum build-amortization tokens     8100
```

## Gates

```text
{"build": false, "integrity": true, "oracle": false, "resource": true, "router": false}
```

This is a small-real-checkpoint favorable-oracle result. It is not a complete MLP replacement, successor-state result, CUDA benchmark, 405B execution, 8-GiB allocation, or 4B-class latency measurement.
