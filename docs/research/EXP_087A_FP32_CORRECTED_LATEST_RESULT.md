# EXP-087A corrected latest result

## Provenance correction

- superseded control-failure result commit: `4a23f49cd6e5354e8616eb930814e11db95cd933`
- corrected source SHA: `78e9901ced2937e76be9fa75a1383baa98f0057b`
- raw corrected result: `results/exp_087a_fp32/78e9901ced2937e76be9fa75a1383baa98f0057b/result.json`
- deterministic core: `c5600395900403ad181314b6e9919b8efeea959a40f42f5f8d43e2d24408a905`
- evidence: `E1`

The earlier build rejection used FP64 predicate labels with an FP32 deployed sidecar. The corrected compiler selects and fits predicates with the exact charged FP32 runtime representation. Checkpoint, states, language capacity, resource equations, and thresholds are unchanged.

## Corrected public-checkpoint result

```text
decision                              REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_AT_ORACLE_GATE
build program exact                   [True, True, True, True]
build oracle vectors exact            96 / 96
evaluation oracle vectors exact       0 / 96
evaluation oracle vector fraction     0.0
evaluation oracle row-exact p50       0.0017361111240461469
evaluation oracle mismatched rows p50 575.0
evaluation router vectors exact       0 / 96
```

## Frozen favorable ledger

```text
sidecar GiB                           1.6611328125
compiled MLP operation fraction       0.00036154114283048187
whole-model fraction if MLP-only      0.18774338904543011
minimum build-amortization tokens     8100
gates                                 {"build": true, "integrity": true, "oracle": false, "resource": true, "router": false}
```

This result is scoped to the official small public checkpoint and one complete MLP boundary. Complete Transformer successor state, physical sparse/bitwise lowering, TARGET-W, 8-GiB VRAM, and 4B-class p50/p95 remain `NOT TESTED`.
