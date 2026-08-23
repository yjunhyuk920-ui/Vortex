# EXP-099A latest result — Dyadic Reassociation / Native-Order Lock Gate

- source commit: `4fbd6699fda885fbedbe7d2158e7ae7cb3c30d3f`
- decision: `PROMOTE_DYADIC_REASSOCIATION_LOCK_TO_EXPLICIT_RECTANGULAR_FMM_GATE`
- deterministic core: `09c8555081baaa82ad67497d40c1625bbaf06b3111997984a3974a26190c6119`
- holdout native-order repair p50: `0.000000000000%`
- holdout native-order repair p95: `0.059678819444%`
- holdout repair min/max: `0.000000000000% / 0.075954861111%`
- generic FP32-cell certificate p50/p95: `43.614969135802% / 46.392746913580%`
- maximum holdout role repair p95: `0.173611111111%`
- integrity failures: `[]`

## Holdout role repair fractions

- `down_proj`: repair p50 `0.000000000000%`, p95 `0.173611111111%`, exact-real cell certificate p50 `100.000000000000%`
- `gate_proj`: repair p50 `0.000000000000%`, p95 `0.081380208333%`, exact-real cell certificate p50 `100.000000000000%`
- `k_proj`: repair p50 `0.000000000000%`, p95 `0.000000000000%`, exact-real cell certificate p50 `100.000000000000%`
- `o_proj`: repair p50 `0.000000000000%`, p95 `0.173611111111%`, exact-real cell certificate p50 `100.000000000000%`
- `q_proj`: repair p50 `0.000000000000%`, p95 `0.000000000000%`, exact-real cell certificate p50 `100.000000000000%`
- `up_proj`: repair p50 `0.000000000000%`, p95 `0.065104166667%`, exact-real cell certificate p50 `100.000000000000%`
- `v_proj`: repair p50 `0.000000000000%`, p95 `0.000000000000%`, exact-real cell certificate p50 `100.000000000000%`

## Claim boundary

This Gate uses an FP64-enclosed exact-real proxy and a perfect official-match selector.
An explicit rectangular fast-matrix algorithm, causal long block, packed kernel,
TARGET-W, physical 8-GiB execution and same-machine 4B latency remain `NOT TESTED`.
