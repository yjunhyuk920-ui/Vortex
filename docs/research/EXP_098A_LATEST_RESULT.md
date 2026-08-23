# EXP-098A latest result — Q4/BF16 Rounding-Cell Oracle Gate

- source commit: `9abe9963d65b581ef7ec693f594f3c864a179258`
- decision: `REJECT_Q4_ROUNDING_CELL_ROW_REPAIR_AS_10X_CORE`
- deterministic core: `42a99d80cd4cf630dad22189e11a83376170487cf64f5b54a77451f7ab3483e3`
- holdout oracle original-BF16 repair p50: `98.961046006944%`
- holdout oracle original-BF16 repair p95: `99.236382378472%`
- holdout oracle original-BF16 repair min/max: `98.437500000000% / 99.273003472222%`
- maximum holdout role repair p95: `100.000000000000%`
- integrity failures: `[]`

## Holdout role repair fractions

- `down_proj`: p50 `99.305555555556%`, p95 `99.826388888889%`
- `gate_proj`: p50 `98.860677083333%`, p95 `99.625651041667%`
- `k_proj`: p50 `98.697916666667%`, p95 `100.000000000000%`
- `o_proj`: p50 `98.958333333333%`, p95 `99.652777777778%`
- `q_proj`: p50 `98.611111111111%`, p95 `99.348958333333%`
- `up_proj`: p50 `99.283854166667%`, p95 `99.609375000000%`
- `v_proj`: p50 `99.479166666667%`, p95 `100.000000000000%`

## Claim boundary

The match selector is a perfect oracle. A sound BF16 rounding-cell certificate,
packed Q4 plus exact repair kernel, TARGET-W, physical 8-GiB execution, exact
complete successor-state integration, and same-machine 4B latency remain `NOT TESTED`.
