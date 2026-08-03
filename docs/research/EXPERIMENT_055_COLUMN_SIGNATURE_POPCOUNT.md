# EXP-055 — Exact Column-Signature Popcount Aggregation Gate

## Question

Can identical or exact-negated multi-class weight columns be grouped into activation popcounts while preserving signed modular top-1 decisions and meeting the universal runtime budget?

## Authority

- workflow `30820909775`
- source head `c15b1bb94496ad629bf8911d30d47a7cbe792595`
- workflow merge `58e83895bbc626391cb9ac70397cea14b70c84a4`
- artifact `8858805996` (`53989` bytes)
- artifact ZIP SHA-256 `983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8`
- config SHA-256 `688e176e57f1a2cabebc55d2907bc6d6198b4536015839f32001d9bf36222ff5`

## MEASURED

- 48 cases, 96 plans, 6 families;
- 248,832 scalar validations; 150,528 exhaustive and 98,304 deterministic sampled;
- score/top-1/packed mismatches: 0/0/0;
- runtime truth-table representations: 0;
- p50/p90 operations: 62.5%/250%;
- p50/p90 query bytes: 63.64%/200%;
- dense/forced-unique p50 operations: 250%;
- repeated n=64: 7.8125%; sign-related n=64: 9.375%;
- maximum projected logical 405B-Q4 storage: 0.7597 TiB;
- 21 cases had infinite compile amortization because runtime work did not beat baseline.

## Decision

```text
REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY
```

Exact repetition is a genuine auxiliary optimization, but it is not a universal core for arbitrary dense weights. Real Transformer extraction and hardware execution were not tested.
