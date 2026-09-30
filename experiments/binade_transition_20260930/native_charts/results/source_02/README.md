# Complete scalar native chart constructor

[Proof, exact domain, cost and evidence](REPORT.md)

This separate continuation implements the native chart partition left DERIVED in
the sealed parent package. It does not supply a cheap Transformer effect producer.

- `constructor.py`: complete non-NaN scalar state map with exact finite products
- `validate_native.py`: exhaustive tiny-format and selected FP32 checks
- `PREREGISTRATION.md`, `REFINEMENT_REGISTRATION.md`: frozen scopes
- `results/`: preserved source versions, raw counts and per-step records

Run with a fresh output path:

```sh
python experiments/binade_transition_20260930/native_charts/validate_native.py --output /tmp/native_chart_replay.json
```

All coefficients/products remain paid. No10x, HF, GPU or latency claim.
