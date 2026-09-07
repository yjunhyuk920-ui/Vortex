# Validation matrix — 2026-09-07

[Current record](experiments/context_response_20260907/REPORT.md).
[Previous matrix unchanged](docs/research/history/pre_context_response_20260907/VALIDATION_MATRIX.md).

| Item | Exact evidence/scope |
|---|---|
| Constructor/state |Checked sign-code Q/K, finite BF16 V, append/index/actual serialization and raw KV reconstruction|
| Native source |Nonmatching stable FP32 exp is exactly zero under declared geometry; not top-k approximation|
| Numeric/order |640 queries/10240 coordinates match FP32/BF16; original sparse tree grouping and signed zero retained|
| Independent checks |Scalar struct oracle; all65280 finite BF16 zero-tag inputs; position/order collision|
| Domain rejections |16 generic/perturbed keys rejected; not counted as successes|
| Query costs |1024 favourable p50=6.96%/7.79%,p95>10%; all-equal646.60%; logical bytes not measured GPU traffic|
| Preparation/state |1024 state88.32%; online init/append/query78.32%/62.99%; cold workload3.42x/3.44x disclosed|
| Whole-work gate |All dense projections retained; even free QK/PV leaves95.98% countedMAC at context4096|
| Tests/reproduction |16 tests; initial241 and final241 generated files byte-identical; capsule replay both manifests matches|
| Full theory |O1-O6 OPEN; no core admission or claim of three qualifying new principles|
| Public HF/full Transformer-RNG/CUDA/GPU/405B/4BQ4/TTFT |NOT TESTED|
| Full repository suite/Actions |NOT RUN|

Remote capsule has20 complete UTF-8 files and whole generated-manifest anchors.
Binary arrays/state/per-query traces regenerate; full originals are in user ZIP,
not all embedded remotely. Archival compression is not inference compression.
Prior evidence and fixed constraints are preserved. Handoff is independently verified.
