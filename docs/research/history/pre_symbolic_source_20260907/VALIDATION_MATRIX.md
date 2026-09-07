# Validation matrix — 2026-09-07

[Current record and scope](experiments/native_response_code_20260907/REPORT.md).
[Prior matrix unchanged](docs/research/history/pre_native_response_code_20260907/VALIDATION_MATRIX.md).

| Item | Exact evidence/scope |
|---|---|
| Constructor |Actual small native responses, exact GF(2) factors, original pivot rows, serialized input-domain/address/output program|
| Runtime |Python and C use current input and code only; original weights and activation LUT absent from query|
| Numeric scope |BF16 stores, separately rounded FP32 products/balanced trees, fixed torch2.10 CPU SiLU primitive|
| Main coverage |10 synthetic SwiGLUs,148000 explicit-domain inputs,592000 output coordinates,0 Python/C mismatches|
| Independent checks |544 scalar struct references;826 singleton SiLU comparisons; exact binary pivot minors|
| Other forms |Full Mobius reconstruction and interval reconstruction; explicit queries; zero/identity/signed-zero controls|
| Costs |768-byte h32 weights yield208/2160/33368/468328-byte programs; exhaustive Q^n construction; logical reads not GPU traffic|
| Scope witness |Native whole3.515625 vs independently stitched1.4609375; only rejects naive additive composition|
| Reproduction |20 tests;171 generated files byte-identical; verify.py checks full-manifest SHA|
| Full theory |All O1-O6 OPEN; no core admission or three qualifying principle claim|
| Public HF/full Transformer KV-RNG/CUDA/GPU/405B/4BQ4/TTFT |NOT TESTED|
| Full repository suite/Actions |NOT RUN|

Remote source regenerates all synthetic inputs/programs/traces. Full originals are
in the user archive; no claim that all binary data is embedded remotely. Archival
compression is not an inference gain. Prior evidence, constraints and plans preserved.
