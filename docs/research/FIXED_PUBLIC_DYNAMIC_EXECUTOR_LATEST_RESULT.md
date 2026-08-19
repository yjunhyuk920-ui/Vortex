<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:7777eda66232c6a37c39f4e55f5b9ce56032fe92 -->
## Fixed-public dynamic executor hosted result — source `7777eda66232c6a37c39f4e55f5b9ce56032fe92` / run `32219219400`

- Evidence commit: `50cc5e34e7e27b00be2edb868f1da5d9f36461e0`.
- Clarification commit: `96a0e3659dc589b647cbc95241b10aad0f3623bb`.
- Verdict: `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4`.
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; repository metadata resolved to the pinned SHA, but gated `config.json`, tensor index, weights, and performance remain `NOT TESTED`.
- Official reference forward: executed.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`.
- Six preregistered workload families completed 128 decode transitions each: `768/768` transitions, token mismatches `0`, successor-state mismatches `0`.
- Actual layer tensor audit: `9` parameter tensors, `7,080,192` reference bytes.
- `checkpoint_mlp_output_row_streamed_lossless_existing_isa`: G2=`PASS`, G3=`PASS`, G4=`PASS`; output-row tile `128`; artifact `4,223,092 B`; compiled layer resident `1,771,776 B`; peak decoded tile `393,216 B`; accounted footprint `6,388,084 B` versus `7,080,192 B` reference (`692,108 B`, `9.775%` reduction).
- Wall time did not improve: baseline p50/p95 `113.832/203.051 ms`; candidate `151.764/241.453 ms`.
- Each measured decode transition read `4,206,522 B` of cold artifact data, materialized `5,308,416 B` of raw MLP tensors, executed `29` tile projection calls, and performed `58` integrity probes.
- CPU compile wall time was `858,822,082 ns`; recorded compile peak RSS was `964,100,096 B`. The long-context case reached approximately `1,992,736,768 B` process RSS and `71,838,720 B` KV state at its final measured step.
- Existing-ISA execution used PyTorch eager CPU primitives. Instruction/SASS counts, CUDA, PCIe, GPU VRAM, TARGET-W latency, G5 4B-class path, and G6 8GiB target ledger are not promoted by this DEV-W result.

## Gate interpretation

This result establishes a real checkpoint-specific dynamic executor at a complete Transformer-layer boundary with exact successor-state semantics and a smaller fully charged layer footprint. It does not establish a speedup: output-row streaming trades persistent decoded-weight footprint for more calls, decompression, and latency. The next constructive gate is model-wide accounting plus TARGET-W metadata/tensor access; no DEV-W compression ratio is extrapolated to 405B.
