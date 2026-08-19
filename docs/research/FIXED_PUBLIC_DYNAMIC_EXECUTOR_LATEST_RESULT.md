<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:7777eda66232c6a37c39f4e55f5b9ce56032fe92 -->
## Fixed-public dynamic executor hosted result — commit `7777eda66232c6a37c39f4e55f5b9ce56032fe92` / run `32219219400`

- Verdict: `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `checkpoint_mlp_output_row_streamed_lossless_existing_isa`: G2=True, G3=True, G4=True; output-row-tile=128; artifact=4223092 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; peak-hot=393216 B; baseline p50/p95=113.832 ms/203.051 ms; candidate p50/p95=151.764 ms/241.453 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.
