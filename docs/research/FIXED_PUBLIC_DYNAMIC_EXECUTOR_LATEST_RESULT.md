<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:3b9534800671e17eaef74869cf792186e683d6de -->
## Fixed-public dynamic executor hosted result — commit `3b9534800671e17eaef74869cf792186e683d6de` / run `31982198413`

- Verdict: `REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `cold_packed_projection_sequential_materialization`: G2=True, G3=True, G4=False; artifact=4208296 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; baseline p50/p95=105.321 ms/202.285 ms; candidate p50/p95=137.679 ms/234.948 ms.
- `checkpoint_mlp_torchinductor_existing_isa`: G2=False, G3=True, G4=False; artifact=169700 B; reference-layer=7080192 B; compiled-layer-resident=7080192 B; baseline p50/p95=102.057 ms/102.057 ms; candidate p50/p95=2268.660 ms/2268.660 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.
