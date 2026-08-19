# Fixed-Public Dynamic Executor — hosted run 31980847825 failure record

This file preserves the first actual hosted execution failure for commit `70f27c51066ee841db5ff5f83f68f9faf2c10006`.

## What actually executed

- GitHub Actions run: `31980847825`
- job: `real-checkpoint-constructor` (`95247318046`)
- runner: Ubuntu 24.04, AMD EPYC 7763, 4 logical CPUs, 16,766,414,848 bytes RAM
- pinned packages installed successfully:
  - `torch==2.5.1+cpu`
  - `transformers==4.46.3`
  - `safetensors==0.4.5`
- the model-free contract job passed before this job.
- the actual constructor step ran from `2026-08-17T00:04:58Z` to `2026-08-17T00:28:38Z`.
- the trace reached the second primitive, `checkpoint_mlp_torchinductor_existing_isa`; therefore the first cold-packed primitive did not satisfy its G4 physical-saving gate in that process.

## Failure fingerprint

This run is **not** a scientific mechanism-class rejection. It is a harness-path defect in the second primitive.

TorchInductor was configured with a relative `TORCHINDUCTOR_CACHE_DIR` under:

`results/fixed_public_dynamic_executor/70f27c51066ee841db5ff5f83f68f9faf2c10006/artifacts/checkpoint_mlp_torchinductor_existing_isa/torchinductor`

Inductor invokes `g++` from a temporary build working directory. The generated compile command therefore referenced a relative `.cpp` path that was no longer valid. The observed fatal compiler message was:

`cc1plus: fatal error: .../torchinductor/...cpp: No such file or directory`

The Python exception was `torch._dynamo.exc.BackendCompilerFailed`, caused by `torch._inductor.exc.CppCompileError`.

## Evidence preservation

The failed job uploaded 13 files as Actions artifact ID `9272656699` with archive SHA-256:

`699de213c76cb339018b29795d0c6207b5d5da27166077b3a795c525885ade31`

Because the process failed before `result.json` construction finished, the canonical scientific ledgers were intentionally not updated from incomplete metrics. The raw hosted log remains authoritative evidence for this failure.

## Corrective action

`vortex_runtime/fixed_public_dynamic_inductor.py` is changed so the artifact root and `TORCHINDUCTOR_CACHE_DIR` are resolved to absolute paths before TorchInductor compilation. No eager fallback is enabled and no hypothetical ISA is introduced.

The same frozen workload and immutable checkpoint pins must be rerun. Only the rerun may issue the scientific G2/G3/G4 verdict.
