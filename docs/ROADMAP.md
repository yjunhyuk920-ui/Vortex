# Roadmap

## Milestone 0 — Reproducible baseline

Status: **complete**

- deterministic tiny Llama checkpoint generator;
- safetensors layout inspection;
- byte-budgeted cache;
- exact streamed Llama path;
- automated tests and JSON validation output.

## Milestone 1 — Progressive output decision

Status: **complete at prototype scale**

- in-memory low-bit base plus exact residual;
- sound tile bounds;
- exact greedy argmax certification;
- disk-backed VTX matrix format;
- selective safetensors residual reads.

## Milestone 2 — Progressive internal projections

Status: **next active milestone**

1. Introduce a common linear-operator protocol used by `StreamingLlama`.
2. Support `ExactLinear`, `DiskProgressiveLinear`, and instrumentation wrappers.
3. Add progressive Q/K/V/O execution on the tiny checkpoint.
4. Add progressive gate/up/down execution.
5. Propagate or repair nonlinear error through RMSNorm, RoPE, softmax, SiLU, and elementwise multiplication.
6. Compare final tokens against the exact path for every test seed.
7. Record bytes, residual fraction, compute, wall-clock, and peak memory per projection type.

Exit gate:

- exact token equality across the committed test matrix;
- machine-readable internal-projection metrics;
- no regression in the existing seven tests.

## Milestone 3 — Batched target-weight amortization

- process multiple speculative/Jacobi positions per weight tile;
- maintain exact committed-prefix semantics;
- measure accepted tokens per target stream;
- add suffix-repair experiments;
- reject approaches that only improve theoretical FLOPs but regress wall-clock.

## Milestone 4 — Native backend

- packed low-bit VTX storage;
- fused progressive CUDA kernels;
- asynchronous host/storage pipeline;
- true 8GB GPU residency enforcement;
- benchmarks on 8GB NVIDIA hardware.

## Milestone 5 — Real model scaling ladder

Run the same protocol on increasing dense-model sizes:

1. 1B–3B
2. 7B–8B
3. 30B–34B
4. 70B
5. 405B

At every step, preserve the same API and record failures. Do not infer 405B behavior solely from tiny checkpoints.

## Milestone 6 — Universal Hugging Face graph lowering

- formal VORTEX IR;
- automatic architecture/operator discovery;
- Llama-family optimized lowerer;
- generic primitive fallback;
- MoE, multimodal, and state-space operator support;
- safe handling of `trust_remote_code` models.

## Final acceptance

The project is complete only when all gates in `docs/VALIDATION_PROTOCOL.md` pass, including real 405B execution, 8GB peak VRAM, original-quality acceptance, and 4B-class wall-clock behavior on the same test machine.
