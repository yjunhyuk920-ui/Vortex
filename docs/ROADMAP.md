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

## Milestone 2 — Base-free internal operator path

Status: **prototype path implemented; real-model gate pending**

Completed:

1. Added `OnlineAtlasLinear`, storing input basis `U` and exact operator image `WU`.
2. Added exact cold fallback and online basis expansion.
3. Routed selected `StreamingLlama` projections through a common `_linear` operator boundary.
4. Integrated O and down projections.
5. Added persistent safetensors atlas save/load.
6. Verified token-identical tiny-Llama replay with zero managed-projection cold reads.
7. Added machine-readable rank, hit, cold-read, byte, and capsule metrics.

Next gate:

- run real pretrained 1B–3B activation traces;
- measure rank growth and hit rate on unseen continuations/prompts;
- verify exact token agreement;
- reject the path if cold streams/token do not rapidly decline;
- implement FP16/BF16 capsules and real wall-clock instrumentation.

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
