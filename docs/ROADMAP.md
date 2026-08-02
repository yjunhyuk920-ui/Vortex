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

Status: **E1 prototype complete; Atlas alone is not promoted**

Completed:

1. Added `OnlineAtlasLinear`, storing input basis `U` and exact operator image `WU`.
2. Added exact cold fallback and online basis expansion.
3. Routed selected `StreamingLlama` projections through a common `_linear` boundary.
4. Integrated O and down projections on the deterministic tiny checkpoint.
5. Added persistent atlas save/load and machine-readable metrics.

The same-trace replay result remains E1. It is not evidence that a model-wide atlas fits in 8 GiB or generalizes to unseen prompts.

## Milestone 2.5 — Architecture Gate 0

Status: **conditional budget implemented; real-model falsification pending**

Completed on `research/architecture-gate-zero`:

1. Defined the full Cascade Capsule v0 path for embeddings, all projections, attention, KV, LM head, repair, and scheduling boundaries.
2. Added `vortex_runtime/gate0_budget.py` and `scripts/run_gate0_budget.py`.
3. Committed `gate0_budget.json` with explicit proxy assumptions.
4. Derived the decisive threshold `A >= 246.889` amortized tokens per full-model-equivalent repair.
5. Added `GatedProjectedLinear`, which replaces the real operation and offloads the exact matrix to the cold path.
6. Added a Hugging Face falsification harness using disjoint build/evaluation prompts.
7. Added unit tests for budget rejection, memory rejection, basis construction, exact fallback, and fast-path equality.

Current certificate at target `A=512`:

- projected VRAM: 3.881 GiB;
- projected traffic: 1.650 GiB/token against a 2.400 GiB/token proxy gate;
- projected compute: 7.898 GFLOP/token against a 9.600 GFLOP/token proxy gate.

These are conditional projections, not measured feasibility. The 4B baseline is still a proxy, low-bit capsule kernels are not implemented, and the active-token attention path is only budgeted.

Exit gate:

- run the real-operation harness on a pretrained 1B–3B model;
- use disjoint Korean, English, code, math, JSON, and planning prompts;
- preserve the declared token/quality threshold;
- measure full-model-equivalent cold repairs per token;
- reject or revise if observed `A < 246.889`;
- replace proxy 4B values with same-machine measurements.

## Milestone 3 — Attention and cold-repair amortization

Starts only after the hidden-axis Gate 0 component survives falsification.

- implement bounded active-token attention and KV representation;
- measure quality as context grows;
- implement progressive low-bit basis/image correction;
- batch or coalesce layer repairs where measurement shows reuse;
- preserve exact committed-prefix semantics in exact mode;
- reject approaches that improve projected FLOPs but regress wall-clock.

## Milestone 4 — Native backend

Starts only after Architecture Gate 0 has measured support.

- packed 3-bit projected-image and 8-bit basis formats;
- fused basis projection and projected-image CUDA kernels;
- asynchronous host/storage cold-repair pipeline;
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
