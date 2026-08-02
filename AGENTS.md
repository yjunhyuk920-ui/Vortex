# AGENTS.md — VORTEX session contract

This file is the first document every new AI or human development session must read.

## Fixed mission

Build a universal runtime that can take an arbitrary Hugging Face model and automatically execute it under an 8GB VRAM budget. The flagship acceptance target is a 405B dense model with original-model quality and wall-clock behavior comparable to a native 4B model on the same machine.

The target must not be silently changed into any of the following:

- training or distilling a replacement student model;
- fine-tuning, LoRA, or model-specific calibration performed by the user;
- requiring a user-authored adapter for each architecture;
- merely making 405B fit while accepting unusably slow token generation;
- reporting theoretical FLOPs, compression ratio, projected traffic, or weight-equivalent speed as final success without wall-clock measurement;
- claiming final feasibility or success from tiny-model, synthetic, same-prompt, or replay validation.

Automatic first-run runtime-format generation is allowed, provided it is initiated transparently by the runtime and requires no model-specific user work.

## Mandatory proof-first rule

Read `docs/PROOF_FIRST_CONTRACT.md` before proposing or implementing a model-wide architecture.

No architecture becomes the main path until its committed feasibility certificate shows a plausible closure of all three flagship inequalities:

```text
M_hot + M_kv + M_work + M_repair <= 8 GiB
B_hot + B_cold / A <= 1.2 * B_4B
C_hot + C_repair / A <= 1.2 * C_4B
```

The certificate must define every term, use measured values where available, use conservative bounds otherwise, and include a fast falsification experiment.

Do not implement a large backend first and calculate 405B scaling afterward.

## Evidence language

Every result must be labeled E0, E1, E2, E3, or E4 as defined in `docs/PROOF_FIRST_CONTRACT.md`.

- E0: idea only.
- E1: local synthetic/tiny primitive.
- E2: real-model operation replacement on disjoint traces.
- E3: measured scaling trajectory across multiple model sizes.
- E4: real 405B flagship completion.

Only E4 may be described as the target being achieved. Never describe E1 or E2 as proof that the full target is possible.

## Current implementation truth

The repository currently contains E1 research primitives:

1. safetensors model/shard discovery without full model construction;
2. tensor and row-slice access;
3. byte-bounded tile caching;
4. a streamed tiny Llama reference decoder;
5. exact progressive argmax certification for a linear LM head;
6. disk-backed low-bit base plus lossless residual refinement;
7. exact Jacobi block decoding on tiny deterministic checkpoints;
8. exact-on-span `OnlineAtlasLinear` replay for selected tiny-model internal projections.

These primitives do not prove unseen-prompt generalization, a model-wide 405B resource budget, an end-to-end CUDA runtime, 8GiB peak VRAM, or 4B-class wall-clock speed.

## Required session startup

Before editing:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Read:

- `docs/PROOF_FIRST_CONTRACT.md`
- `docs/SESSION_HANDOFF.md`
- `docs/ROADMAP.md`
- `docs/VALIDATION_PROTOCOL.md`
- `validation_results.json`

Inspect the implementation before proposing a replacement architecture. Preserve verified behavior unless a test demonstrates that a change is necessary.

## Development loop

Every meaningful architecture change follows this order:

1. State one measurable hypothesis.
2. Write its correctness or declared-quality contract.
3. Derive 405B memory, traffic, and compute equations.
4. Demonstrate on paper that the flagship inequalities can close.
5. Define a falsification test and rejection threshold.
6. Implement the smallest executable version.
7. Replace the real operation when testing real models; hook-only analysis is not an E2 result.
8. Use disjoint build and evaluation traces.
9. Add or update automated tests.
10. Run the full tests and validation.
11. Record positive and negative results in machine-readable form.
12. Update session handoff and architecture documents.
13. Commit with a message describing the verified evidence level, not the hoped-for outcome.

## Immediate engineering priority

Stop extending `OnlineAtlasLinear` as though it were already the final architecture.

The next milestone is Architecture Gate 0:

1. Define one complete model-wide candidate execution path, including embeddings, all attention/MLP projections, nonlinear operations, LM head, KV handling, cold repair, storage, and CUDA scheduling.
2. Produce a committed 405B feasibility certificate for memory, bytes/token, compute/token, and cold-stream amortization.
3. Create an executable falsification harness that measures the certificate terms on a real pretrained 1B–3B model with disjoint prompts.
4. Reject the architecture unless the observed slope remains compatible with the final 405B gate.

Do not call a component a core solution merely because it avoids a weight read on a replayed trace.

## Evidence rules

A result is accepted only when reproducible from committed code. Keep raw metrics in machine-readable JSON. Never rewrite a failed result as success; failed hypotheses are project data.

The final target may be declared achieved only after the real-hardware gates in `docs/VALIDATION_PROTOCOL.md` pass.