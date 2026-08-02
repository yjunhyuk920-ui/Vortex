# AGENTS.md — VORTEX session contract

This file is the first document every new AI or human development session must read.

## Fixed mission

Build a universal runtime that can take an arbitrary Hugging Face model and automatically execute it under an 8GB VRAM budget. The flagship acceptance target is a 405B dense model with original-model quality and wall-clock behavior comparable to a native 4B model on the same machine.

The target must not be silently changed into any of the following:

- training or distilling a replacement student model;
- fine-tuning, LoRA, or model-specific calibration performed by the user;
- requiring a user-authored adapter for each architecture;
- merely making 405B fit while accepting unusably slow token generation;
- reporting theoretical FLOPs, compression ratio, or weight-equivalent speed as final success without wall-clock measurement;
- claiming final success from tiny-model or synthetic validation alone.

Automatic first-run runtime-format generation is allowed, provided it is initiated transparently by the runtime and requires no model-specific user work.

## Current implementation truth

The repository currently proves these runtime primitives:

1. safetensors model/shard discovery without full model construction;
2. tensor and row-slice access;
3. byte-bounded tile caching;
4. a streamed Llama reference decoder;
5. exact progressive argmax certification for a linear LM head;
6. disk-backed low-bit base plus lossless residual refinement;
7. exact Jacobi block decoding on tiny deterministic checkpoints.

It does not yet prove fast internal Transformer projections or the final 405B wall-clock target.

## Required session startup

Before editing:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Read:

- `docs/SESSION_HANDOFF.md`
- `docs/ROADMAP.md`
- `docs/VALIDATION_PROTOCOL.md`
- `validation_results.json`

Inspect the implementation before proposing a replacement architecture. Preserve verified behavior unless a test demonstrates that a change is necessary.

## Development loop

Every meaningful change follows this loop:

1. State one measurable hypothesis.
2. Implement the smallest executable version.
3. Add or update an automated test.
4. Run the full tests.
5. Run `scripts/run_validation.py` when performance or certification behavior changes.
6. Record both positive and negative results.
7. Update `docs/SESSION_HANDOFF.md` and, when architecture changes, `docs/ARCHITECTURE.md`.
8. Commit with a message describing the verified change, not the hoped-for outcome.

## Immediate engineering priority

Extend progressive refinement from the final LM head to internal Llama projections:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

The first experiment must be executable on the generated tiny checkpoint and must compare every progressive layer output or final token against the exact streamed path. It must measure:

- base bytes read;
- residual bytes read;
- residual fraction read;
- additional compute;
- exact output/token match;
- peak cache bytes;
- wall-clock time.

## Evidence rules

A result is accepted only when it is reproducible from committed code. Keep raw metrics in machine-readable JSON. Never rewrite a failed result as success; failed hypotheses are useful project data.

The final target may be declared achieved only after the real-hardware gates in `docs/VALIDATION_PROTOCOL.md` pass.
