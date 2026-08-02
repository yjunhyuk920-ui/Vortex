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

## Mandatory durable-progress rule

Read and obey `docs/WORK_SESSION_PROTOCOL.md`.

Before a user-facing progress or completion answer after meaningful repository work, commit the current state to GitHub. At minimum:

- append the measured positive or negative result to `docs/RESEARCH_PROGRESS_LEDGER.md`;
- update `docs/SESSION_HANDOFF.md` with the active branch, PR, head commit, workflow state, exact next command, and unresolved proof obligations;
- commit raw metrics under `results/` when available;
- update the experiment document and PR decision when a gate is promoted or rejected.

Do not claim that progress was documented unless the Git commit exists. If repository writing fails, state that failure explicitly in the answer.

This rule is permanent and does not require the user to repeat it in future sessions.

## Current implementation truth

The repository contains a broad E1/E2 research corpus, including:

1. safetensors model/shard discovery and slice access without full model construction;
2. byte-bounded tile caching and streamed tiny-Llama reference execution;
3. exact progressive LM-head refinement and bit-exact residual storage;
4. decision-certificate, activation-atlas, lossless-entropy/speculation, MLP dictionary, heavy-hitter, and layer-allocation falsification branches;
5. real TinyLlama all-layer operation replacements on disjoint prompts for multiple candidates.

Many candidate families have been rejected. Read `docs/RESEARCH_PROGRESS_LEDGER.md`; do not recreate a rejected architecture under a new name.

None of these results proves a universal end-to-end CUDA runtime, real 8 GiB peak VRAM, 4B-class wall clock, or 405B completion.

## Required session startup

Before editing:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. `docs/SESSION_HANDOFF.md`
6. `docs/ROADMAP.md`
7. `docs/VALIDATION_PROTOCOL.md`
8. the active experiment document, workflow, PR comments, and result JSON

Then verify the current branch, PR head, CI/workflow conclusion, and latest evidence. Never assume a run mentioned in an earlier chat is still current.

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
10. Ensure the experiment workflow references only files that exist on its branch.
11. Run the full tests and validation and inspect actual logs.
12. Record positive and negative results in machine-readable form.
13. Update the research ledger, session handoff, and architecture documents.
14. Commit with a message describing the verified evidence level, not the hoped-for outcome.
15. Only then provide the user-facing progress answer.

## Immediate engineering priority

Use `docs/SESSION_HANDOFF.md` as the source of truth for the current active gate. Do not continue a family already rejected in `docs/RESEARCH_PROGRESS_LEDGER.md` without a new measurable mechanism that directly addresses its recorded failure.

The active candidate must always specify a complete path or a clearly isolated falsification question, including embeddings, attention/MLP projections, nonlinear operations, LM head, KV handling, storage, repair/fallback, and physical scheduling implications.

Do not call a component a core solution merely because it avoids a weight read on a replayed trace.

## Evidence rules

A result is accepted only when reproducible from committed code. Keep raw metrics in machine-readable JSON. Never rewrite a failed result as success; failed hypotheses are project data.

The final target may be declared achieved only after the real-hardware gates in `docs/VALIDATION_PROTOCOL.md` pass.
