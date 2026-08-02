# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## Mandatory first read

Read `docs/PROOF_FIRST_CONTRACT.md` and `AGENTS.md` before proposing or implementing a new architecture.

The project previously promoted local prototype successes too aggressively. That process has been corrected. No component result may be described as proof of the 405B/8GiB/4B-speed target unless the evidence level and flagship resource equations justify it.

## Current verified evidence level

The repository currently has E1 primitives only:

1. disk-backed progressive LM-head certification;
2. tiny-model streamed Llama execution;
3. tiny-model Jacobi sequence equivalence;
4. `OnlineAtlasLinear`, which exactly replays inputs inside a cached span using `U` and `WU`;
5. tiny-Llama O/down projection replay with persisted atlas state.

Validation command:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Last observed result:

```text
10 passed
validation script completed successfully
```

These results prove their local tested properties only. They do not prove unseen-prompt generalization, model-wide scaling, a CUDA runtime, 8GiB peak VRAM, or native-4B-class wall-clock behavior.

## Why the previous next task was cancelled

The previous handoff proposed extending Atlas to more projections before establishing the full 405B resource budget. That order is no longer allowed.

An exact-span atlas may grow toward the input dimension, and a model-wide collection of FP32 `U/WU` capsules may exceed the VRAM budget. Hook-based activation analysis also does not measure actual end-to-end replacement speed. Therefore adding Q/K/V and gate/up first would repeat the same implementation-before-proof error.

## Exact next task — Architecture Gate 0

Do not begin by adding more operators.

Produce a complete model-wide candidate execution path and a committed feasibility certificate containing:

```text
M_hot + M_kv + M_work + M_repair <= 8 GiB
B_hot + B_cold / A <= 1.2 * B_4B
C_hot + C_repair / A <= 1.2 * C_4B
```

The candidate must account for:

- embeddings and LM head;
- Q/K/V/O and gate/up/down;
- RMSNorm, RoPE, attention, softmax, SiLU, residuals;
- KV storage/offload/compression;
- persistent runtime representation;
- cold exact fallback or repair;
- host/storage/GPU transfer scheduling;
- CUDA workspaces and allocator reserve;
- prefill and decode separately;
- build time, cold start, and warm decode.

The certificate must define `A`, the number of committed tokens amortized per cold stream, and show where that number comes from rather than assuming it.

## Required falsification harness

After the equations are committed, implement only the smallest experiment that measures their unknown terms on a real pretrained 1B–3B model.

The test must:

1. replace the real operation during generation rather than use hooks alone;
2. use disjoint build and evaluation prompts;
3. record actual bytes/token, compute/token, fallback rate, representation growth, peak memory, token agreement, and wall-clock;
4. compare against the exact target path and a native 4B-class baseline where applicable;
5. reject the architecture when the measured scaling slope cannot close the flagship inequalities.

## Communication rule

Until E4 completion, report only the exact evidence level and tested scope.

Correct example:

> E1: Atlas replayed selected tiny-model projections on a previously built activation span. Full-target feasibility is not established.

Forbidden example:

> The core solution works and will make 405B run at 4B speed.

## Files that must change in the next architecture session

- a new feasibility-certificate document with formulas and populated budgets;
- a machine-readable budget JSON;
- a falsification script for the unknown terms;
- tests for the budget calculations;
- this handoff document after the experiment;
- `docs/ARCHITECTURE.md` only after the candidate passes Architecture Gate 0.