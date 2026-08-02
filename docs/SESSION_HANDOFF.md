# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or architecture-specific adapter authoring;
- original-model quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence is below E4. Do not claim the target is solved or proven feasible.

## Mandatory startup

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. the active experiment document, branch, workflow, PR comments, and result JSON

Before editing, verify the current PR head and workflow state from GitHub. Do not trust an older chat status.

## Durable progress rule

After meaningful repository work and before a user-facing progress answer:

- update `docs/RESEARCH_PROGRESS_LEDGER.md`;
- update this file;
- commit raw evidence when available;
- record the active branch, PR, commit, workflow result, rejection reason, and exact next step.

This rule is permanent and does not require the user to repeat it.

## Current research frontier

Active family: nonlinear exact-neuron MLP allocation.

Active branch:

```text
research/nonlinear-heavy-hitter-allocation
```

Draft PR:

```text
#29 research: nonlinear layer-damage heavy-hitter allocation
```

Current code head after workflow-isolation fix:

```text
b7ad7aefb8ef8a64cc1979735e1c9ba487e944ac
```

The initial workflow run `30762648684` failed before pretrained measurement because it referenced `tests/test_adjoint_heavy_hitter.py`, which exists only on a sibling branch. Full repository CI passed. This is an infrastructure failure, not experimental evidence.

The workflow now:

- validates only `tests/test_mlp_heavy_hitter.py` and `tests/test_nonlinear_heavy_hitter.py`;
- asserts required test paths exist before pytest;
- remains isolated to its own branch and concurrency group.

## Active hypothesis

Uniform exact-neuron fractions and first-order adjoint allocation failed. However, their results showed layer sensitivity is nonuniform.

The active gate therefore measures the actual nonlinear damage of replacing one MLP layer at a time.

For each TinyLlama MLP layer and count:

```text
counts = {1, 4, 8, 16, 32, 64}
```

it performs a real model forward with only that layer replaced by the optimistic exact-activation original-neuron oracle and records the final exact-token cross-entropy damage.

A discrete dynamic program chooses one measured count per layer under total budgets corresponding to:

```text
0.10%, 0.25%, 0.50% of intermediate neurons
```

The chosen nonlinear allocation and an equal-total uniform allocation are then evaluated with all 22 TinyLlama MLPs replaced simultaneously on a disjoint Korean prompt.

## Promotion conditions

The exact-neuron family advances only when a disjoint point satisfies all of:

```text
projected 405B partial MLP traffic <= 1.6 GiB/token
teacher-forced top-32 >= 95%
autonomous exact prefix >= 4 tokens
```

The experiment remains an optimistic upper bound because it computes full exact gate/up activations before selecting neurons. A pass would still require a causal pre-load selector and a sound omitted-tail certificate.

## Prior decisive rejections

Read `docs/RESEARCH_PROGRESS_LEDGER.md` for details. Do not recreate these under new names:

- MLP centroid, gauge dictionary, and functional skeleton;
- unsigned residual decision bounds;
- global orthogonal residual proof sketches;
- fixed/adaptive exact LM-head row proofs;
- static prompt activation atlas;
- online proof atlas expansion;
- lossless entropy plus speculative ZIPTREE as the primary solution;
- uniform exact-neuron heavy hitters;
- first-order adjoint layer allocation.

Key measured constraints:

```text
online atlas: 32 expansions / 32 tokens, 2.9355 GiB/token LM-head residual traffic
ZIPTREE: 11.333 bits/weight, 10,649-token required straight acceptance
uniform 0.25% MLP oracle: 1.546 GiB/token, top-32 43.75%, prefix 0
adjoint 0.25%: top-32 56.25%, prefix 0, traffic about 1.638 GiB/token
```

## Exact next actions

1. Confirm that the workflow triggered from commit `b7ad7aef...`.
2. Inspect the branch-owned tests and actual workflow logs.
3. Complete all 132 single-layer damage measurements.
4. Commit `results/tinyllama_1_1b_nonlinear_heavy_hitter_allocation.json`.
5. Read the PR report and close or promote PR #29 with factual measured values.
6. Update the research ledger and this file before replying to the user.

If the nonlinear allocator fails without a traffic-compatible quality point, close the exact-neuron heavy-hitter family. The next candidate must not be another static activation basis or independent-neuron subset. It should test a reusable certified multi-layer decision influence cone or another representation that directly addresses cross-layer interaction.

## Validation commands

```bash
python -m pytest -q
python scripts/run_validation.py
python -m pytest -q tests/test_mlp_heavy_hitter.py tests/test_nonlinear_heavy_hitter.py
python scripts/run_nonlinear_heavy_hitter_allocation.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --count-options 1,4,8,16,32,64 \
  --total-fractions 0.001,0.0025,0.005 \
  --calibration-tokens 4 \
  --eval-tokens 16 \
  --output results/tinyllama_1_1b_nonlinear_heavy_hitter_allocation.json
```

## Correct communication

Use wording equivalent to:

> E1/E2 research: several low-dimensional, proof-bound, entropy, and exact-neuron candidates have been falsified on real TinyLlama replacements. The active nonlinear layer-damage gate is being rerun after fixing a branch-isolation workflow error. No 405B completion or end-to-end speed proof exists.
