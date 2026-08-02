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

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, branch, workflow, PR comments, and result JSON

Before a user-facing progress/completion answer after meaningful work, commit the research ledger and this handoff. This rule is permanent and is already merged into `main` through PR #30.

## Current state

There is no promoted research PR. The latest experimental sequence is closed with raw evidence:

```text
PR #29  nonlinear exact-neuron allocation              rejected
PR #31  global-bound Signed Dual Cone                  rejected
PR #32  partitioned norm-only Signed Dual Cone         rejected
PR #33  disjoint Block Signed Residual Code            rejected
PR #34  cross-layer Global Margin Refinement           rejected
```

Latest evidence head:

```text
research/global-margin-refinement
b6c6bb25bfe8e93e630293b561c7dcb442e81320
```

Latest successful workflow:

```text
Global margin refinement gate run 30766038116
conclusion: success
```

## Latest measured result

Using 8-bit hot MLP weights and the strongest static signed residual code from PR #33:

```text
block size: 1024
signed rank: 2
metadata: 3.6298828125 GiB
```

PR #34 compared three exact-refinement policies on disjoint TinyLlama warm-decode traces:

```text
equal per-layer mean refinement: 92.39491864669421%
global interval-width refinement: 90.74485085227273%
dual-price two-sided refinement: 90.74323669938016%
maximum dual-price refinement: 93.33919808884298%
maximum projected 405B exact traffic: 573.3446044921875 GiB/token
```

The dual-price rule was:

```text
lower uncertainty l_i = a_i - L_i
upper uncertainty u_i = U_i - a_i
score_i(lambda) = lambda l_i + (1-lambda) u_i
```

with 41 prices in `[0,1]` and global constraints:

```text
sum unrefined l_i <= 0.5 * exact top-two margin
sum unrefined u_i <= 0.5 * exact top-two margin
```

The global formulation was sound and recovered some layer slack, but only about 1.65 percentage points. It remained more than two orders of magnitude outside the partial MLP traffic gate.

## Decisive interpretation

The tested static representations are closed:

- magnitude-only global or block bounds;
- static activation or dual subspaces;
- static signed residual codebooks built from disjoint prompts;
- uniform, adjoint, nonlinear, or globally reordered independent-neuron refinement.

Signed cancellation is a real useful signal: PR #33 reduced gate/up residual radii to about 69.2% of global. However, the disjoint activation and dual state remained mostly outside the static build span, leaving exact refinement above 90% even after globally optimalized error allocation.

Do not create another candidate that only changes:

- static basis rank;
- static block size;
- norm metadata precision;
- neuron ordering;
- per-layer versus global budget;
- build-prompt dictionary size;

unless it introduces and charges a fundamentally new multi-token reuse mechanism.

## Next architecture frontier

Derive both candidates on paper before choosing one.

### Candidate A — Semantic-state-keyed signed residual program

At token `t`, use a small resident state signature to select or compose a signed residual program before exact weight reads.

Required proof obligations:

```text
M_program + M_index + M_KV + M_work <= 8 GiB
B_program_build / A + B_exact_refine <= 1.2 * B_4B
C_program_build / A + C_hot <= 1.2 * C_4B
```

`A` must be measured as the number of future tokens reusing the same program. Tokenwise full residual construction is immediate rejection.

### Candidate B — Multi-token decision program

One exact target interaction produces a certified program for several future token decisions or a verified token tree.

Required measurements:

```text
accepted/committed tokens per exact target interaction
verified positions per committed token
exact target bytes per interaction
program memory and construction compute
```

ZIPTREE already proved that whole-model FP16 streaming would require an unrealistic 10,649-token straight run at measured lossless entropy. The new program must avoid whole-model exact streaming.

## Exact next steps

1. Create a fresh proof-first branch from `main` after this documentation update is merged.
2. Write one architecture certificate comparing Candidate A and Candidate B.
3. Calculate the minimum reuse factor for each at 405B.
4. Implement only the candidate with a plausible symbolic closure.
5. Use disjoint multi-token TinyLlama traces and charge every program-build read.
6. Persist raw JSON, PR decision, ledger, and this file before reporting progress.

## Correct communication

Use wording equivalent to:

> E2 research has falsified static activation, residual-bound, signed-codebook, and exact-neuron families on real TinyLlama operations. Signed cancellation reduces bounds but static programs do not transfer enough; even global refinement projects to 573 GiB/token. The objective remains unchanged and no 405B end-to-end proof exists. The next work is a proof-first comparison of semantic-state-keyed versus multi-token decision programs.
