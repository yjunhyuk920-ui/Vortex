# VORTEX Decision Log

All architecture decisions are append-only. Later corrections must reference the prior entry instead of deleting it.

## D-001 — Final target remains fixed

Date: 2026-08-03

Decision:

- arbitrary public unmodified Hugging Face dense model;
- runtime replacement only;
- 405B flagship;
- 8 GiB VRAM;
- original capability/output contract preserved;
- 4B-class user-perceived performance;
- independent reproducibility.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

Date: 2026-08-03

Decision:

GitHub Actions CPU results cannot be labeled as 8 GiB GPU, 405B, CUDA, PCIe, SSD, TTFT, or tokens/second measurements. All such target claims remain `NOT TESTED` until real hardware execution.

Status: ACTIVE, NON-NEGOTIABLE.

## D-003 — Adopt Phase A/B/C/D validation separation

Date: 2026-08-03

Decision:

Every experiment must declare one or more validation phases. A result may not inherit evidence from a later unexecuted phase.

Status: ACTIVE.

## D-004 — Adopt E0–E7 evidence scale

Date: 2026-08-03

Decision:

Use E0 idea, E1 synthetic/reference, E2 small real model, E3 held-out generalization, E4 accessible representative hardware improvement, E5 medium/large scaling, E6 8 GiB target execution, E7 405B at 4B-class performance.

Status: ACTIVE. This supersedes the repository's older E0–E4-only scale.

## D-005 — Separate MEASURED, DERIVED, PROJECTED, UNVERIFIED

Date: 2026-08-03

Decision:

Every metric and claim must carry one of these provenance labels. PROJECTED and UNVERIFIED values must never be presented as MEASURED.

Status: ACTIVE.

## D-006 — Reclassify decision-index replay work as auxiliary

Date: 2026-08-03

Evidence:

- PR #50 mmap VM;
- PR #52 bounded TinyLlama compiler;
- PR #54 exact future-suffix DAG.

Decision:

These are valid auxiliary representation and replay components. They do not directly answer how to skip original Transformer operations on unseen prompts. They cannot be the primary architecture without a new causal operation-skipping mechanism.

Status: ACTIVE.

## D-007 — Reject raw prefix enumeration as a core runtime

Date: 2026-08-03

Evidence:

- 64 path records produced 64 unique exact-prefix nodes excluding the intentional duplicate;
- held-out start coverage was 0%;
- first miss was step zero on all held-out prompts.

Decision:

Do not scale or rename raw prefix memoization as a universal runtime.

Status: REJECTED CORE MECHANISM.

## D-008 — Accept exact future-DAG compression only as body compression

Date: 2026-08-03

Evidence:

64 raw records reduced to 38 exact suffix-DAG nodes at horizon eight, but causal held-out start coverage remained 0%.

Decision:

Preserve the DAG implementation as an auxiliary component. Future-token oracle routing is forbidden.

Status: ACCEPTED AUXILIARY COMPONENT.

## D-009 — Make unseen-prompt causal operation skipping the primary research filter

Date: 2026-08-03

Decision:

A core research proposal must directly answer the twelve questions in `RESEARCH_STATE.md`, especially the skipped original operation, causal selector, verification, fallback, worst-case correctness, and 405B resource gap.

Status: ACTIVE.

## D-010 — Select EXP-047 CPTC as the next core falsification

Date: 2026-08-03

Decision:

Test whether sample-without-replacement, finite-population/martingale confidence certificates can exploit signed cancellation and safely avoid dense weight-tile reads, with full exact fallback when certification fails.

Reason:

Earlier deterministic residual norms were too conservative and required near-complete refinement. CPTC directly tests whether statistical cancellation can provide a valid lower-cost certificate rather than renaming the failed deterministic method.

Status: ACTIVE E0; Phase A/B pending.
