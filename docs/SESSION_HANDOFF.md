# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or architecture-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the runtime target is solved.

## Mandatory startup

Read:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, workflows, PR comments, and raw JSON

Before a user-facing progress/completion answer after meaningful work, commit the ledger and this handoff.

## Current repository state

Latest decisions:

```text
PR #36  semantic-state program routing                    rejected
PR #37  prompt-compiled Hankel decision program           rejected
PR #38  perfect-oracle sparse repair                      rejected
PR #40  prompt-only nonlocal exact decision memory        rejected
PR #42  exact dense-operator lower-bound certificate      accepted/merged
PR #44  metadata-aware exact top-1 function bound         accepted/merged
```

Latest main merge:

```text
PR #44 merge: aca6657578b0decb58adbf98bcd22555169a6847
```

Authoritative Experiment 041 evidence:

```text
branch: research/top1-function-information-bound
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate workflow: 30782192795
Python 3.10/3.12 CI + validation: 30782192768
```

## Accepted Experiment 040 result

For arbitrary Q4 405B exact dense operator output:

```text
exact checkpoint information: 188.98828125 GiB
resident allowance: 8 GiB
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
ratio to 4B dense arithmetic: 101.462310912x
```

The injectivity/cardinality theorem proves the exact-output representation needs `N*b` bits in the worst case. The skipped-coordinate gate produced 115/115 indistinguishable-observation adversaries with different exact outputs and top-1 winners.

Scope:

```text
exact-output N*b bound: proven
coordinate omission can flip top-1: proven
metadata-aware complete top-1 N*b bound: not proven by Experiment 040
```

## Accepted Experiment 041 result

### Direct classifier theorem

For an `m x d` dense classifier:

```text
p = min(floor(m/2), floor(d/2))
q = d - p
K = p q
```

The selector/payload family encodes `K` independent bits. Each bit has one query whose unique top-1 winner reveals it. Therefore the family has `2^K` distinct top-1 decision functions and any exact checkpoint-specific metadata for the family needs at least `K` bits.

Exhaustive certificate:

| Shape | K | Expected functions | Observed | Minimum margin |
|---|---:|---:|---:|---:|
| 2x2 | 1 | 2 | 2 | 1.0 |
| 4x4 | 4 | 16 | 16 | 1.0 |
| 4x5 | 6 | 64 | 64 | 1.0 |
| 6x6 | 9 | 512 | 512 | 1.0 |

All encoded bit tables decoded exactly from the winner signatures.

### Llama-405B-shaped independently callable operator collection

Per decoder layer:

```text
Q:    67,108,864 bits
K:     8,126,464 bits
V:     8,126,464 bits
O:    67,108,864 bits
gate: 67,108,864 bits
up:   67,108,864 bits
down:369,098,752 bits
sum: 653,787,136 bits = 77.9375 MiB
```

Projection:

```text
126-layer stack: 9.5899658203125 GiB
directly callable LM head: 8 MiB
total: 9.5977783203125 GiB
excess over 8 GiB: 1.5977783203125 GiB
```

Accepted conclusion:

> Arbitrary checkpoint-specific exact top-1 metadata for the constructed independently callable Llama-shaped operator collection is lower-bounded above 8 GiB.

## Critical scope boundary

Experiment 041 did not prove:

```text
full end-to-end Transformer final-token bound
real 405B execution
measured GPU wall clock
```

Layerwise/operator bounds cannot be summed into a final language-model theorem until one explicit Llama-like construction exposes their independent bits through final token winners.

Current classification:

```text
arbitrary dense exact output with 8 GiB only: contradicted
metadata-aware exact top-1 for direct dense classifiers: lower-bounded
independently callable Llama-shaped operator collection: >8 GiB lower bound
full Transformer final-token metadata bound: open
405B/8 GiB/4B-speed runtime: unsolved
```

## Accumulated execution evidence

```text
semantic program reuse: about 1 token
prompt recurrence: maximum 2 exact tokens
perfect-token repair: exact target on 68%–89% of tokens
future-aware prompt suffix oracle: 75 / 28 / 5 tokens
required full-interaction reuse: about 247 tokens
```

## Prohibited repeats

Do not continue by only changing static rank, recurrence order, repair thresholds, ANN settings, speculative block length, or uncharged lossless metadata. Do not report the operator-collection bound as a full Transformer theorem.

## Current frontier — Experiment 042 End-to-End Llama Decision Routing Bound

The next Gate must embed independent selector/payload decision bits inside an actual Llama-like residual/attention/MLP composition and expose them through final vocabulary top-1 decisions.

### Proof target

Construct a quantized Llama-style family with:

- RMSNorm;
- causal self-attention;
- residual connections;
- SwiGLU MLP;
- final norm and LM head;
- multiple layers carrying independent bit tables.

For every encoded layer/operator bit, provide a legal token sequence/query whose final next-token winner reveals that bit with a strictly positive margin.

If `K_total` independent layerwise bits are exposed through final token decisions, the family has `2^K_total` distinct language-model decision functions and any exact checkpoint-specific metadata needs at least `K_total` bits.

### Required work

1. Start with a minimal exact Llama-like micro-model, not a generic operator collection.
2. Construct selector channels and payload channels that survive RMSNorm with known scale.
3. Either:
   - route bits through attention values and output projection, or
   - route through a SwiGLU configuration with a rigorously bounded positive margin.
4. Make unselected layers/operators contribute a checkpoint-independent constant.
5. Verify all small bit tables exhaustively against final vocabulary winners.
6. Prove additivity across at least two layers before scaling the symbolic count.
7. Extend the count to target hidden/intermediate/layer shapes only after the small end-to-end construction passes.
8. Keep runtime success, information theorem, and hardware measurement separate.

## Exact next steps

1. Merge this documentation update after full CI.
2. Create `research/llama-final-decision-routing-bound`.
3. Add `docs/EXPERIMENT_042_LLAMA_FINAL_DECISION_ROUTING_BOUND.md`.
4. Implement a deterministic Llama-like micro-model and exhaustive function counter.
5. Add tests, workflow, and raw JSON.
6. Update the ledger and handoff before reporting further progress.

## Correct communication

> Experiment 041 proved a metadata-aware exact top-1 lower bound of 9.5978 GiB for an independently callable Llama-405B-shaped operator collection, but not yet for final Transformer token decisions. The runtime objective remains unsolved. Experiment 042 must expose independent layerwise bits through an actual Llama-like final-token winner before the lower bound can be promoted to the full model.
