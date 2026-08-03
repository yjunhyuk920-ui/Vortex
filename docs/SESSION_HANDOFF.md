# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or architecture-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the target is solved.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, workflows, PR comments, and raw result JSON

Before a user-facing progress/completion answer after meaningful work, commit the ledger and this handoff.

## Current repository state

Latest research decisions:

```text
PR #36  causal semantic-state program routing             rejected
PR #37  prompt-compiled Hankel decision program            rejected
PR #38  perfect-oracle sparse Hankel repair                rejected
PR #40  prompt-only nonlocal exact decision memory         rejected
PR #42  exact dense-operator lower-bound certificate       accepted and merged
```

Main merge for PR #42:

```text
663dd3d02095f19be269ef60a7c16959f6e16f2f
```

Authoritative PR #42 evidence:

```text
branch: research/exact-operator-lower-bound
head:   7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate workflow: 30781557141
Python 3.10/3.12 CI + validation: 30781557096
```

The bot evidence commit triggered follow-up `action_required` runs with no jobs. Those are not test failures.

## Experiment 040 accepted result

### Exact-output information theorem

For `N` independently selectable `b`-bit parameter codes, the arbitrary checkpoint family has size:

```text
2^(N b)
```

Any checkpoint-specific representation supporting every exact dense operator output must be injective, so its worst-case size is at least:

```text
N b bits
```

For the 405B target at Q4:

```text
parameters: 405,849,243,648
exact information: 1,623,396,974,592 bits
                 = 188.98828125 GiB
resident allowance: 8 GiB
resident fraction: 4.2330667%
minimum external exact information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
4B dense arithmetic proxy: 8 GFLOP
ratio: 101.462310912x
```

Accepted conclusion:

> Arbitrary dense exact-output execution is incompatible with retaining only 8 GiB of checkpoint information unless the remaining exact information is read or equivalently represented and its transfer/construction cost is charged or amortized.

### Skipped-coordinate exact top-1 adversary

For every tested coordinate, the executable gate constructed `W0/W1` that:

- differ only at one uninspected/unrepresented coordinate;
- have identical inspected observations;
- produce different exact outputs for a one-hot input;
- produce different unique top-1 winners.

Coverage:

```text
2x4, 3x5, 4x7, 8x8 matrices
115 coordinates tested
115 adversaries passed
100% coverage
```

Accepted conclusion:

> No arbitrary weight coordinate may be declared universally irrelevant to exact top-1 unless its effect is read or represented.

## Critical scope boundary

Experiment 040 deliberately did **not** prove all of the following:

```text
metadata-aware exact top-1 representation requires N*b bits: not proven
measured 405B GPU wall clock: not performed
real 405B execution: not performed
```

The exact-output cardinality theorem must not be misreported as a complete top-1-only information theorem.

Current fixed-target classification:

```text
arbitrary dense exact operator output + 8 GiB only: contradicted
coordinate omission without metadata for universal top-1: contradicted
exact top-1 via charged checkpoint-specific decision metadata: conditionally open
405B/8 GiB/4B-speed runtime: unsolved
```

## Accumulated representation evidence

```text
semantic-state program reuse: about 1 token
prompt recurrence: at most 2 autonomous exact tokens
perfect-token recurrence repair: exact target on 68%–89% of tokens
prompt suffix global-oracle maxima: 75 / 28 / 5
required full-interaction reuse: about 247 tokens
```

Prompt-derived static, dynamic, repaired, and nonlocal programs are exhausted at the tested scale.

## Prohibited repeats

Do not continue by only changing:

- static rank/block size/state count;
- norm precision or neuron ordering;
- Hankel rank/order/ridge/lift;
- recurrence repair thresholds;
- ANN rank/index/distance/top-k width;
- speculative block size while dense target arithmetic remains per position;
- lossless metadata that omits its exact information content or build/transfer cost.

## Current frontier — Experiment 041 Metadata-Aware Top-1 Function Bound

The remaining loophole is a checkpoint-specific representation that does not reproduce exact internal operators but preserves the complete exact top-1 decision function.

The next Gate must count **distinct top-1 functions**, not distinct weight tensors.

### Candidate injective classifier family

For an `m x d` dense classifier, choose:

```text
p = min(floor(m/2), floor(d/2))
q = d - p
```

Use `p` selector input coordinates and `p` row pairs. For every pair `r` and payload coordinate `j`, encode one independent bit `a[r,j]` by assigning the payload advantage to one row or the other. Query:

```text
x_(r,j) = selector_r + payload_j
```

A large fixed selector margin suppresses every other row, and the exact top-1 winner reveals `a[r,j]`.

This constructs:

```text
K = p q independent decision bits
2^K distinct exact top-1 functions
minimum metadata for this family >= K bits
```

For a square `H x H` classifier with even `H`:

```text
K = H^2 / 4 = N / 4 bits
```

This lower bound already includes arbitrary checkpoint-specific metadata because two checkpoints with different encoded bit tables implement different top-1 functions and therefore cannot share one exact representation.

### Required Experiment 041 work

1. Formalize the selector/payload construction and prove unique winners.
2. Implement encode/decode queries and exhaustive small-shape tests.
3. Enumerate all `2^K` functions for small matrices and verify injectivity.
4. Compute `K=p(d-p)` for target attention/MLP matrix shapes.
5. Keep three conclusions separate:
   - direct dense-classifier top-1 metadata lower bound;
   - independently callable operator-collection bound;
   - full end-to-end transformer decision-function bound.
6. Do not sum layerwise bits into a full-transformer theorem until an explicit Llama-like routing construction exposes each layer's independent bits through final token decisions.
7. Add workflow and raw JSON evidence.

## Exact next steps

1. Merge the documentation update after full CI.
2. Create `research/top1-function-information-bound` from the new `main`.
3. Add `docs/EXPERIMENT_041_TOP1_FUNCTION_INFORMATION_BOUND.md`.
4. Implement the injective top-1 classifier family and tests.
5. Produce shape-level and 405B operator-shape budgets.
6. Record exactly whether the result exceeds 8 GiB for a direct operator collection and what remains to extend it to a full transformer.
7. Update the ledger and handoff before the next progress response.

## Correct communication

Use wording equivalent to:

> Experiment 040 proved that arbitrary Q4 405B exact operator output carries 188.99 GiB of worst-case checkpoint information and that every unrepresented coordinate can adversarially flip top-1. It did not prove a full metadata-aware top-1 N*b lower bound. The original runtime objective remains unsolved. Experiment 041 now counts distinct exact top-1 decision functions using an injective selector/payload classifier family.
