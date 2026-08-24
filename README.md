# VORTEX

VORTEX researches an executor-only path for arbitrary public, unmodified dense
Hugging Face 405B-class models on one 8-GiB GPU while preserving exact declared
output/successor-state behavior and approaching same-machine native-4B-Q4
latency (`p50<=1.2x`, `p95<=1.5x`).

## Non-negotiable rules

- no retraining, fine-tuning, distillation, LoRA, or semantic target-weight modification;
- `REAL_EXECUTOR_ONLY` is authoritative;
- all storage, traffic, arithmetic, KV/state, metadata, verification, repair,
  fallback, packing, and synchronization are charged;
- future target state, perfect selectors, free transforms/workspace, unmeasured
  compression, hidden compute, and configured `K` reported as accepted `A`
  cannot promote a mechanism;
- research and validation are local first, followed by commit/push and remote-SHA
  verification; duplicate GitHub Actions execution is not required;
- README freshness is part of round completion.

## Research selection: no homework-mode experiments

Every core round starts by inventing three materially different principles, but
none is implemented merely because the process requires three ideas. Before a
new EXP number or implementation, each principle receives a pre-result technical
prior:

```text
PRIOR=<HIGH|MEDIUM|LOW>
WHY_IT_MIGHT_WORK
WHY_IT_MIGHT_FAIL
DOMINANT_TERM_CHANGED
MAX_IMPACT_IF_TRUE
ARBITRARY_CHECKPOINT_ARGUMENT
CHEAPEST_KILL
FALSIFICATION_COST
IMPLEMENTATION_COST
DECISION=<GO|NO_GO|CHEAP_KILL_ONLY>
```

Research priority follows the heuristic

\[
\text{Research Value}\propto
\frac{P(\text{works})\times\text{impact if true}}
{\text{cost to falsify}}.
\]

This is an ordering heuristic, not a claim of measured probability. Candidates
that cannot reach a 10× core effect even under success, leave the dominant cost
unchanged, already have a simple arbitrary-dense counterexample, or merely
retune a closed family are killed before implementation.

A low-prior moonshot is eligible as `CHEAP_KILL_ONLY` when success would remove a
governing cost term and falsification is cheap. But `CHEAP_KILL_ONLY` is not a
research result: it means the cheapest Gate must be run immediately in the same
round.

```text
CHEAP_KILL_ONLY
-> run CHEAPEST_KILL now
-> FAIL: extract failure premise and return to ideation
-> PASS: continue to next decisive validation
```

Likewise `GO` means only “worth implementing/testing.” It is not a round exit.

```text
GO
-> implement minimum decisive mechanism
-> run frozen local validation now
-> FAIL: extract failure premise and return to ideation
-> PASS: VALIDATED_SURVIVOR for this Gate
```

### Three ideas are the minimum batch, not the end of the search

If all three candidates are `NO_GO`, or every candidate that entered validation
fails, the round does **not** end. Extract the common premise that killed them,
invert/remove that premise, and generate a new batch of three materially
different principles.

A new batch is valid only when it materially changes at least one of the
information source, computation order, verification unit, state representation,
weight-access dependency, causal schedule, or cross-token/cross-layer sharing
mechanism. Parameter sweeps, renames, and nearby variants of a closed family do
not count.

### Failed validation returns to ideation immediately

The mandatory closed loop is:

```text
IDEATE
-> PRIOR SCREEN
-> TEST
-> FAIL: IDEATE AGAIN IN THE SAME ROUND
-> PASS: NEXT DECISIVE VALIDATION
```

Do not discover a `CHEAP_KILL_ONLY` or `GO` candidate, postpone its actual Gate
to the next EXP, and end the current research round. A promising question is not
a validated result.

### Only two valid core-round exit conditions

A core research round may end only as:

```text
VALIDATED_SURVIVOR
STRUCTURAL_CLOSURE
```

`VALIDATED_SURVIVOR` means a candidate actually passed the frozen decisive Gate
for the current rung with required exactness and fully charged accounting.

`STRUCTURAL_CLOSURE` means a separately recorded theorem, finite lower bound,
information argument, or exhaustive structural result closes the remaining
admissible design space under the fixed mission.

The following are not valid exits:

```text
three ideas failed
a promising paper was found
CHEAP_KILL_ONLY was found
GO was assigned
the next EXP will test it
```

Normative detail:
[`docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`](docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md).

## Workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Research, checkpoint execution, tests, measurements, and evidence generation
happen locally. GitHub stores the validated source/results and handoff. Duplicate
GitHub Actions execution is not required unless explicitly requested.

## Latest authoritative result — EXP-106A

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

A concrete all-126-layer Q4 width-thin surrogate was fully charged. Widths up to
1,664 can fit the resident and scan-alone budgets, but every resource-feasible
linear bridge has a nonzero kernel. Exact finite-word controls found distinct
states with byte-identical narrow codes and different target decisions. A dense
flat-spectrum operator also retains relative operator-norm error 1 under every
rank-deficient approximation.

Full width removes the collision but requires `202.377974 GiB` resident and
`214.493273 GB` per proposed token. A single checkpoint bitplane costs
`47.247070312 GiB`, so the bit-sliced all-layer arm also fails.

Previous EXP-105A rejected the activation-ordered exact residual-bound head
index when combined with the minimum complete target-width layer.

## Active frontier — EXP-107A

`Exact Distinct-State Shared-Weight-Sweep Gate` asks whether distinct causal
states can share one value-changing coded weight computation before nonlinear
separation. It must fully charge branch matrices, exact encoding/decoding,
weight traffic, arithmetic, nonlinear cells, KV/state, workspace, actual `N/A`,
and target verification. Ordinary batching receives no credit.

Before any EXP-107A implementation, candidate batches must pass the research-
prioritization screen. Any `CHEAP_KILL_ONLY` Gate or `GO` validation found in that
search must be executed in the same round; if it fails, the round returns to a
new principle batch rather than deferring the test to a future EXP.

Read `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`,
`docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`, and
`docs/research/VORTEX_RESEARCH_HANDOFF.md` before continuing.

## Claim boundary

Complete 405B execution, physical complete 8-GiB allocation, target CUDA/SASS,
target SSD/PCIe/HBM behavior, and final same-machine p50/p95 remain `NOT_TESTED`.

```text
README_CURRENT=true
README_UPDATED=mandatory validation-return loop; CHEAP_KILL_ONLY/GO are non-terminal; round exits only on VALIDATED_SURVIVOR or STRUCTURAL_CLOSURE
```