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
- every core round starts with three materially different principles;
- research and validation are local first, followed by commit/push and remote-SHA
  verification; duplicate GitHub Actions execution is not required.

## Workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

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

Full width removes the collision but requires
`202.377974 GiB` resident and
`214.493273 GB` per proposed token. A single
checkpoint bitplane costs `47.247070312 GiB`, so the bit-sliced
all-layer arm also fails.

Previous EXP-105A rejected the activation-ordered exact residual-bound head
index when combined with the minimum complete target-width layer.

## Active frontier — EXP-107A

`Exact Distinct-State Shared-Weight-Sweep Gate` asks whether distinct causal
states can share one value-changing coded weight computation before nonlinear
separation. It must fully charge branch matrices, exact encoding/decoding,
weight traffic, arithmetic, nonlinear cells, KV/state, workspace, actual `N/A`,
and target verification. Ordinary batching receives no credit.

Read `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, and
`docs/research/VORTEX_RESEARCH_HANDOFF.md` before continuing.

## Claim boundary

Complete 405B execution, physical complete 8-GiB allocation, target CUDA/SASS,
target SSD/PCIe/HBM behavior, and final same-machine p50/p95 remain `NOT_TESTED`.

```text
README_CURRENT=true
README_UPDATED=EXP-106A result; EXP-107A frontier; local-validation/commit-only workflow retained
```
