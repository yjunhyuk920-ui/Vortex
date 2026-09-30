# Whole-program causal source: bounded theory plan

Date: 2026-09-30 UTC. This plan is recorded after initial analytical exploration
and before final derivation, report, and document validation. It is not presented
as a preregistration preceding that exploration. No numerical experiment is
proposed, run, or admitted.

## Fixed final theorem

For every checkpoint in the unchanged arbitrary public, unmodified HF dense
Transformer mission, every legal input/context and original RNG/state, construct
an automatic finite causal executor preserving every exposed output (including
logits when exposed), RNG consumption and required successor-state relation for
all legal continuations. Batch one; one GPU with total peak <=8 GiB; warm
same-machine native 4B Q4 p50 <=1.2x and p95 <=1.5x; existing TTFT requirement.
All construction, original/transformed weights, CPU/RAM/SSD/PCIe/HBM, state,
metadata, queries, movement, verification, repair and fallback costs count.
Actual ABI, workload population, context limits, baseline and statistical
protocol must be frozen before acceptance; this package does not invent them.

## Question and three principles

Can the entire causal transition be evaluated without assuming that every dense
projection must first produce all its internal native states? Compare:

1. Minimize the reachable transition system by exact continuation equivalence
2. Eliminate native instruction relations across the complete observed transition
3. Maintain an exact previous trace and propagate only actual native word changes

These are different execution principles, not a claim of three new qualifying
cores or novel mathematical techniques. Compare each to recorded failures and
reject aliases rather than reopening them. Concentrate on relational elimination
only if it exposes a concrete new construction/cost question; an algorithm with
an exponential boundary table does not qualify as a cheap source.

## O1-O6 at entry

- O1 OPEN: the original runtime is finite on admitted calls; a uniform cheap
  compiler and exact support for every admitted public checkpoint are absent
- O2 OPEN: complete cheap prefill/output/state transition is absent
- O3 OPEN: scalar ABI results do not prove the complete reference ABI/state/RNG
- O4 OPEN: no sufficient paid whole-program source upper bound
- O5 OPEN: no target allocation, TTFT or same-machine p50/p95 closure
- O6 OPEN: final theorem artifacts absent; present work can provide scoped proofs

## Decision and validation rules

No core promotion unless a finite constructor/query supplies a paid >=10x route
and the complete unchanged model/state/target path. Conditional factors, no-op
controls, proof counts, tables with ignored storage, or tiny tests do not count.

Allowed checks are symbolic derivation and document/provenance/link/hash checks.
No model, GPU, large numerical, solver search, backend, remote write, or root-ledger
edit. Parent owns integration and publication. Preserve existing failures and
original artifacts. Stop if no further qualified route can be constructed;
record the actual scientific blocker, not an impossibility claim.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
HANDOFF_STATUS=IN_PROGRESS
