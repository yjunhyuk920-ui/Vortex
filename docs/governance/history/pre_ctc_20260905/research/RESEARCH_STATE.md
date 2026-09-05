# VORTEX Research State

Snapshot: 2026-09-05, on the PR #118 lineage based at
`ff70c1ebbca943161684f33e6afa1dc169fa8f4c`.

## Mission unchanged

Arbitrary public unmodified dense 405B; executor replacement only; single 8 GiB
GPU; original output/RNG and exact or proven-bisimilar successor state;
same-machine native-4B-Q4 warm p50 <=1.2x and p95 <=1.5x. Full costs count.

## Current result

`SCOPED_2X2_MINIMUM_PROVED_NO_EXECUTOR_PROMOTED`.

[Proof, decision, assumptions and handoff](docs/research/E0_GLOBAL_DECODER_KERNEL_AUDIT.md)
with [raw/processed evidence](results/e0_global_decoder/summary.json).
A two-probe exact decoder for all 2x2 binary rank-one queries needs at least five
stored bits, even under arbitrary nonlinear encoding. Five bits suffice. The
proof does not apply to redundant seven-bit encoders, arbitrary word probes,
restricted public checkpoints or native Transformer states.

Local validation: 18 tests passed; independent unittest run also passed; exact
summary regeneration. E1 ceiling. The bounded general seven-bit 2x3 solver
returned UNKNOWN/timeout; no existence/nonexistence conclusion.

The inherited PR #118 decision remains `NO_ROUTE_PROOF_REACH_CORE_PROMOTED`.
EXP-102A is the latest completed numbered real-model Gate on this lineage;
its frozen causal external-draft source is rejected. Separate EXP-103A..108A
branches are not merged or superseded by this snapshot. The EXP-103A and
EXP-108A state snapshots were cross-checked.

## Active constructive requirement

A non-entrywise redundant global encoding with an explicit causal decoder,
nontrivial native successor-state path, two-scale same-grammar validation, and
fully charged construction/storage/physical-query/compute/verification costs.
See [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md). More tiny negative tests or a larger
SMT timeout are not the next authorized core task.

## Acceptance truth

| Item | Status |
|---|---|
| Complete 405B execution | NOT TESTED / NOT ACHIEVED |
| Complete physical peak allocation <=8 GiB | NOT TESTED |
| Same-machine native-4B p50/p95 | NOT TESTED |
| Public Transformer layer replacement in this round | NOT PERFORMED |
| General nonlinear word-probe impossibility | NOT ESTABLISHED |
| Seven-bit 2x3 encoder | UNRESOLVED |
| Native numerical/state lift | NOT CONSTRUCTED |

The local CPU sandbox did not have a complete repository checkout or a cached
public checkpoint; no target machine was modified. Full repository regression
and target hardware were not run. No Actions job was dispatched.

Historical [full state ledger](docs/research/history/pre_global_decoder_20260905/RESEARCH_STATE.md)
is preserved as its original Git blob, not overwritten or summarized away.
