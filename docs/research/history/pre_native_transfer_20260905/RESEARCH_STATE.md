# VORTEX Research State

Snapshot: 2026-09-05, on the PR #118 lineage based at
`ff70c1ebbca943161684f33e6afa1dc169fa8f4c`.

## Mission unchanged

Arbitrary public unmodified dense 405B; executor replacement only; single 8 GiB
GPU; original output/RNG and exact or proven-bisimilar successor state;
same-machine native-4B-Q4 warm p50 <=1.2x and p95 <=1.5x. Full costs count.

## Latest recorded scientific result (not rerun for this policy edit)

`SCOPED_2X2_MINIMUM_PROVED_NO_EXECUTOR_PROMOTED`.

[Proof, decision, assumptions and handoff](docs/research/E0_GLOBAL_DECODER_KERNEL_AUDIT.md)
with [raw/processed evidence](results/e0_global_decoder/summary.json).
A two-probe exact decoder for all 2x2 binary rank-one queries needs at least five
stored bits, even under arbitrary nonlinear encoding. Five bits suffice. The
proof does not apply to redundant seven-bit encoders, arbitrary word probes,
restricted public checkpoints or native Transformer states.

Prior research validation: 18 tests passed; independent unittest run also passed; exact
summary regeneration. E1 ceiling. The bounded general seven-bit 2x3 solver
returned UNKNOWN/timeout; no existence/nonexistence conclusion.

The inherited PR #118 decision remains `NO_ROUTE_PROOF_REACH_CORE_PROMOTED`.
EXP-102A is the latest completed numbered real-model Gate on this lineage;
its frozen causal external-draft source is rejected. Separate EXP-103A..108A
branches are not merged or superseded by this snapshot. The EXP-103A and
EXP-108A state snapshots were cross-checked.

## Active constructive requirement — superseded by CTC-2026-09-05

Start from the final goal theorem and O1-O6 ledger in
[NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md), compare three materially different
principles, and close the strongest survivor's missing causal/native/state/cost
construction. The global-code route is one candidate, not the mandated architecture.
An isolated parity result or a larger timeout is not a core completion.

[Governance decision](docs/governance/CTC_20260905_DECISION.md): policy-only;
prior scientific evidence and classifications above are unchanged.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
```

These are full-mission statuses, not denial of the scoped auxiliary theorem.
Remote handoff status must be established by actual post-commit read-back.

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
