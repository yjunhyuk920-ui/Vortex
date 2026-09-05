# VORTEX Research State — 2026-09-05 native-transfer continuation

Base: `947b307da98b75f7843babbed691c70bd3caa964` on the PR #119 lineage.
Mission and CTC-2026-09-05 unchanged: arbitrary public unmodified dense 405B,
executor only, one total <=8-GiB GPU, original output/RNG/successor state,
same-machine native 4B Q4 p50 <=1.2x / p95 <=1.5x and existing TTFT requirement.

## Actual result

Finite native numerical subroutines now exist: guarded parity-phase ordered-FMA
composition; causal suffix erasure only on equal native interval endpoints;
exact direct rounding-cell preimages with explicit signed-zero handling.
[Proof, finite costs, scope and handoff](docs/research/NATIVE_TRANSFER_CONSTRUCTION.md).

This is **scoped E1 constructive discovery, not a promoted core**. Summary
construction remains linear in products/reads. Ordinary synthetic suffix rows
(36/36) failed early certification and paid all weights plus 2.75x–2.99609375x
FMA work. Direct inverse construction removes searches at nonzero boundaries,
but does not provide free products, target outputs or certificate coverage.
No actual public Transformer numerical ABI or successor-state path was replaced.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
FULL_MISSION_O1_O6=OPEN
CORE_ADMISSION=false
```

All target hardware, complete physical 8-GiB fit and same-machine latency remain
NOT TESTED. The missing piece is an algorithmic information/cost construction,
not just hardware validation. No impossibility theorem or feasibility increase
is claimed. Remote handoff needs actual post-commit SHA/file read-back.

## Evidence and continuity

[Summary](results/native_transfer/summary.json),
[inverse summary](results/native_transfer/inverse_summary.json),
[decision/assumptions](docs/research/NATIVE_TRANSFER_LEDGER.md),
[validation](VALIDATION_MATRIX.md), [next work](NEXT_EXPERIMENT.md).
20 focused tests passed under pytest and unittest; five scientific captures
regenerated identically. Full-repository and checkpoint tests were not run.
No Actions dispatched and no target host or checkpoint download was used.

The prior [root state](docs/research/history/pre_native_transfer_20260905/RESEARCH_STATE.md)
is preserved as its original blob. Its scoped 2x2 proof, unresolved 2x3 problem,
PR #118 rejection and EXP-102A scientific decision are unchanged. Separate
EXP-103A..108A branches remain separate. Historical root decision/failure/assumption
ledgers are preserved; this round's new entries are additive in the linked ledger.
