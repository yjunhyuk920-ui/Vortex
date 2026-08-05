# Recent Failed and Demoted Approaches

Continuation of `FAILED_APPROACHES.md`. This is a permanent anti-repetition register. Revisit an entry only with a mechanism that directly addresses the recorded failure and a stronger preregistered falsification.

<!-- EXP-066-AUTHORITATIVE-FINAL -->
## F-034 — Exact classical TT/MPO bond-rank core

Frozen real-Q4 rank evidence gave favorable p50 operation/query fractions of 3.8941%/2.9984%, but p50/p90 static storage lower bounds were 11.0524%/22.9883%. The measured small-model extrapolation placed the 405B Q4 lower-bound representation at about 14.315 GB, above 8 GiB before runtime state.

Do not revive classical single-matrix TT/MPO by hiding factor storage, reporting only operation/query savings, or treating unresolved unit-boundary cuts as an exact reconstruction. Retain the MPO rank certifier as an auxiliary.

<!-- EXP-067-AUTHORITATIVE-FINAL -->
## F-035 — Exact joint Q/K/V row and common-right-factor reuse

Across 24 complete Q/K/V groups and 10,752 Q4 rows, exact reusable rows were zero. The common-right rank was 100% of input width at p50/p90; operation fractions remained 100%, while storage rose to 107.41%/114.12%.

Do not reopen with equality/sign/proportional-row variants, larger repeated blocks, or a joint kernel before a genuinely new shared circuit is proven. Retain the group certifier as an auxiliary.

<!-- EXP-068-AUTHORITATIVE-FINAL -->
## F-036 — Exact absolute-unread global demand certificates

Even after granting every preceding Transformer operation and weight read, the winning LM-head row, all metadata, and an independently optimal reveal order for every competitor for free, the output-head-only mandatory lower bound was 13.7697% at p50 and 19.2524% at p90. The p50 target failed before any real scheduler, propagation, fallback, or kernel cost.

Do not reopen with tile-size/order tuning or the same norm/absolute-unread bounds. Retain the bound auditor as an auxiliary.

<!-- EXP-069-AUTHORITATIVE-FINAL -->
## F-037 — Causal exact temporal-span replay

Exact dyadic modular-rank certificates covered 833 warm projection traces. Certified-independent arrivals alone required 100% of dense weight reads and operations at p50/p90; model p50 values were 69.244%/100%/100% for TinyStories-1M/3M/8M. No exact duplicate replay hit occurred, and the favorable basis cache was 391.97% of one Q4 projection-weight population at p50.

Do not reopen with approximate subspaces, numerical tolerances, post-selected longer traces, future/cross-prompt dictionaries, or uncharged coefficient/cache work. Retain the dyadic rank auditor as an auxiliary.

<!-- EXP-070-AUTHORITATIVE-FINAL -->
## F-038 — Exact Q4 short-block local-pattern table circuits

Across all 144 frozen real-Q4 dense projections and 3,024 preregistered width/order plans, the best single joint plan per matrix still required p50/p90 operation fractions of 88.4856%/91.4423%. Exact dictionaries, pattern IDs, offsets, row scales, and routing raised p50/p90 query and static representation fractions to 111.0294%/112.7907%. Even the most favorable matrix had a joint worst-axis fraction of 105.4244%.

Do not reopen by adding block widths or column orders after observation, reporting arithmetic without bytes, hiding dictionaries/IDs/routing/scales, using selected matrices, or approximately merging patterns while claiming exactness. Retain the block-pattern analyzer only as a conditional auxiliary for models with independently measured repetition.

<!-- EXP-071-AUTHORITATIVE-FINAL -->
## F-039 — Unqualified impossibility claims from online matrix-vector lower bounds

The Boolean/F2 reduction was exact in 1,052,740 exhaustive cases, but the strongest registered succinct theorem did not cover any Llama-405B tensor family under the full 8 GiB side-information allowance. For the largest valid square subproblem, 8 GiB is 1,024x above the theorem's `n^2/4` redundancy ceiling. Neither registered source supplies the required direct-sum theorem for 884 jointly preprocessed matrices, and all displayed bounds hide asymptotic constants.

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

Do not divide the 8 GiB state by tensor count, sum per-matrix asymptotic bounds, set hidden Omega constants to one and call the result certified, equate one cell probe with one GPU/PCIe/SSD transaction, or claim that all exact software executors are impossible. Retain the theorem/reduction auditor as a guardrail. This entry does not establish feasibility.
