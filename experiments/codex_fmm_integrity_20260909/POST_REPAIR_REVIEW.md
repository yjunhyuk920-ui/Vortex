# EXP-100A post-repair independent static review

Review mode: current-file static inspection and read-only artifact comparison. I did not run tests, the experiment, GPU work, network access, or Git commands.

## Decision

The corrected run supports the declared **bounded negative**. The former empty direct cell is now shown to be a completed retained-beam search whose generated direct plans were all rejected by the unchanged 8 GiB workspace screen. It is not evidence corruption, and it is not a universal FMM or native-execution impossibility.

The repair has one decision-relevant integrity gap: `search_integrity_failures` checks that reported counters agree with one another, but it does not check that the reported generated count equals the number of plans required by the retained states, cuts, and word widths. A paired undercount can therefore receive a non-`INVALID` scientific decision. The current saved run is not affected: I independently derived the expected count from every saved state population and found exact agreement for all 50 cells. This gap should be repaired before treating the classifier as generally fail-closed.

## Findings

### 1. Decision-relevant gap: silent partial evaluation can pass the classifier

Severity: medium for this saved result; high for reuse of the classifier.

The search records each evaluated plan and divides it into workspace-rejected or workspace-admissible counts (`experiments/exp_100a/run_experiment.py:532-536,581-606`). The classifier requires a positive generated count, `generated == rejected + admissible`, a consistent minimum/budget relation, agreement between admissible count and direct-frontier emptiness, and at least one completed depth (`experiments/exp_100a/run_experiment.py:890-907`). It also correctly rejects a missing search collection/key and any empty oracle (`experiments/exp_100a/run_experiment.py:885-910`). Positive search caps are now validated (`experiments/exp_100a/run_experiment.py:189-194`).

Those checks do not establish complete evaluation. For example, on the actual all-rejected cell, changing both `evaluated_direct_plan_count` and `workspace_rejected_count` from `2304` to `2303` would still satisfy every condition at `experiments/exp_100a/run_experiment.py:897-905`. The function does not consult `state_count_by_depth`, the configured cut count, or the configured transformed-word widths. `retained_search_completed` is set to `True` by the producer rather than derived by the classifier (`experiments/exp_100a/run_experiment.py:632-642`). The `termination` value and the contents of `completed_evaluated_depths` are also not checked beyond truthiness.

This conflicts with the preregistered requirement that a non-invalid empty result come from a “completely evaluated retained search” (`experiments/codex_fmm_integrity_20260909/PREREGISTRATION.md:7-13`). The tests cover missing accounting, a false completion flag, an unmatched rejected-count mutation, a false minimum, a lost feasible frontier, missing cells, empty collections, and zero caps (`tests/exp_100a/test_search_integrity.py:26-58`). They do not cover a matched generated/rejected undercount, an incomplete depth list that remains nonempty, an inconsistent termination reason, or partial oracle evaluation.

Minimum repair:

- Derive an expected direct evaluation count independently from the retained populations. For depth `d`, the current frozen generator owes `state_count[d] * (1 + d * len(static_scalar_bytes))` plans: one cut-zero plan plus every configured word width for each cut `1..d`. Require the cumulative expected count to equal `evaluated_direct_plan_count`.
- Record and check `evaluated_oracle_plan_count == sum(state_count[d])` over evaluated depths. The retained oracle output may be deduplicated/truncated, so compare the raw evaluation count rather than the final count.
- Require `completed_evaluated_depths` to equal exactly the positive populated depths at or above `minimum_depth`. Require `beam_exhausted` to correspond to a final zero population, or `maximum_depth_reached` to correspond to all configured depths being processed.
- Add paired-underflow and partial-oracle mutation tests. All zero-generated and empty-oracle cases should remain `INVALID`, as intended.

This repair adds integrity checks only. It need not change the frozen config, search order, costs, pruning, thresholds, or decision ladder.

### 2. The actual empty cell is fully accounted and correctly treated as no admissible plan

The corrected cell `(K=16384, rows=128256, columns=16384)` records populations `[96,96,96,96,0]`, 64 oracle survivors, and zero direct survivors. It also records 2,304 evaluated direct plans, 2,304 workspace rejections, zero admissible plans, completed depths 1–4, and normal beam exhaustion (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:6432-6473`).

The required direct count can be derived without trusting the new counter. With two configured nonzero-cut word widths, the four populated depths require:

`96 * ((1+2*1) + (1+2*2) + (1+2*3) + (1+2*4)) = 2304`.

I applied the same derivation to all 50 saved search rows: the expected and reported totals are both `84,712`, with zero cell mismatches. This independently rules out missing direct-plan evaluations in the saved corrected run, even though the production classifier does not yet enforce that invariant.

The smallest generated workspace is `13,145,334,184` bytes against an `8,589,934,592` byte limit, so even the best generated witness exceeds the screen by `4,555,399,592` bytes (`experiments/codex_fmm_integrity_20260909/verification.json:17-42,61-75`). The empty direct family at `K=16384` is therefore reported as `MISSING_FAMILY_CANDIDATE` for `lm_head`, while the oracle remains selected (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:403-430`). That is a normal block-level no-plan result, not silent acceptance of missing work.

At `K=32`, both arms instead report `NO_PLAN_IN_FROZEN_IO_BIN_DP` with no missing family (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:80-97`). The added status distinguishes a complete candidate population that fails the frozen joint I/O selection from missing family candidates.

### 3. No search-cost or selected-plan change was found

Whole-file comparison with `original_runner.py` found only Windows portability, registered-LF transport handling, positive-cap validation, accounting/status fields, the integrity classifier, and an explicit decision-scope field. The structural workspace filter, deduplication, Pareto ordering, and cap selection are unchanged between `experiments/codex_fmm_integrity_20260909/original_runner.py:440-482` and `experiments/exp_100a/run_experiment.py:464-506`. The repaired search calls the same evaluator with the same target dimensions, cuts, word widths, and byte-width inputs; it assigns the returned plan to a local variable only so it can count it (`experiments/codex_fmm_integrity_20260909/original_runner.py:509-572`; `experiments/exp_100a/run_experiment.py:538-615`). Joint selection still happens before the new descriptive status is attached (`experiments/exp_100a/run_experiment.py:782-849`).

I independently compared the red and corrected artifacts. All 50 old search-row fields matched after excluding new accounting and timing, and all ten old block numerical summaries matched after excluding new selection-status fields. The red and corrected `catalog_rows.json`, `factorization_control_rows.json`, and `selected_plan_rows.json` are byte-identical. Their SHA-256 values agree with the recorded verification hashes (`experiments/codex_fmm_integrity_20260909/verification.json:11-14,77-78`). This is direct evidence that factorization controls and selected costs/plans did not change.

The executed-source manifest hashes the executed working runner separately and warns that the result's source-commit field names the base revision (`experiments/codex_fmm_integrity_20260909/corrected_run/executed_source_manifest.json:12-17`). I independently checked that its runner hash matches the preserved executed-source snapshot. The tracked runner now has a different raw hash because it was normalized from CRLF to LF after execution; the transport record binds both hashes and reports an identical Python AST (`experiments/codex_fmm_integrity_20260909/source_transport.json:1-5`), while the snapshot index binds the executed hash to the preserved runner (`experiments/codex_fmm_integrity_20260909/executed_sources.json:10-12`). These provenance files must remain attached to the corrected result because the result alone still records the base commit at `experiments/codex_fmm_integrity_20260909/corrected_run/result.json:6476`.

### 4. The negative is bounded and does not advance the native mission

The corrected result has no integrity failures and no direct or oracle 10x pass block (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:443-465`). The best retained direct joint row has arithmetic ratio `0.3825164968636731`, and the best retained oracle row has `0.13010262621990515`; both fail the 0.10 first gate (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:4-18,42-72`). The unchanged decision ladder therefore reaches the rejection branch only after integrity, direct-pass, and oracle-pass checks (`experiments/exp_100a/run_experiment.py:960-999`).

The decision itself carries the necessary limit: it applies only to programs retained and evaluated by the frozen catalog, orientation, beam, cut, family, and I/O-bin search, with no universal FMM or native impossibility claim (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:1-3`). The search contract also says it is not exhaustive and retains the favorable future-block, repair-selector, offline-transform, and finite-word grants (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:4391-4417`). Actual packed kernels, TARGET-W execution, physical 8 GiB allocation, successor-state construction, and same-machine native 4B performance remain unverified (`experiments/codex_fmm_integrity_20260909/corrected_run/result.json:499-529`).

Accordingly, this result does not close O1–O5 or complete O6. It does not prove exact native logits/output/RNG consumption or next-state/KV equality for arbitrary legal continuations, and it does not meet the paid full-resource, single-GPU 8 GiB, or native-4B-Q4 latency mission. The verification artifact correctly keeps `CORE_ADMISSION=false`, O1–O5 `OPEN`, O6 `PARTIAL`, theory `NOT_ESTABLISHED`, and hardware `NOT_TESTED` (`experiments/codex_fmm_integrity_20260909/verification.json:1-10,79-81`).

## Residual uncertainty

- This review did not rerun tests or the gate. Execution claims come from the saved result and verification artifacts, checked against current source hashes and static artifact comparisons.
- The independent expected-count reconciliation establishes complete direct evaluation for this saved run. There is no equivalent saved raw oracle-evaluation counter, so partial oracle evaluation is excluded here by source-loop inspection, not by artifact-level accounting.
- The rejection remains conditional on the retained beam and all optimistic EXP-100A grants. It says nothing about excluded replay/catalog/free-decoder families, arbitrary global-advice decompositions, or native hardware feasibility.
