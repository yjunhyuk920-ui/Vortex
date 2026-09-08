# Independent review: EXP-100A empty-search integrity classification

Reviewed source commit: `8affea0634005600df0c77f8997e0457b43a23d7`

Scope: static, read-only review of the named EXP-100A source, frozen configuration, gate document, and historical result. No experiment, test, GPU workload, network access, or result regeneration was performed.

## Verdict

`empty_direct_shape_search` is misclassified as an integrity fault when the search ran to a normal terminal condition, evaluated direct plans, and the frozen workspace screen rejected every one. That condition is a valid scoped observation: **no workspace-admissible direct plan was found for that shape/block cell in the retained frozen search**. It should make that cell unavailable to the joint planner and may contribute to a scoped scientific rejection; it should not invalidate all otherwise completed evidence.

The historical run is exactly such a case, as far as this source revision can determine:

- The only empty direct cell is `(block_length=16384, rows=128256, columns=16384)`. It reports direct count `0`, oracle count `64`, and beam counts `[96, 96, 96, 96, 0]` (`results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/raw/search_rows.json:804-819`). The nonzero depth-1 through depth-4 populations and nonzero oracle population rule out an empty catalog/orientation input for this cell.
- For every nonempty beam at or above minimum depth, `search_shape_block` evaluates an oracle plan and then evaluates direct plans for every cut and configured word width (`experiments/exp_100a/run_experiment.py:509-565`). The frozen minimum depth is `1`, static widths are `[2, 4]`, and the direct survivor cap is `160` (`experiments/exp_100a/config.json:41-55`).
- After evaluation, direct plans are passed through `prune_structural_plans` with the frozen 8 GiB limit (`experiments/exp_100a/run_experiment.py:566-572`; `experiments/exp_100a/config.json:28-36`). Inside that function, the workspace predicate is the only step that can turn a nonempty input into an empty output under the positive cap: later deduplication and frontier construction retain at least one row whenever the filtered list is nonempty (`experiments/exp_100a/run_experiment.py:440-482`).
- The beam's final zero is a normal terminal result of `expand_beam`: it returns empty only when no child survives the monotone per-axis padding bounds, after which the caller breaks (`vortex_runtime/explicit_rectangular_fmm.py:987-1010`; `experiments/exp_100a/run_experiment.py:520-524`). It is not, by itself, evidence of an interrupted or corrupt search.
- Nevertheless, `execute` currently adds `empty_direct_shape_search` for any empty cell without asking why it is empty, and any such label forces the global invalid decision (`experiments/exp_100a/run_experiment.py:863-901`). The historical result consequently says `INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE` and records this sole integrity label (`results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json:1-2,341-343`).

This conclusion is stronger than a guess but weaker than self-contained historical instrumentation. It follows deterministically from the checked-in source and frozen values: completed evaluation plus final zero survivors implies that all evaluated direct plans were removed by the workspace predicate. The historical JSON does not record the pre-filter count, rejected count, or smallest rejected workspace, so a future reader cannot prove the cause from the artifact alone.

## Minimum repair

Keep the source data, configuration, thresholds, decision strings, and historical invalid result unchanged. Repair future runs by making search completion and filtering explicit.

1. For each shape/block cell, record these cumulative counters and facts in `SearchResult` and `search_rows`:
   - `direct_plans_evaluated`
   - `direct_plans_within_workspace`
   - `direct_plans_over_workspace`
   - `minimum_evaluated_workspace_bytes` and, when present, `minimum_rejected_workspace_bytes`
   - `workspace_limit_bytes`
   - `search_completed`
   - `terminal_reason`, at minimum `padding_frontier_exhausted` or `maximum_depth_reached`

   Per-depth versions of the three plan counts are useful and cheap, but cumulative counts plus a terminal reason are the minimum needed for classification. Count plans before deduplication/frontier/cap so the accounting describes evaluation and the workspace screen rather than retention policy.

2. Enforce these invariants before assigning a scientific status:
   - `direct_plans_evaluated = direct_plans_within_workspace + direct_plans_over_workspace`.
   - A completed search with `direct_structural_plan_count == 0`, `direct_plans_evaluated > 0`, `direct_plans_within_workspace == 0`, and `direct_plans_over_workspace == direct_plans_evaluated` is `COMPLETE_NO_WORKSPACE_ADMISSIBLE_DIRECT_PLAN`, not an integrity failure.
   - A completed search with no state reaching `minimum_depth`, zero evaluated plans, and `terminal_reason == padding_frontier_exhausted` is a separate valid `COMPLETE_NO_STRUCTURALLY_ADMISSIBLE_SEQUENCE` observation. It must not be confused with an unexplained missing evaluation.
   - If `direct_plans_within_workspace > 0`, the positive retention cap requires at least one direct survivor. Zero survivors in that case is an integrity failure.
   - Nonempty states at or above `minimum_depth` require a positive evaluated-plan count. A mismatch is an integrity failure.
   - Validate `maximum_structural_plans_per_shape`, `maximum_oracle_plans_per_shape`, `maximum_family_candidates`, `per_split_survivors`, and `beam_width` as positive configuration values. The historical values are positive, but `validate_config` currently validates the frozen gate and block list without establishing these retention invariants (`experiments/exp_100a/run_experiment.py:143-185`).
   - An exception or incomplete search must prevent a completed result from being classified as a normal no-plan observation. The current writer emits the evidence only after search and decision construction (`experiments/exp_100a/run_experiment.py:1034-1040`), but an explicit completion bit makes this contract auditable rather than implicit.

3. Replace the unconditional direct-empty integrity rule at `experiments/exp_100a/run_experiment.py:871-872` with the outcome test above. Preserve an integrity label for unexplained zero populations, broken accounting, incomplete execution, or a positive eligible count followed by zero survivors. Apply the same distinction to the oracle rule at `experiments/exp_100a/run_experiment.py:873-874`: “no candidate in the defined scope” is a scientific observation; unexplained or incomplete population loss is a control failure.

4. Record why `select_joint_plan` returns `None`. It currently uses the same `None` for an empty family, an impossible budget, or dynamic-program exhaustion (`vortex_runtime/explicit_rectangular_fmm.py:894-943`), and `evaluate_blocks` serializes only `null` (`experiments/exp_100a/run_experiment.py:745-789`). A small status such as `missing_family_candidate` or `no_joint_io_admissible_plan`, with the blocking family when applicable, prevents the same ambiguity one stage later.

Under those rules, the historical pattern would be classified as a completed no-workspace-admissible direct cell. The normal decision ladder could then run. The checked-in rows show no direct or oracle 10x pass blocks (`results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json:330-340`), while the best retained direct plan has arithmetic ratio about `0.3825`, well above the `0.10` gate (`results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json:3-17`; `experiments/exp_100a/config.json:35-36`). Thus removing the false integrity failure would support the configured scientific-rejection branch for this frozen run, without rewriting the preserved historical artifact.

## What beam emptiness can support

Beam exhaustion permits a negative statement only about the retained frozen search. Products along a retained recursive path only grow, so when none of its children satisfy the padding screen, that retained frontier has no deeper admissible descendant. It does not recover paths already discarded by the per-split survivor limit or beam width.

The experiment already states the correct boundary: beam width `192` and per-split survivors `2` are frozen (`docs/research/EXPERIMENT_100A_EXPLICIT_RECTANGULAR_FMM_GATE.md:230-243`), the search is not exhaustive, and a negative result closes only the frozen catalog/language/search scope (`docs/research/EXPERIMENT_100A_EXPLICIT_RECTANGULAR_FMM_GATE.md:245-248`). The result also records `bounded_search_not_exhaustive: true` (`results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json:4250-4269`). Therefore beam emptiness cannot justify rejection of all exact bilinear algorithms, all catalog sequences under an exhaustive traversal, or native 405B feasibility.

The strongest defensible conclusion is: **none of the programs actually retained and evaluated by the frozen EXP-100A bounded search produced a complete-population joint plan passing its optimistic arithmetic, I/O, and favorable-workspace gates**. The document independently limits EXP-100A to an integer-exact structural/resource ledger and excludes TARGET-W execution, successor-state construction, finite-word equality, physical 8 GiB allocation, and native 4B/405B latency (`docs/research/EXPERIMENT_100A_EXPLICIT_RECTANGULAR_FMM_GATE.md:332-339`). No native equivalence follows from the real/integer tensor algebra.

## Unresolved concerns

- The historical artifact lacks the minimum rejected workspace and pre/post-filter accounting. Its workspace-prune diagnosis depends on matching the artifact to this exact source/config revision; it should remain historically invalid rather than being edited in place.
- The configured rejection token, `REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE`, can be read more broadly than the actual beam-limited evidence (`experiments/exp_100a/config.json:62-68`). The surrounding claim boundary is adequate for a careful reader, but programmatic consumers should always carry the bounded-scope field or an equally explicit scope status beside the token.
- `state_count_by_depth` records population size but not how many expansions were attempted or rejected by each axis. A `padding_frontier_exhausted` reason is enough for the classification repair; per-axis rejection counts would make later search-quality analysis easier but are not required to fix integrity semantics.
- Candidate truncation at the structural, family, and joint-search stages remains part of the scientific scope. Correcting the integrity label does not turn the retained beam into an exhaustive catalog search and does not strengthen any O1-O6 or native-execution claim.
