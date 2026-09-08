# Actual validation

## Final authority after independent-review hardening

Canonical result: [hardened_run/result.json](hardened_run/result.json),
SHA256 375a4f4a9d5055429664f0c042c02f2d128fbf2cd8196b1fcdc6ffa87980a121; verification: hardening_verification.json.
The first corrected_run and its review remain historical intermediate evidence.

Independent review found that mutually consistent undercounts could pass the
first classifier. Four actual red mutations are preserved in post_review_red.log.
The final classifier independently derives expected evaluation counts from
retained states, cuts and frozen word widths, and checks exact completed depths
and termination. Raw oracle evaluations are counted before retention.
All six new unittest tests and eleven old functions passed. The FULL gate was
executed again: 84,712 direct and 15,608 oracle evaluations exactly reconcile
across all 50 cells; 132 integer controls have zero mismatches. All old scientific
fields and catalog/control/selected raw bytes match red and first corrected runs.
Cost formulas, selected plans and the bounded negative do not change.

Final source hash:68d8276323cf96f86ef22de661644d318f30dc9da37bd5e2a55bfa8a180fc56d.
hardened_sources preserves all manifest-bound execution bytes. The earlier
source_transport.json refers only to the first corrected intermediate source.
Current full execution used the actual final LF runner. No native/GPU claim.


Windows original import failed with missing resource. Portable-only full gate
reproduced empty_direct_shape_search. A focused budget-empty test was actually
red before repair. Source/result of the full red run are saved; standalone
import/focused-red transcripts are not saved.

run_focused_tests.py actually invoked11 existing plain functions (explicit
temporary directory for tmp_path) and5 unittest regressions; all16 passed.
This is not pytest or a repository-wide test suite.

run_full_gate.py --output-dir experiments/codex_fmm_integrity_20260909/corrected_run
actually downloaded/hash-checked the pinned catalog, ran132 integer controls,
50 shape cells and ten block selections: exit0, no integrity failure.
verify_repair.py passed all saved run checksums, source bindings, exact current
red/green old scientific fields, three raw table byte comparisons and the
independent six-component integer workspace sum.

Python3.12.14/NumPy2.3.5/Windows CPU. RSS unavailable/null.
No native HF,torch,GPU,405B or latency execution in this turn.

Tracked runner line endings were normalized after execution, with exact raw
source snapshots retained. source_transport.json confirms identical Python AST;
verify_repair.py accepts only exact raw or CRLF-normalized source identity.
