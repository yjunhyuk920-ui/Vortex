# Frozen CPU reproduction

From repository root use Python3.12/NumPy2.3.5, PYTHONUTF8=1 and
PYTHONDONTWRITEBYTECODE=1. Commands:

    python experiments/codex_fmm_integrity_20260909/run_focused_tests.py --output <fresh-focused-json>
    python experiments/codex_fmm_integrity_20260909/run_full_gate.py --output-dir results/exp_100a/<fresh-directory>
    python experiments/codex_fmm_integrity_20260909/verify_hardening.py

The full gate downloads only the pinned public catalog and verifies its blob.
It refuses a populated directory. Never overwrite historical, red or corrected
evidence. The final command verifies the saved pair, not the fresh directory.
Raw executed-source snapshots permit auditing across line-ending conventions.

portable_only_runner.py --config experiments/exp_100a/config.json
--output-dir <fresh-directory> replays the old invalid classifier.
original_runner.py is unchanged historical Git source and needs Unix resource.
Do not run old update_ledgers.py wholesale on newer frontier documents.
No Actions/GPU was started or is required for this repair.

Tracked runner line endings were normalized after execution, with exact raw
source snapshots retained. source_transport.json confirms identical Python AST;
verify_repair.py accepts only exact raw or CRLF-normalized source identity.

Final authority is hardened_run; corrected_run is the first intermediate.
The final verifier checks both saved corrections, source snapshots and all
state-derived direct/oracle counts. Current runner is the final executed LF source.
