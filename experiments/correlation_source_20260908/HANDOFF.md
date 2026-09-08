# Handoff: partial constructive correlation-source work

Parent: `66c01825635a357f86f9dfa5369088d7a239c92d`, PR #146.
Branch: `research/constructive-adaptive-word-20260908`.
Local checkout: `C:\ChatOnSteroids\Vortex` (connected Core Git and Python 3.12 work).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
HANDOFF_STATUS=IN_PROGRESS
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN

Read MISSION_AND_WORKING_PRINCIPLES.md and docs/CONSTRUCTIVE_THEORY_CONTRACT.md
again at the actual remote head. Do not resurrect stale 2026-08-03 handoff tasks
as current scientific directions. Archived prior entry-point bytes are in
docs/research/history/pre_correlation_source_20260908/.

Implemented: exact observed joint-pattern constructor, bit-packed positional
IDs, actual selected-position histogram generation, arbitrary supplied ANF
evaluation; independent file/parser/function checks; paid ordered FP32 term
witness continuation. The missing cross-pattern information is generated, but
every selected position is visited and the native continuation keeps full
ordered reductions. This is a bounded auxiliary attempt, NOT a qualifying core.

The raw words are not retained by the query object, but the ENTIRE source bytes
are resident in host RAM. The native witness assumes product terms already
exist and does not remove their production. No original model, GPU, target
server, performance benchmark or full RNG/KV substitution ran.

Start from REPORT.md and obligations.json, not test counts. Main surviving open
task is a genuinely cheaper native joint-effect producer: a finite layout,
address rule and decoder that avoid full positional-ID or weight-product scans,
retain native order/live state, and prove complete construction/query costs.
No such candidate is supplied. A named function, an available capacity bound or
a small conditional descriptor is not that constructor.

Three flows compared: (A) joint information then nonlinear consumer, (B) global
native relation elimination, (C) persistent transformed causal state. A was
implemented and then continued to ordering; B's small-scope elimination and C's
nonenumerative native conjugacy remain unconstructed. Do not implement larger
truth tables, repeat local 2x3 SAT/linear seeds, tune dictionary sizes, or scale
planted patterns as the next primary result. A new B/C step must give a real
generation rule that defeats its missing-cost premise before a backend.

Reproduce inside this directory with Python 3.12:

    py -3.12 reproduce.py --workdir replay/check1

The work directory must not exist; no evidence is overwritten. Frozen science
hashes are compared, not regenerated as a replacement baseline. See validation.json.

All result files and raw inputs are committed directly, not merely in a ZIP.
No Actions or force/main push is authorized. A later remote_receipt.json or PR
comment records actual post-commit verification; a snapshot cannot contain its
own final commit hash. Scientific completion is independent of persistence.
