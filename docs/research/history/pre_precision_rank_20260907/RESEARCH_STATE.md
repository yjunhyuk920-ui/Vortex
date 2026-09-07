# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Current bounded construction, obligations and source-cost gate](experiments/selector_adjoint_20260907/REPORT.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed selector-linear scalar extraction over F2: explicit coefficient/linear-node types, finite serialization, forward coefficient sweep and reverse extraction. No derivative through rounded real arithmetic or selector-bearing Boolean-idempotence substitution. No cheap universal native scalar source was constructed.

331776 FP32 rawword->BF16 conversion plus raw-state-copy cases match independent C;512 Fraction checks. This is a primitive, not a full checkpoint/state executor.225 direct coefficient gates ->463 extraction operations.8 GF2 matrices/2048queries/122880bits match;128wide representation ~65.7x bitpacked original.12Boolean circuits/3072inputs/49152bits match. Costs count topology and extraction, not just scalar output.

Unique zero-energy solution need not be reachable by greedy one-bit descent; directional expectation is not a finite exact answer. Fixed linear-frame rank gate is not a lower bound on all decoders. Original state/raw outputs are preserved in the limited primitive only.

16tests/84scientific files reproduce from source to frozen manifest b3a1f11efec95418855c985478e9dbb58ff10b7003528e0444345389d3b6c96e. Raw binaries and detailed Korean report/logs are in the user ZIP; raw science files regenerate, not all embedded in Git. All target/model/hardware obligations remain open.
[Prior state unchanged](docs/research/history/pre_selector_adjoint_20260907/RESEARCH_STATE.md).
