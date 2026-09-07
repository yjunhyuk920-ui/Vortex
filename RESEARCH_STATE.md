# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Current bounded construction and precision-source gate](experiments/precision_rank_20260907/REPORT.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed W=LDR modulo2^K with paid dense transformations, serialized U,V and source-only Python/C evaluation. Strict Q>2B centered lift reconstructs full signed outputs on integer BF16 coefficients/inputs with FP32-exact partial sums and declared zero-sign metadata. Not general native BF16/Transformer/state execution.

40matrices/320queries/15872coordinates match. At128 random signs/odd r1=1, but rK=128; full factor source65584/98352B versus32768B original BF16,32768 products versus16384. Original tightly encoded signs/odd values2048/16384B. Dense cubic preparation and other address/temporary/modular costs are charged; no measured speedup.

Proved rK minimum only for two-factor modular linear representation. Actual FWHT on same Sylvester controls uses896 add/sub slots despite rK=128: not a universal cost lower bound. Dense-code bit capacity and variable-W border-rank gates do not lower-bound arbitrary fixed-checkpoint algorithms.

18tests/603scientific files regenerate to b502d79e20c4f10f791737ca43581dfa5ea5f92dbc1f1c49b25253b22d8b8bb3. First Fortran/C-boundary failure and pre-guard/cost-correction evidence preserved in userZIP; final source/test/expected results are in remote capsule. No target hardware or full neural state test.
[Prior state unchanged](docs/research/history/pre_precision_rank_20260907/RESEARCH_STATE.md).
