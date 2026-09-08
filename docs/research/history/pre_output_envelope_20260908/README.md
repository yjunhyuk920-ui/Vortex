# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state; same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** No training/weight/mission change; every preparation, storage, movement, arithmetic and state cost counts.

## Current bounded record — 2026-09-08
[Native row-frontier producer](experiments/native_row_frontier_20260908/REPORT.md) automatically generates a lossless row-edit itinerary and preserves the original declared BF16/FP32 balanced reduction tree. No previous-query similarity, input-response bank or error-retry. Unstructured matrices are handled correctly but nearly all work remains.

18 synthetic matrices/360queries/80640output coordinates match separate C/NumPy,378Fraction checks. At512: intentionally coherent controls have1.48-1.72%source and1.26-1.36%FPwork;1%mantissa perturbations retain5.95-6.02%source and8.14-8.21%FPwork. Random controls have~200%source and~99.86%FPwork. Source ratios are not total memory traffic.

A post-hoc current-CPU batch1 diagnostic against handwritten fixed-order C measured~20x/~7.7x for the two favorable families and1.59-1.62x slowdown for random data. Compilation/loading excluded; this is NOT optimized BLAS/HF/GPU/native4BQ4 or full-model performance. Preparation uses full W and O(m^2 N) comparisons; first8complete costs not closed.

14tests/147deterministic sciencefiles replay to a fixed manifest. Source/C/tests/prereg/CPUsummary/verification are in a checked capsule; raw science regenerates. Korean detailed report, all initial CPU samples and available logs are in user ZIP. FullHF/KV/RNG/CUDA/405B/8GiB/4BQ4/TTFT not tested.

THEORY_STATUS=NOT_ESTABLISHED; CORE_ADMISSION=false; TARGET_HARDWARE_STATUS=NOT_TESTED; FULL_MISSION_O1_O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false; README_CURRENT=true.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_native_row_frontier_20260908/README.md).
