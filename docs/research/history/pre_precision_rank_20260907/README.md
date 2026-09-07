# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** Every preparation, storage, movement, arithmetic and state cost counts. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Selector-linear extraction and source-cost gate](experiments/selector_adjoint_20260907/REPORT.md): constructed a typed F2 scalar program and reverse coefficient extraction preserving declared output/state bits. A small scalar value is not a cheap whole-native producer. Wrapping the original circuit retains its work; arbitrary GF2 transpose topology is larger than the same original bitpacked matrix. This does not reopen F-040 as a universal hot core.

331776 FP32-word conversion inputs match independent C BF16 RNE output and preserved raw32bit state;512 Fraction spot checks match. This is ONE conversion primitive, not a neural layer or fullKV/RNG test. Direct Boolean source225gates becomes463 logical bit operations;128x128 GF2 program134584–134680B versus2048B original. Energy/descent and fixed-direction expectation witnesses are separately scoped.16tests/84generated scientific files replay to a frozen hash. No full HF/CUDA/405B/8GiB/4BQ4/TTFT/latency test.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`, `FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md), [contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Replay and preservation scope](experiments/selector_adjoint_20260907/README.md).
[Prior README unchanged](docs/research/history/pre_selector_adjoint_20260907/README.md). Scoped proofs/decisions/costs are in the report. Existing policies, evidence and other draft branches remain unchanged.
