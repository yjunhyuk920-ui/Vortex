# VORTEX

The fixed mission remains arbitrary public unmodified HF dense 405B, one total-8-GiB GPU, original output/RNG/required successor state, same-machine native 4B Q4 p50 <=1.2x, p95 <=1.5x and existing TTFT. **Not achieved.**

## Current audit — 2026-09-05

[Theory-closure attempt, native-coding proofs and source assessment](docs/research/THEORY_CLOSURE_NATIVE_CODING.md).
No complete theory or new core was obtained. Recent error-correcting reductions still require a fast weak solver; their algebraic guarantees are not native rounding guarantees. A scoped continuation-separation theorem does not prove a VORTEX lower bound or produce a cheap source.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`.

```bash
python experiments/theory_closure/verify.py --output results/theory_closure
```

[State](RESEARCH_STATE.md), [next obligation](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [raw observations](results/theory_closure/observations.json).

Read [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md) and [constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md). No policies/runtime were changed. No Actions or hardware/checkpoint inference was run.
[Previous README, preserved unchanged](docs/research/history/pre_theory_closure_20260905/README.md). Parent is verified PR #124 commit ab08dbb0f6665e6207c4a1d14a0d0677146f4542; local-only radix code was read as context, not silently merged.
