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

## Subsequent full-contract construction attempt — 2026-09-05

[Full-contract attempt and explicit remaining construction](docs/research/FULL_CONTRACT_CONSTRUCTION_ATTEMPT.md). No all-conditions algorithm was obtained; no new executable or core is promoted. This is an incomplete analytical attempt, not another achieved theory, runtime experiment or hardware result. The existing scientific statuses and next construction obligation remain unchanged.


## Additional construction search — no new core

[Search outcome and source scope](docs/research/ADDITIONAL_CONSTRUCTION_SEARCH_20260905.md). No all-conditions algorithm, new qualifying core, completed theory or runtime result was obtained. This documentation-only record does not change the scientific status, fixed mission or next construction obligation.

## Whole-block transport constructions — 2026-09-06

[Report, proofs, explicit programs and paid gates](experiments/block_transport_20260906/docs/REPORT_KO.md). Exact quadratic compilation, an actually serialized dyadic-power program, and attention-query transport were constructed and checked. None supplies a qualifying >=10x native whole-model route; no new core or completed theory is promoted. The three representations are not claimed as three materially new admitted principles.

```bash
cd experiments/block_transport_20260906
python src/research.py --out results
python -m unittest discover -s tests -v
```

The declared CPU reference ran 12 unit tests and four byte-identical regeneration checks. Raw input/output words are preserved in `results/raw.jsonl.gz`; reproduction emits `raw.jsonl`. No public checkpoint, full state/RNG executor, GPU, 405B, baseline latency or TTFT was tested. [Scope and handoff](experiments/block_transport_20260906/HANDOFF.md). All previous README bytes and scientific constraints are preserved.
