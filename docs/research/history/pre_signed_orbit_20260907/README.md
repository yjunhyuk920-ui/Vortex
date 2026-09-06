# VORTEX

The fixed mission remains arbitrary public unmodified HF dense 405B, one total-8-GiB GPU, batch one, original output/RNG/required successor state, same-machine native 4B Q4 p50 <=1.2x, p95 <=1.5x and the existing TTFT requirement. **Not achieved.**

## Current bounded construction — 2026-09-07

[Joint moments, exact energy verification and a five-moment positive certificate](experiments/joint_energy_20260907/REPORT.md).

A finite <=2-residual constructor/query program was implemented. A follow-on nonnegative annihilator removes the dense Gram verifier, but its strictly positive dense-input success implies rank(W)<=3 in this representation. Arbitrary native rounding, general residual complexity, full successor state and the target budget remain unresolved. This is auxiliary E1, not an admitted core or a closed theory.

There are 27 exact returns among 71 registered queries and 44 unresolved/unsupported cases. All 2,368 returned coordinates match. Ordinary dense control groups have no successful returns; favorable controls were deliberately constructed. Twenty-one local tests pass and 63 manifest-listed files reproduce; no model, GPU, KV/RNG, latency or Actions test was run.

```
python experiments/joint_energy_20260907/restore.py --out /tmp/vortex-joint-energy
cd /tmp/vortex-joint-energy
python -m unittest discover -s tests -v
python src/run.py --out results/reproduced
```

The checked archive preserves all sources, tests, frozen inputs, raw observations, actual packets and full Korean report. Its compression is archival, not an inference claim.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`, `FULL_MISSION_O1_O6=OPEN`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next construction](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md), [constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).

The complete previous README is [preserved byte-for-byte](docs/research/history/pre_joint_energy_20260907/README.md). Historical relative links in that snapshot are interpreted from their original repository-root location. All earlier files, failed evidence and policies remain unchanged. Parent: verified PR #130 head `e5d7c9ecde971def1245beb1edf2e84c12443dba`. Remote persistence is separate from scientific completion.
