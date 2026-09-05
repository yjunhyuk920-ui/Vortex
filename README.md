# VORTEX

## Current continuation: explicit query-state source, 2026-09-05

[Constructor, proof scope, costs and decision](docs/research/QUERY_SOURCE_CONSTRUCTION.md).
A finite checkpoint-derived column index and causal native-state partition
algorithm now produce all serial-FMA row outputs without an oracle. This is
NOT the requested universal cheap source: ordinary 64x64 cases retain 4095/4096
bucket FMAs and pay 12159/4096 total FMAs plus bitmap work; the adversary retains
all dense bucket FMAs. No core, complete theory or model capability is promoted.

New reproduction: `python -m experiments.query_source.audit` and
`python -m unittest discover -s tests/query_source -v`.
Five focused tests pass; 24 synthetic all-row queries match the serial reference.
These are not checkpoint, Transformer-state, hardware or latency measurements.
`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`.
`README_CURRENT=true`: this annotation adds the actual source and failed paid-cost
screen. The previous snapshot is archived unchanged; its 20-test count is historical,
not a new run. No existing executable module or goal was changed.


The fixed mission remains arbitrary unmodified public dense 405B, executor-only,
batch one, original output/RNG/required state, total GPU allocation <=8 GiB and
same-machine native-4B-Q4 warm p50 <=1.2x, p95 <=1.5x, with the existing TTFT goal.
No hidden compute or free preparation/selection/verification/fallback is allowed.
Read [AGENTS](AGENTS.md), [the canonical mission](MISSION_AND_WORKING_PRINCIPLES.md)
and [the constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).

[Previous snapshot](docs/research/history/pre_query_source_20260905/README.md) is preserved by its original Git blob.
No existing runtime or governance contract is changed. Separate branches are not merged.
