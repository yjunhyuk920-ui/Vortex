# Current validation addendum: query-state source, 2026-09-05

| Item | Actual evidence/status |
|---|---|
| Scoped serial-FMA all-row equality | 24 synthetic queries, zero word mismatches |
| Focused controls | 5 unittest tests passed; includes 16 accumulator continuations |
| Index/source integrity | Serialized roundtrip and deterministic result regeneration |
| Universal low-cost source | NOT ESTABLISHED; ordinary/adversarial cost screen fails |
| Original HF ABI/RNG/KV continuation | NOT TESTED; accumulator controls are not LLM state |
| 405B / total 8 GiB / TTFT / native-4B p50,p95 | NOT TESTED |
| Full repository suite / GitHub Actions | NOT RUN |

[Proof scope, raw evidence and accounting](docs/research/QUERY_SOURCE_CONSTRUCTION.md).
Logical counters and encoded byte sizes are not physical traffic, resident RAM
or latency measurements. All full-mission O1-O6 stay OPEN. The prior matrix is archived unchanged, with historical counts rather than
newly rerun tests.


The fixed mission remains arbitrary unmodified public dense 405B, executor-only,
batch one, original output/RNG/required state, total GPU allocation <=8 GiB and
same-machine native-4B-Q4 warm p50 <=1.2x, p95 <=1.5x, with the existing TTFT goal.
No hidden compute or free preparation/selection/verification/fallback is allowed.
Read [AGENTS](AGENTS.md), [the canonical mission](MISSION_AND_WORKING_PRINCIPLES.md)
and [the constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).

[Previous snapshot](docs/research/history/pre_query_source_20260905/VALIDATION_MATRIX.md) is preserved by its original Git blob.
No existing runtime or governance contract is changed. Separate branches are not merged.
