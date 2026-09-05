# VORTEX

Proof-first research for arbitrary public, unmodified Hugging Face dense models.
The fixed mission remains **405B / one total-8-GiB GPU / original output, RNG
and successor-state contract / same-machine native-4B-Q4 p50 <=1.2x,
p95 <=1.5x and the existing TTFT requirement**. It is not achieved.

## Current frontier — 2026-09-05, latest-source/native-observation audit

[Research, source links, three candidate constructions, proofs and costs](docs/research/LATEST_RESEARCH_NATIVE_OBSERVATIONS.md).
Eleven primary-source records include 2026 lossless codecs, low-bit engines,
recent speculative work, and native arithmetic models; latest included submission
is 2 September. Paper measurements are not local reproductions.

A causal weight-prefix source now constructs an exact stored BF16 result for a
declared normal eight-product native-style block, without consulting unknown
mantissa bits. Its **9-bit-per-weight header floor and paid refinement reject
it as a 10x core**. The new contribution is a scoped source/cut certificate,
not a fast full Transformer. Exact real products can still give different
native-model stored results; dead private register bits need not be preserved
when all actual live-out/state/RNG boundaries are preserved.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

256 model blocks: zero mismatches; 254 early certificates but 90.9332% mean
logical weight-bit reads. 65,536 exhaustive stored-bound controls and 4,096
rounding controls passed. These are CPU reference checks, not LLM or GPU tests.
12 focused unit tests passed. No checkpoint/GPU/8-GiB/4B latency measured.

## Reproduce this scoped reference

```bash
python -m pip install -r experiments/native_observable/requirements.txt
python experiments/native_observable/audit.py
python -m unittest discover -s tests/native_observable -v
```

[Exact recorded summary and lossless case capture](results/native_observable/summary.json),
[local validation](results/native_observable/validation.json),
[current state](RESEARCH_STATE.md), [next research gate](NEXT_EXPERIMENT.md),
[validation matrix](VALIDATION_MATRIX.md).

## Governance and continuity

Read [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md),
[constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md),
[research efficiency](docs/RESEARCH_EFFICIENCY_CONTRACT.md),
[work protocol](docs/WORK_SESSION_PROTOCOL.md) and
[commit/handoff mandate](docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md).
Policies are unchanged. GitHub Actions is not a required duplicate laboratory.

This branch starts from the real PR #122 head, not an unpushed reconstruction.
[Previous README](docs/research/history/pre_frontier_20260905/README.md) and
three prior root ledgers are preserved by their original Git blobs.
Earlier scientific outcomes remain valid in their stated scope.
The later local-only causal-source ZIP is context, not silently added to the
remote lineage. Its provenance is recorded separately. No unrelated PR is merged.
Remote recording status is established by the post-commit branch/tree receipt,
not by a self-referential claim inside this snapshot.
