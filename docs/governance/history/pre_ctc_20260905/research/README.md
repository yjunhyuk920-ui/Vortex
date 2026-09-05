# VORTEX

A proof-first research runtime for public, unmodified Hugging Face dense
Transformers. **The final 405B / single 8 GiB / 4B-class latency target has not
been achieved.** This repository contains experiments and evidence, not a
finished 405B executor.

## Fixed mission and rules

Replace only the executor: no retraining, fine-tuning, distillation, LoRA,
semantic weight changes, user-authored model adapters, or hidden remote compute.
Preserve the declared original output/RNG and exact or proven-bisimilar successor
state. Use one GPU with total peak VRAM <=8 GiB. On the SAME target machine,
warm time/token must satisfy p50 <=1.2x and p95 <=1.5x native 4B Q4.

All construction, checkpoint storage, CPU/RAM/SSD/PCIe/HBM traffic, decoding,
metadata, KV/workspace, verification, repair, fallback and synchronization count.
Every core round starts with three materially different principles and a
credible >=10x route; failed families are not reopened by parameter changes.

Read [AGENTS.md](AGENTS.md) and
[MISSION_AND_WORKING_PRINCIPLES.md](MISSION_AND_WORKING_PRINCIPLES.md) first.
The [commit mandate](docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md),
[reality-first contract](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md),
[proof-first contract](docs/PROOF_FIRST_CONTRACT.md), and
[efficiency contract](docs/RESEARCH_EFFICIENCY_CONTRACT.md) remain in force.

## Workflow

`LOCAL_RESEARCH -> LOCAL_VALIDATION_PASS -> COMMIT_PUSHED -> REMOTE_COMMIT_VERIFIED`

GitHub persists already locally validated work. Do not rerun the same research
on Actions unless explicitly requested. Missing hardware measurements stay
`NOT TESTED`. Push only a research branch, preserve evidence, and read back its
remote commit SHA. README freshness is part of the handoff.

## Current branch snapshot — 2026-09-05

This snapshot extends PR #118 / `ff70c1ebbca943161684f33e6afa1dc169fa8f4c`.
It does not merge or replace the separate EXP-103A through EXP-108A branches.

The latest auxiliary result is the
[E0 global-decoder kernel audit](docs/research/E0_GLOBAL_DECODER_KERNEL_AUDIT.md):

- A proof establishes the exact five-bit minimum for all 2x2 binary rank-one
  parity queries with two adaptive bit reads, even with arbitrary nonlinear
  encoders. It is NOT a 405B or general word-probe lower bound.
- 18 focused tests pass. The general 2x3/seven-bit synthesis remains unresolved:
  the bounded solver returned `unknown`, not `unsat`.
- Native/parity, rounding-order, corruption and toy successor-state checks prevent
  a synthetic success from being mislabeled as a native Transformer executor.

Decision: `SCOPED_2X2_MINIMUM_PROVED_NO_EXECUTOR_PROMOTED` (ceiling E1).
The existing PR #118 decision remains `NO_ROUTE_PROOF_REACH_CORE_PROMOTED`.
EXP-102A remains the latest numbered completed real-model Gate on this lineage;
its frozen causal external-draft mechanism was rejected. No result here changes
that decision or establishes complete target execution, physical 8-GiB fit,
CUDA/PCIe performance, or same-machine 4B latency.

Current [state](RESEARCH_STATE.md), [next gate](NEXT_EXPERIMENT.md),
[validation](VALIDATION_MATRIX.md), and
[new decision/handoff](docs/research/E0_GLOBAL_DECODER_KERNEL_AUDIT.md#7-decision-assumptions-and-next-handoff).

## Reproduce this auxiliary result

```bash
python experiments/e0_global_decoder/audit.py --output results/e0_global_decoder/summary.json
python -m pytest -q tests/e0_global_decoder/test_audit.py
python -m unittest discover -s tests/e0_global_decoder -v
python experiments/e0_global_decoder/synthesis.py
sha256sum -c results/e0_global_decoder/checksums.sha256
```

The audit uses the Python standard library. Optional bounded synthesis needs
local libz3; the recorded search used 4.13.3.0. The default synthesis command only
prints the reproducible formula hash. Solver timeout is not a scientific result.

## Repository map and broader setup

`vortex_runtime/` contains existing runtime prototypes; `experiments/` runners;
`tests/` controls; `results/` evidence; `docs/research/` research and handoffs.
The root decision/failure/assumption/architecture/hardware/reproducibility ledgers
remain available and their older evidence is not erased.

For the wider repository, use a virtual environment, install the project with
`python -m pip install -e .` and its experiment-specific pinned dependencies,
then run appropriate tests. This round ran the NEW focused suite only, not the
full repository suite or a public-checkpoint forward.

Previous root snapshots are preserved byte-for-byte under
`docs/research/history/pre_global_decoder_20260905/`. Historical status headings
must not override committed newer evidence.

```text
README_CURRENT=true
README_UPDATED=current auxiliary result; precise open frontier; reproduction; history links
```
