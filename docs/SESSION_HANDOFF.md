# Session Handoff

Last updated: 2026-08-03 Asia/Seoul

## Mandatory startup

Read in order:

1. `AGENTS.md`
2. `RESEARCH_STATE.md`
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `ARCHITECTURE.md`
9. `HARDWARE_VALIDATION_PLAN.md`
10. `REPRODUCIBILITY.md`
11. EXP-047R document and `results/exp_047r/summary.json`
12. PR #57.

Root files and machine-readable result JSON are authoritative; conversation memory is not.

## Fixed target

Real arbitrary unmodified Hugging Face dense model, runtime only, 405B flagship, <=8 GiB VRAM, original contract preserved, and p50 warm time/token <=1.2x native 4B Q4 on the same machine.

## Environment truth

```text
8 GiB target GPU unavailable
405B storage/execution unavailable
CUDA/PCIe/target SSD profiling unavailable
real 405B TTFT/tokens/sec/VRAM NOT TESTED
Phase D NOT TESTED
```

## Current branch and PR

```text
repository yjunhyuk920-ui/Vortex
branch research/exp-047r-oracle-stratified-audit
PR #57
```

## EXP-047R authoritative evidence

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
workflow merge SHA 213e69a54c4d2b5c2d4102f8651cab847ade312f
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

Pinned external revisions:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

## EXP-047R MEASURED result

```text
9 EXP-047R tests passed
repository validation passed
3 models
18 held-out current-token states
wrong accepts 0
bound violations 0
future generated tokens false
real operation replacement false
C1 exact-state oracle median 100%
C1 exact-state oracle p90 100%
C2 median 100%
C2 p90 100%
C2 best 254/256 = 99.21875%
C2 materialized CPU primitive/full-sum median 2165.057x
peak RSS 649808 KiB
```

The CPU ratio is not a GPU or intrinsic lower-bound result. The C1 coverage failure alone decides the Gate.

PROJECTED:

```text
405B Q4 full stream 188.592821 GiB
1.2x 4B allowance 2.235174 GiB/token
required target-equivalent fraction 1.185185%
C1 oracle median / required 84.375x
```

## Scientific decision

```text
EXP-047/047R correctness primitive: ACCEPT E1 in scope
range-based CPTC core: REJECT
certificate/fault rejection/exact fallback: AUXILIARY
C3 variance rescue: DO NOT IMPLEMENT
real operation replacement: NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```

Required phrase:

> The exact per-state range oracle evaluated 100% at median and p90 on 18 states from three pinned small checkpoints. Therefore range-based CPTC is rejected as a primary runtime; only its E1 certificate/fallback primitive is retained.

## Frozen evidence layout

```text
results/exp_047r/summary.json
results/exp_047r/raw/artifact_provenance.json
results/exp_047r/raw/checkpoint_manifest.json
results/exp_047r/raw/cases_part_01.jsonl
results/exp_047r/raw/cases_part_02.jsonl
results/exp_047r/raw/cases_part_03.jsonl
results/exp_047r/processed/aggregate.json
results/exp_047r/logs/run.log
results/exp_047r/checksums.sha256
```

`.github/workflows/exp_047r_gate.yml` is manual-only and writes reproduction output to `results/exp_047r_reproduction`.

## Next work — EXP-048

`Causal Block Verification Amortization Gate` changes mechanism class.

Deployable hypothesis:

- use an early-layer prefix of the same unmodified checkpoint as a training-free causal draft;
- propose a token block;
- execute one exact full-target teacher-forced causal verification pass;
- commit only the longest matching prefix plus exact correction;
- charge every draft layer stream, target pass, rejected position, KV rebuild, and correction.

Controls:

```text
B0 exact sequential greedy
B1 future-token perfect-proposal oracle, upper bound only
B2 existing Jacobi baseline with every pass charged
B3 causal partial-layer self-draft
B4 tree only if B3 survives
```

Traffic arithmetic:

```text
required target-equivalent fraction 0.01185185
zero-cost perfect-proposal minimum 85 accepted tokens/full target pass
real draft cost raises required acceptance
```

Early rejection:

```text
any exact mismatch
any future information in deployable path
p50 accepted tokens/verification <16
p90 target-equivalent fraction >10%
accounted B3 cost >= exact sequential B0
materially worsening size/depth trend
```

Promotion still requires p50 accepted block >=85 and p90 target-equivalent fraction <=1.185185% with zero mismatches/future information.

## Reproduction

```bash
git checkout research/exp-047r-oracle-stratified-audit
python -m pytest -q tests/exp_047r
python scripts/run_validation.py
bash experiments/exp_047r/reproduce.sh
cd results/exp_047r && sha256sum -c checksums.sha256
```

Do not overwrite `results/exp_047/` or `results/exp_047r/`. EXP-048 must use a new branch and new result directory.
