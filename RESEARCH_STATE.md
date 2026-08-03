# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored model-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 on the same target machine;
- independently reproducible evidence.

The target is unchanged.

## Current environment

MEASURED capability: GitHub repository, GitHub Actions CPU, Python, and small downloadable checkpoints.

Unavailable and NOT TESTED: target 8 GiB GPU, 405B checkpoint execution, CUDA, PCIe, SSD, target power, TTFT, tokens/second, and target peak VRAM.

## Validation system

- Phase A theory/structure;
- Phase B synthetic/reference;
- Phase C small-real-model falsification or real-operation replacement;
- Phase D actual target hardware, currently NOT TESTED.

Evidence E0–E7 and provenance labels MEASURED/DERIVED/PROJECTED/UNVERIFIED remain mandatory.

## Component classification

Auxiliary accepted:

- exact/checksummed mmap pointer VM;
- bounded TinyLlama compiler in its finite tested grammar;
- exact future-suffix DAG as body compression only;
- CPTC causal probabilistic certificate and exact fallback as an E1 reference primitive.

Rejected as core:

- raw prefix/future routing for unseen prompts;
- static compression, deterministic residual, recurrent program, repair, and related families in `FAILED_APPROACHES.md`;
- global-range CPTC-v1;
- range-based CPTC as a primary execution path, including further C2/C3 range/variance tuning intended to rescue it.

## EXP-047 authoritative evidence

Authoritative source:

```text
results/exp_047/summary.json
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
phase A/B
evidence E1
```

MEASURED:

```text
525 cases
wrong accepts 0
fallback/reference mismatches 0
certified 4
fallback 99.238%
N=1024 mean evaluated 98.294%
positive control 10.449%
```

Decision: correctness primitive accepted; global-range savings failed.

## EXP-047R authoritative evidence

Authoritative source:

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
workflow merge SHA 213e69a54c4d2b5c2d4102f8651cab847ade312f
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
phase A/B/C-observation
evidence E1
```

Pinned checkpoints:

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

MEASURED:

```text
3 models
18 held-out current-token states
wrong accepts 0
checkpoint-derived bound violations 0
C1 exact per-state oracle median evaluated fraction 100%
C1 exact per-state oracle p90 evaluated fraction 100%
C2 stratified median 100%
C2 stratified p90 100%
C2 best case 254/256 = 99.21875%
C2 materialized CPU primitive/full-sum median ratio 2165.057x
peak RSS 649808 KiB
future generated tokens used false
real operation replacement false
```

PROJECTED:

```text
required 405B Q4 evaluated fraction before overhead 1.185185%
C1 oracle median / required fraction 84.375x
```

The CPU ratio measures the Python certificate primitive after contributions were already materialized. It is not LM-head, GPU, PCIe, SSD, or target speed.

## Scientific decision

- ACCEPT EXP-047/047R correctness evidence at E1 only.
- REJECT range-based CPTC as the core execution architecture. The strongest favorable C1 oracle still read 100% in every case, so tighter static range metadata or variance-adaptive C3 cannot repair the core gap without changing the mechanism class.
- RETAIN certificate/fallback code only as an auxiliary guard for future mechanisms that independently create a cheap, high-margin decision.
- DO NOT implement C3 as a continuation of EXP-047R.
- Real operation replacement remains NOT TESTED; Phase D remains NOT TESTED; E6/E7 are not achieved.

Required wording:

> EXP-047R, E1: exact current-token pair-margin reconstruction, sound bound validation, causal sampling, and fallback remained correct on three pinned small checkpoints. The exact per-state range oracle nevertheless evaluated 100% at median and p90, so range-based CPTC is rejected as a primary runtime. No 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/second claim was tested.

## Primary unresolved bottleneck

The next core mechanism must change the cost structure rather than improve scalar partial-sum ranges. It must either amortize one full weight stream across many accepted tokens or supply another model-independent operation replacement with a measured path to <=1.185% target-equivalent weight traffic before overhead.

## Current frontier

`EXP-048 — Causal Block Verification Amortization Gate`, defined in `NEXT_EXPERIMENT.md`.

This investigates training-free self-speculation using an unmodified target model: a cheap causal partial-layer draft proposes a block, one exact batched target pass verifies it, and only the longest exact prefix is committed. Future-token oracle proposals are allowed only as a labeled non-deployable upper bound. Existing Jacobi, suffix replay, and future-DAG results remain controls, not evidence of success.

## Reproduction

```bash
git checkout research/exp-047r-oracle-stratified-audit
python -m pytest -q tests/exp_047r
python scripts/run_validation.py
bash experiments/exp_047r/reproduce.sh
```

The frozen authoritative result is `results/exp_047r/`; reproduction writes to an isolated output and must not overwrite it.

## Next-session reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. EXP-047R document and frozen summary
9. PR #57
