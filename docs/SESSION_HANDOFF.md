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
11. EXP-048 document and `results/exp_048/summary.json`
12. PR #58.

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
branch research/exp-048-causal-block-amortization
PR #58
```

## EXP-047/047R closed decision

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

The exact per-state range oracle evaluated 100% at median and p90. C3 was not continued. Certificate/fault rejection/exact fallback remain auxiliary.

## EXP-048 authoritative evidence

```text
results/exp_048/summary.json
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
workflow merge SHA d60e392d66d694fc020f2cfe2435e47e5f5a22ca
artifact 8850040445
artifact size 17689 bytes
artifact ZIP SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
```

Pinned external revisions:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

## EXP-048 MEASURED result

Exactness and causality:

```text
9 EXP-048 tests passed
repository validation passed
3 models × 6 families = 18 cases
B1 exact mismatches 0
B2 exact mismatches 0
B3 exact mismatches 0
B3 deployable future information uses 0
peak RSS 682552 KiB
```

B1 perfect future oracle:

```text
96 exact tokens / 1 target pass
target-equivalent fraction 1.0416667%
future information true
deployable false
```

B2 hard Jacobi:

```text
p50 target passes / 32 exact tokens 58
p50 accepted tokens per target pass 0.551724
p50 fraction 181.25%
p90 fraction 193.75%
maximum matching prefix 3
```

B3 partial-layer self-draft:

```text
18 cases, 54 fixed variants
best cases with nonzero matching prefix 4/18
maximum matching prefix 1
p50 committed tokens per target verification 1
model medians 1 / 1 / 1
minimum fully accounted fraction 1333.463%
p90 fully accounted fraction 2893.843%
```

PROJECTED:

```text
405B Q4 full stream 188.592821 GiB
1.2x 4B allowance 2.235174 GiB/token
required target-equivalent fraction 1.185185%
zero-cost perfect-proposal minimum 85 tokens/full target pass
B1 oracle fraction / requirement 0.87890625
B3 p90 fraction / requirement 2441.6793
```

## Scientific decision

```text
EXP-048 exact block verifier: ACCEPT E1 AUXILIARY
B1 perfect future proposal: NON-DEPLOYABLE UPPER BOUND ONLY
B2 hard Jacobi core: REJECT
B3 partial-layer self-draft core: REJECT
B4 tree continuation from failed B3: DO NOT IMPLEMENT
complete real operation replacement: NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```

Required phrase:

> The exact block verifier safely preserved greedy output, and a future-aware 96-token oracle reached a logical 1.0417% stream fraction. The causal partial-layer draft matched at most one proposal token and had p50 one committed token with p90 28.9384 target-equivalent streams per token, so partial-layer self-drafting is rejected as the core runtime.

## Frozen evidence layout

```text
results/exp_048/summary.json
results/exp_048/raw/artifact_provenance.json
results/exp_048/raw/checkpoint_manifest.json
results/exp_048/raw/cases.jsonl.gz.b64
results/exp_048/processed/aggregate.json
results/exp_048/logs/run.log
results/exp_048/artifacts/
results/exp_048/checksums.sha256
```

Restore raw cases:

```bash
base64 -d results/exp_048/raw/cases.jsonl.gz.b64 | gunzip > /tmp/exp_048_cases.jsonl
sha256sum /tmp/exp_048_cases.jsonl
# b70d56f3e13ab1f39dd8947be468e663d6b5691fb20236b990f20a343bcbe4d2
```

## Next work — EXP-049

`Anderson-Accelerated Continuous Block Fixed-Point Gate` removes the sequential per-token draft loop.

```text
soft future-token embeddings
-> 1/2/4 full causal target block passes
-> damped Picard or bounded Anderson update
-> hard proposal
-> retained exact block verifier
```

Conditions:

```text
S0 charged hard Jacobi baseline
S1 damped continuous Picard
S2 Anderson history m in {2,4,8}
S3 exact future-state oracle, non-deployable
S4 adversarial triangular causal models
K in {64,128,256}
solver passes in {1,2,4}
```

Early rejection:

```text
any exact mismatch/future information/unhandled numerical failure
p50 matching prefix <16 after <=4 solver passes
p90 accounted fraction >10%
Anderson p50 improvement <4x over hard Jacobi
materially worsening model-size trend
universal >1-position/round claim contradicted by adversarial model
```

Theoretical obligation:

> For arbitrary causal dense models, a target-only synchronous black-box solver may be unable to guarantee more than one new exact token position per target round in the worst case.

EXP-049 must formalize and test this before making any universal acceleration claim.

## Reproduction

```bash
git checkout research/exp-048-causal-block-amortization
python -m pytest -q tests/exp_048
python scripts/run_validation.py
bash experiments/exp_048/reproduce.sh
cd results/exp_048 && sha256sum -c checksums.sha256
```

Do not overwrite `results/exp_047/`, `results/exp_047r/`, or `results/exp_048/`. EXP-049 must use a new branch and result directory.
