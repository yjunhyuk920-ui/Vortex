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
11. `docs/PROOF_FIRST_CONTRACT.md`
12. `docs/WORK_SESSION_PROTOCOL.md`
13. EXP-049 document, `results/exp_049/summary.json`, PR #59.

Root files and machine-readable result JSON are authoritative; conversation memory is not.

## Fixed target

Real arbitrary unmodified Hugging Face dense target, runtime only, 405B flagship, total peak GPU VRAM <=8 GiB, original contract preserved, p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same machine.

## Environment truth

```text
8 GiB target GPU unavailable
405B storage/execution unavailable
CUDA/PCIe/target SSD profiling unavailable
real 405B TTFT/tokens/sec/VRAM NOT TESTED
physical block weight reuse NOT TESTED
Phase D NOT TESTED
E6/E7 not achieved
```

## Current branch and PR

```text
repository yjunhyuk920-ui/Vortex
branch research/exp-049-anderson-continuous-fixed-point
PR #59
```

## EXP-049 authoritative evidence

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact size 105493 bytes
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

Pinned external revisions:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

## EXP-049 MEASURED result

```text
9 EXP-049 tests passed
repository validation passed
3 models × 6 families = 18 cases
1,458 fixed solver trajectories
excluded states 0
selected exact mismatch 0
selected future-information uses 0
unhandled numerical failures 0
S3 oracle alignment failures 0
peak RSS 684684 KiB
```

Favorable exact-reference-selected S1/S2:

```text
p50 matching prefix 4.5
maximum matching prefix 6
p90 target-equivalent fraction 168.778596%
model medians 4.5 / 5.0 / 4.0
all selected rows: 4 target passes, block 64
17/18 hard top-1 Picard
0/18 Anderson
```

Controls:

```text
hard Jacobi p50 prefix after four passes 4
Anderson p50 prefix after four passes 1
Anderson/Jacobi improvement 0.25x
```

Triangular audit:

```text
Picard prefixes 1,2,3,4
Anderson prefixes 1,2,3,3
hidden suffix transcript indistinguishability true
one-new-exact-position-per-round barrier true
```

## Scientific decision

```text
EXP-049 solver/numerical reference: ACCEPT E1 AUXILIARY
exact block verifier: RETAIN E1 AUXILIARY
target-only continuous fixed-point core: REJECT
405B/8 GiB/4B-class performance: NOT TESTED
```

Required phrase:

> Even with exact-reference selection of the best fixed Picard/Anderson trajectory, EXP-049 achieved only p50 4.5 exact proposal tokens and p90 1.6878 target-equivalent streams per committed token. Hidden triangular targets also preserved the one-new-position-per-round transcript barrier. Target-only continuous fixed-point proposal generation is rejected as core; 405B and target hardware remain untested.

## Frozen evidence layout

```text
results/exp_049/summary.json
results/exp_049/raw/artifact_provenance.json
results/exp_049/raw/workflow_summary.json
results/exp_049/raw/checkpoint_manifest.json
results/exp_049/raw/triangular_audit.json
results/exp_049/raw/cases.jsonl
results/exp_049/processed/aggregate.json
results/exp_049/logs/run.log
results/exp_049/artifacts/
results/exp_049/checksums.sha256
```

`.github/workflows/exp_049_gate.yml` is manual-only and writes isolated reproduction output. The one-shot evidence-freeze workflow installed byte-identical authoritative evidence and does not trigger on ordinary result/document changes.

## Next work — EXP-050

`Target-Independent External Draft Advice Gate` changes the information source.

Algorithm:

1. load exact prompt into target and another already-published unmodified draft checkpoint;
2. generate a causal draft continuation with draft KV cache;
3. verify the entire draft block with one exact target pass;
4. commit only target-matching prefix plus exact first-mismatch correction;
5. charge every draft token forward, target verification, selector, rejected position, KV state, and correction.

Fixed initial pool:

```text
Target 1M <- Drafts 3M,8M
Target 3M <- Drafts 1M,8M
Target 8M <- Drafts 1M,3M
```

Controls:

```text
E0 target-independent first-token counterexample
E1 every cross-checkpoint single draft
E2 exact-reference favorable pool selection, non-deployable
E3 exact future-target oracle, non-deployable
E4 tree forbidden unless E2 survives
```

Early rejection:

```text
any verifier mismatch
any target-future leakage
favorable pool p50 exact prefix <16
p90 4B/405B-normalized fraction >10%
any required family with zero useful proposal acceptance
worsening size trend
universal first-token counterexample succeeds
```

Universal boundary: a fixed target-independent draft can always be contradicted by an arbitrary target choosing a different first token. Practical pool evidence is separate from the arbitrary-model claim.

PROJECTED 4B draft requirement:

```text
4/405 = 0.0098765432 target streams/draft token
required total fraction 0.01185185185
perfect proposal minimum after draft cost 507 tokens
```

## Reproduction

```bash
git checkout research/exp-049-anderson-continuous-fixed-point
python -m pytest -q tests/exp_049
python scripts/run_validation.py
bash experiments/exp_049/reproduce.sh
cd results/exp_049 && sha256sum -c checksums.sha256
```

Do not overwrite frozen `results/exp_047*`, `results/exp_048`, or `results/exp_049`. EXP-050 uses a new branch/result directory.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-052 handoff

Enumerative exact advice is rejected. Read `results/exp_052/summary.json` and `NEXT_EXPERIMENT.md`; continue with EXP-053 or a materially new mechanism only.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053 handoff

Bit-exact AIG structural hashing is rejected as core. Read `results/exp_053/summary.json` and continue with EXP-054 reduced decision diagrams or a materially new mechanism only.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054 handoff

Reduced decision diagrams are rejected as core. Continue with EXP-055 word-level column-signature/popcount aggregation or a materially new mechanism only.

<!-- EXP-072A-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-072A

Branch: `research/exp-072a-information-capacity`.

Authority: `results/exp_072a/summary.json`; source `468f297925e10bdc541fe48f19c2f72a1e3f5e14`; evidence commit `f9ac26befb01fd9a71c7c6e1efed4c4b4df31389`.

Decision: reject a self-contained exact Q4 arithmetic DAG as a universal 8 GiB hot core. Exact basis outputs make the artifact an injective matrix encoding; worst-case Q4 information is `188.98828125 GiB`, `23.62353515625x` the hot allowance. Restricted synthesis remains auxiliary. Cold-backed online execution is not ruled out.

Next: EXP-073 sanitized, read-only inventory of the privately identified Ubuntu target, followed only with separate authorization by storage/transfer/native-4B baselines. Never commit connection details. Do not download 405B, install packages, restart services, or disturb workloads during inventory.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## Current handoff after EXP-074

Branch: `research/exp-074-weight-stationary-block-gate`.

Authority: `results/exp_074/summary.json`; source
`8abc06e73c884b839927cf41d5f4fa6cbb8fc051`; evidence
`c1d778af011672ec7fadfa66935ba2548de8e115`.

Decision: MTP-1 plus expert paging is rejected as a 1B-class core after
retaining `10.0x` baseline traffic under free drafting and fixed routes. The
long-block path is revised: fixed/free requires nine perfect accepted tokens,
0.8B draft requires 25, and independent-uniform expected routing requires 98.

Next: metadata-only native-MTP surface audit, then the smallest unchanged
checkpoint accepted-prefix Gate if the surface exists. Do not download 35B or
122B and do not build a page scheduler. EXP-073 Stage 2 remains not run.
