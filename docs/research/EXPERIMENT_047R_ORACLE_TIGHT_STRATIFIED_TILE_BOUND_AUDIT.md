# EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit

## Final status

```text
Scientific decision: REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
Phase: A/B/C-observation
Evidence: E1
Real operation replacement: false
Phase D: NOT TESTED
```

Authoritative evidence:

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
workflow merge SHA 213e69a54c4d2b5c2d4102f8651cab847ade312f
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

## Question tested

EXP-047 CPTC-v1 used one broad contribution range and evaluated almost the entire population. EXP-047R tested whether the failure was mainly loose metadata or an intrinsic limitation of range-only finite-population certification.

Pre-registered favorable oracle Gate:

```text
C1 exact per-state range median evaluated fraction <=10%
C1 p90 <=25%
wrong accepts = 0
```

If C1 failed, C2/C3 could not rescue the range family because C1 already knew the exact realized min/max contribution range.

## Operation audited

For final hidden state `h`, exact top and runner-up LM-head rows, bias margin `b`, and input-dimension tiles `T_i`:

```text
margin = logit_top - logit_runner
       = b + sum_i c_i
c_i    = sum_{j in T_i} (w_top,j - w_runner,j) h_j
```

The top and runner-up were identified from fully evaluated reference logits. Full contributions were materialized for offline validation. Therefore this was not a deployable selector and not E2.

No future generated token was used.

## Conditions

### C0 — global checkpoint-derived range

```text
s_j = max_o W[o,j] - min_o W[o,j]
B_i = sum_{j in T_i} |h_j| s_j
range = [-max_i B_i, +max_i B_i]
```

### C1 — exact per-state oracle range

Exact min/max of all realized `c_i`. Non-deployable strongest favorable range-only control.

### C2 — checkpoint-span stratified range

Tiles grouped by `B_i`; sampling without replacement inside each stratum; per-stratum Serfling intervals; union accounting over stratum and adaptive sample count:

```text
delta_s   = delta * 6 / (pi^2 (s+1)^2)
delta_s,n = delta_s * 6 / (pi^2 n^2)
```

### C3 — variance-adaptive range

Not implemented. The pre-registered protocol forbade C3 from influencing the decision before an independent proof. C1 failed, so C3 is no longer a valid continuation of this core family.

## Frozen checkpoints

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact file hashes are in `results/exp_047r/raw/checkpoint_manifest.json`.

## MEASURED results

```text
primitive tests: 9 passed
repository validation: passed
models: 3
held-out current-token states: 18
wrong accepts: 0
checkpoint-derived bound violations: 0
future generated tokens used: false
real operation replacement: false
```

Coverage:

```text
C1 exact-state oracle median: 100%
C1 exact-state oracle p90: 100%
C1 cases below full evaluation: 0/18
C2 median: 100%
C2 p90: 100%
C2 best: 254/256 = 99.21875%
```

CPU reference cost after contributions were already materialized:

```text
C2 certificate/full math.fsum median ratio: 2165.056897860546x
peak RSS: 649808 KiB
```

The CPU ratio is an implementation measurement, not an intrinsic lower bound and not LM-head/GPU/PCIe/SSD timing. The C1 coverage failure alone is sufficient for rejection.

## PROJECTED target gap

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required evaluated fraction before overhead: 1.185185%
C1 oracle median / required fraction: 84.375x
```

These are parameter-count projections, not target-hardware measurements.

## Decision

C1 missed both pre-registered thresholds maximally: 100% versus 10% median and 100% versus 25% p90. Since C1 used the exact realized range, the result rejects the claim that static bound tightening, additional strata, or variance adaptation can turn this range-only decision primitive into the core 405B execution architecture.

Permanent classification:

- keep causal sampling, fault rejection, probabilistic interval reference, and exact fallback as auxiliary safety machinery;
- reject global/oracle-tight/stratified range-based CPTC as the primary executor;
- do not implement C3 as a rescue of EXP-047R;
- do not claim real operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/second.

## Raw evidence layout

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

Concatenate case parts in numeric order to reconstruct the original 18-row JSONL stream. The exact original GitHub artifact archive is identified by its SHA-256 above.

## Reproduction

```bash
python -m pytest -q tests/exp_047r
bash experiments/exp_047r/reproduce.sh
```

Reproduction writes to an isolated directory and must not overwrite frozen evidence.
