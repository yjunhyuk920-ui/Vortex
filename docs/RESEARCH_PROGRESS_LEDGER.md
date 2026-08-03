# VORTEX Research Progress Ledger

Last updated: 2026-08-03 Asia/Seoul

Compatibility ledger. Current authority is `RESEARCH_STATE.md`; permanent failures and decisions are in root registers.

## Fixed target and environment

Target: arbitrary public unmodified dense Hugging Face model, runtime only, real 405B, <=8 GiB VRAM, original contract preserved, and 4B-class user experience.

Current Phase D: **NOT TESTED**. GitHub CPU is not target GPU/405B/CUDA/PCIe/SSD/TTFT/tokens-per-second evidence.

## Governance — PR #56

Implemented Phase A–D, E0–E7, provenance labels, nine root research documents, direct unseen-prompt operation-skipping filter, future-information audit, and exact/specified fallback requirements.

Existing mmap/index/DAG work is auxiliary. Raw prefix scaling is rejected.

## Prior milestones

- #42 exact dense-operator information lower bound; metadata is not traffic.
- #44 direct/operator top-1 metadata bound.
- #46 constructed end-to-end Llama decision metadata 26.1586 GiB; sparse host access open.
- #48 serial host probe count does not imply latency.
- #50 atomic/checksummed mmap exact pointer VM.
- #52 bounded TinyLlama compiler: 72/72 replay, 64/64 distinct raw prefix nodes, held-out start 0%.
- #54 exact suffix DAG: 64->38 nodes, causal held-out start 0%.

Detailed numbers remain in Git history and root registers.

## EXP-047 frozen evidence

Authoritative files:

```text
results/exp_047/summary.json
results/exp_047/raw/cases.jsonl
results/exp_047/checksums.sha256
```

Current frozen summary:

```text
PR #56
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
phase A/B
evidence E1
Phase D NOT TESTED
```

Mechanism: causal sample-without-replacement tile contributions, alpha-spending Serfling interval, exact full-tile fallback.

### MEASURED correctness

```text
10 tests passed
525 cases
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial fallback 15/15
future generated tokens false
```

Decision: E1 reference primitive accepted.

### MEASURED performance

```text
certified 4/525
fallback 99.238%
N=512 mean evaluated 98.519%
N=1024 mean evaluated 98.294%
positive control 10.449%
Python optimized/reference about 8.6–9.1x
```

Decision: global-range CPTC-v1 not promoted; architecture REVISE.

### PROJECTED target gap

```text
405B Q4 stream 188.593 GiB
1.2x 4B allowance 2.235 GiB/token
required fraction before overhead 1.185%
positive-control fraction 8.817x above target
```

Not measured on target hardware.

## Corrected infrastructure failures

- `30791055142`: optional dependency import;
- `30791192434`: missing `PYTHONPATH`.

Not scientific evidence.

## Current frontier

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`.

Use held-out current-token states from available unmodified small checkpoints. Compare global, non-deployable oracle-tight, and deployable stratified bounds. Offline analysis remains below E2.

If oracle-tight ranges remain high, reject range-only CPTC rather than tune it.

## Current classification

```text
Governance/provenance implemented
Auxiliary mmap/index/DAG retained
EXP-047 correctness E1 PASS
EXP-047 broad savings FAIL/REVISE
Real operation replacement NOT TESTED
70B/405B scaling NOT TESTED
8 GiB target NOT TESTED
E6/E7 not achieved
```
