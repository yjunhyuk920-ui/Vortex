# VORTEX Research Progress Ledger

Last updated: 2026-08-03 Asia/Seoul

Compatibility ledger. Current authority is `RESEARCH_STATE.md`; permanent failures and decisions are in root registers.

## Fixed target and environment

Target: arbitrary public unmodified dense Hugging Face model, runtime only, real 405B, <=8 GiB VRAM, original contract preserved, and 4B-class user experience.

Current Phase D: **NOT TESTED**. GitHub CPU is not target GPU/405B/CUDA/PCIe/SSD/TTFT/tokens-per-second evidence.

## Governance — PR #56

Phase A–D, E0–E7, provenance labels, root research documents, future-information audit, and exact/fallback requirements are enforced.

## Prior milestones

- #42 exact dense-operator information lower bound; metadata is not traffic.
- #44 direct/operator top-1 metadata bound.
- #46 constructed end-to-end Llama decision metadata 26.1586 GiB; sparse host access open.
- #48 serial host probe count does not imply latency.
- #50 atomic/checksummed mmap exact pointer VM.
- #52 bounded TinyLlama compiler: 72/72 replay, 64/64 distinct raw prefix nodes, held-out start 0%.
- #54 exact suffix DAG: 64->38 nodes, causal held-out start 0%.

## EXP-047 frozen evidence

```text
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
525 cases
wrong accepts 0
certified 4
fallback 99.238%
N=1024 mean evaluated 98.294%
positive control 10.449%
```

Decision: E1 certificate/fallback primitive accepted; global-range CPTC-v1 not promoted.

## EXP-047R frozen evidence — PR #57

```text
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
phase A/B/C-observation
evidence E1
```

Pinned checkpoints:

```text
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

MEASURED correctness:

```text
9 EXP-047R tests passed
repository validation passed
18 held-out states
wrong accepts 0
bound violations 0
future generated tokens false
```

MEASURED coverage:

```text
C1 exact-state oracle median 100%
C1 p90 100%
C2 median 100%
C2 p90 100%
C2 best 254/256 = 99.21875%
```

PROJECTED gap:

```text
required target-equivalent fraction 1.185185%
C1 oracle median / required 84.375x
```

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

C3 is not continued as a rescue. Certificate/fallback code remains auxiliary.

## Current frontier — EXP-048

`Causal Block Verification Amortization Gate` changes the mechanism class from scalar weight skipping to multi-token full-stream amortization.

Reference arithmetic:

```text
one full target verification stream / 85 accepted tokens
= 1.17647% before any draft cost
```

The deployable path will use a training-free partial-layer self-draft from the same unmodified checkpoint and exact longest-prefix target verification. Future-token perfect proposals are non-deployable upper-bound controls only. Existing Jacobi work is a charged baseline.

## Current classification

```text
Governance/provenance implemented
Auxiliary mmap/index/DAG retained
EXP-047/047R correctness E1 PASS in scope
Range-based CPTC core REJECTED
EXP-048 pre-registered, NOT TESTED
Real operation replacement NOT TESTED
70B/405B scaling NOT TESTED
8 GiB target NOT TESTED
E6/E7 not achieved
```
