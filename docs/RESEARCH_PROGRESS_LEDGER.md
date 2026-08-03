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

## EXP-047/047R closed evidence

```text
EXP-047 workflow 30793232558
EXP-047R workflow 30795946233
EXP-047R artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

EXP-047 established E1 certificate/fallback correctness but broad synthetic savings failed. EXP-047R's exact realized range oracle evaluated 100% at median and p90 on 18 small-checkpoint states.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

## EXP-048 frozen evidence — PR #58

```text
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
workflow merge SHA d60e392d66d694fc020f2cfe2435e47e5f5a22ca
artifact 8850040445
artifact size 17689 bytes
artifact SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
phase A/B/C-observation
evidence E1
```

Pinned checkpoints:

```text
TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

MEASURED exactness and causality:

```text
9 EXP-048 tests passed
repository validation passed
3 models × 6 held-out families = 18 cases
B1 mismatches 0
B2 mismatches 0
B3 mismatches 0
B3 future information uses 0
```

B1 non-deployable perfect future proposal:

```text
96 exact tokens / one target pass
logical target-equivalent fraction 1.0416667%
projected requirement 1.185185%
future information true
```

B2 hard Jacobi:

```text
p50 58 target passes / 32 exact tokens
p50 fraction 181.25%
p90 fraction 193.75%
maximum matching prefix 3
```

B3 partial-layer self-draft:

```text
54 fixed variants
best cases with nonzero matching prefix 4/18
maximum matching prefix 1
p50 committed tokens / verification 1
minimum fully accounted fraction 1333.463%
p90 fully accounted fraction 2893.843%
```

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

The verifier is retained. Hard Jacobi, sequential partial-layer self-draft, and B4 tree continuation from failed B3 are rejected as core.

## Current frontier — EXP-049

`Anderson-Accelerated Continuous Block Fixed-Point Gate` removes the separate per-token draft loop.

```text
soft future token embeddings
-> small number of full batched target passes
-> damped Picard / bounded Anderson updates
-> hard proposal
-> retained exact block verifier
```

The experiment also tests a worst-case triangular-dependency lower bound: arbitrary causal target-only black-box rounds may be unable to guarantee more than one new exact token position per round.

Early requirements include p50 exact prefix >=16 after at most four solver passes, p90 accounted stream fraction <=10%, zero mismatch/future information/numerical failure, and at least 4x p50 prefix improvement over hard Jacobi.

## Current classification

```text
Governance/provenance implemented
Auxiliary mmap/index/DAG retained
EXP-047/047R certificate correctness E1 PASS; range core REJECTED
EXP-048 exact block verifier E1 PASS auxiliary
EXP-048 perfect proposal arithmetic PASS but non-deployable
EXP-048 hard Jacobi and partial-layer draft REJECTED as core
EXP-049 pre-registered, NOT TESTED
Complete real operation replacement NOT TESTED
70B/405B scaling NOT TESTED
8 GiB target NOT TESTED
E6/E7 not achieved
```
