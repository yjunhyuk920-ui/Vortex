# VORTEX Research Progress Ledger

Last updated: 2026-08-03 Asia/Seoul

Compatibility ledger. Current authority is `RESEARCH_STATE.md`; permanent decisions/failures are in root registers.

## Fixed target and environment

Target: arbitrary public unmodified dense Hugging Face model, runtime only, real 405B, total peak GPU VRAM <=8 GiB, original contract preserved, p50 warm time/token <=1.2x native 4B Q4.

Current Phase D: **NOT TESTED**. GitHub CPU is not target GPU/405B/CUDA/PCIe/SSD/TTFT/tokens-per-second evidence.

## Governance — PR #56

Phase A–D, E0–E7, provenance labels, root research documents, future-information audit, exact correction/fallback, and anti-overclaim rules are enforced.

## Prior milestones

- #42 exact dense-operator information lower bound; metadata is not traffic.
- #44 direct/operator top-1 metadata bound.
- #46 constructed Llama decision metadata; sparse host access remained open.
- #48 serial host probe count does not prove latency.
- #50 atomic/checksummed mmap exact pointer VM.
- #52 bounded compiler: exact finite-domain replay, raw prefix held-out start 0%.
- #54 exact suffix DAG: body compression only, causal held-out start 0%.
- #56 EXP-047 governance and CPTC E1 primitive; broad savings failed.
- #57 EXP-047R exact range oracle median/p90 100%; range CPTC core rejected.
- #58 EXP-048 exact block verifier retained; hard Jacobi and partial-layer draft rejected.

## EXP-049 frozen evidence — PR #59

```text
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
phase A/B/C-observation
evidence E1
```

MEASURED implementation safety:

```text
9 solver/lower-bound tests passed
repository validation passed
18 cases
1,458 fixed trajectories
exact selected mismatches 0
future information in S1/S2 0
unhandled numerical failures 0
S3 oracle alignment failures 0
```

MEASURED favorable performance:

```text
reference-selected best S1/S2 p50 prefix 4.5
maximum prefix 6
p90 target-equivalent fraction 168.778596%
model medians 4.5 / 5.0 / 4.0
hard Jacobi p50 after four passes 4
Anderson p50 after four passes 1
Anderson improvement 0.25x
```

MEASURED adversarial boundary:

```text
Picard prefixes by round 1,2,3,4
Anderson prefixes by round 1,2,3,3
hidden suffix transcript indistinguishability true
one-new-position-per-round barrier observed true
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

## Current frontier — EXP-050

`Target-Independent External Draft Advice Gate` imports another already-published unmodified checkpoint as the proposal information source.

Initial fixed pool:

```text
Target 1M <- Drafts 3M,8M
Target 3M <- Drafts 1M,8M
Target 8M <- Drafts 1M,3M
```

The exact verifier remains. Every sequential draft forward and target verification pass is charged. An exact-reference best-draft selector is allowed only as a non-deployable favorable upper bound.

Universal boundary: any fixed target-independent draft can be contradicted at the first token by an arbitrary target.

PROJECTED final arithmetic:

```text
4B/405B draft ratio 0.0098765432
required total fraction 0.01185185185
minimum perfect external-draft prefix after draft cost 507 tokens
```

## Current classification

```text
Governance/provenance implemented
Auxiliary mmap/index/DAG retained
CPTC certificate/fallback E1 auxiliary
Exact block verifier E1 auxiliary
Picard/Anderson reference E1 auxiliary
Range CPTC core rejected
Hard Jacobi core rejected
Partial-layer self-draft core rejected
Target-only continuous fixed-point core rejected
EXP-050 pre-registered, NOT TESTED
Real operation replacement NOT TESTED
70B/405B scaling NOT TESTED
8 GiB target NOT TESTED
E6/E7 not achieved
```
