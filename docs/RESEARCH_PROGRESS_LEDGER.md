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
## 2026-08-07 — EXP-072A information-capacity closure

The expensive nonlocal arithmetic-DAG synthesis plan was placed behind a cheaper information Gate. Two finite domains exhaustively produced 272 distinct exact basis signatures for 272 matrices with zero collision/control failure. The same injectivity implies `188.98828125 GiB` of worst-case Q4 information at the registered 405B count, versus 8 GiB hot state (`23.62353515625x` gap) before overhead.

Decision: `REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE_RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY`.

This is a useful family closure, not increased runtime feasibility. A cold-backed online executor remains formally open and unsupported. The next highest-value prerequisite is EXP-073 target-machine calibration so future E0 Gates use measured physical budgets rather than proxies.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## 2026-08-07 — EXP-074 weight-stationary block Gate

The proposed Qwen3.5-122B-A10B surrogate was screened without checkpoint
weights. Seven controls passed over 108 scenarios and 18 minimum searches.
MTP-1 plus paging retained `10.0x` the 1B p50 baseline-equivalent traffic and
was rejected. The longer block path was revised rather than promoted: its ideal
fixed-route/free-draft minimum is nine tokens, while route diversity and draft
cost raise the requirement sharply.

Decision:

```text
REVISE_MTP1_AND_EXPERT_PAGING_INSUFFICIENT_REQUIRE_LONG_CAUSAL_PROPOSAL_AND_ROUTING_LOCALITY_GATES
```

No model download or server run occurred. This narrows the surrogate mechanism
but does not increase dense-405B feasibility.

<!-- EXP-075-AUTHORITATIVE-FINAL -->
## 2026-08-07 — EXP-075 native-MTP surface Gate

Six pinned official metadata/source files totaling 255,779 bytes were audited.
The Qwen3.5-0.8B config declares one MTP layer, its index contains all 15
registered `mtp.*` tensors, and pinned vLLM source exposes the registered
mapping, loader, class registry, and recursive-step interface. Five controls
passed with zero failure.

Decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

This removes a metadata blocker but supplies no acceptance or performance
evidence. No checkpoint payload, inference, or target-server command occurred.
The next bounded falsification is the 0.8B causal accepted-prefix distribution;
dense-405B feasibility is unchanged.

<!-- EXP-076-AUTHORITATIVE-FINAL -->
## 2026-08-07 -- EXP-076 native-MTP accepted-prefix Gate

The pinned unchanged Qwen3.5-0.8B BF16 checkpoint was executed through a causal
CPU native-MTP reference. Six build prompts selected `K=4`; 18 held-out prompts
gave accepted-prefix p05/p50/p95 `0/4/4`, with two zero-accept cases. Required
p05/p50 minima were `9/11`. Zero future-target reads, wrong accepts, cache
mutations, commit-replay mismatches, or rollback-recompute mismatches occurred.

Decision:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

This is useful falsification and family closure, not increased target
feasibility. No target-server, GPU, quantized, 35B/122B, router, or scheduler
work ran. The primary portfolio has no surviving core candidate and must return
to E0 for a materially different query-time information source; Phase D/E2-E7
remain not achieved.

<!-- EXP-077A-AUTHORITATIVE-FINAL -->
## 2026-08-07 -- EXP-077A activation-informed Fractal MLP Gate

The pinned unchanged Qwen3.5-0.8B BF16 target replayed 24 frozen
proposal-conditioned causal-cache cases with zero mismatch over 192 positions.
A free non-deployable oracle retained the top activation/down-norm channels in
every MLP. At `9.988839%`, held-out top-1 was `71.5278%`, mean KL `0.884161`,
and p95 KL `3.080752`; all family Gates failed. The 20% arm reached only
`83.3333%` top-1.

Decision:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

This closes one dynamic fracturing score under favorable conditions; it does
not improve final feasibility and does not prove all dynamic sparsity
impossible. No new model, target-server, GPU, or physical performance work ran.
No core candidate survives; Phase D/E2-E7 remain not achieved.

<!-- EXP-078A-AUTHORITATIVE-FINAL -->
## 2026-08-07 -- EXP-078A frozen tangent-macroblock lifetime Gate

The pinned unchanged Qwen3.5-0.8B baseline matched 192/192 registered target
decisions. The candidate captured each layer's complete post-SiLU gate
coefficient at the exact last prompt token and reused the corresponding full
linear MLP law for seven causal positions.

Held-out top-1 agreement was `0/126`; every case failed on the first reuse
token; mean/p95 KL was `14.898423/25.335417`; MLP relative-L2 p50/p95 was
`0.354193/0.511419`. Direct construction would have required thousands to more
than one hundred thousand reuse positions, so the first-token quality failure is
decisive.

Decision:

```text
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
```

This closes unchanged prior-token operator reuse, not every possible causal
delta correction. It is useful falsification, not increased feasibility. No
server, larger checkpoint, physical matrix, GPU kernel, or Phase D/E2-E7 work
ran; no core candidate survives.

<!-- E0-SPARSE-FUNCTIONAL-DICTIONARY-FRONTIER -->
## 2026-08-12 -- E0 sparse functional dictionary frontier

The cold representation was generalized to arbitrary non-systematic binary
linear forms. Literal all-query storage and direct local rank-one tables fail
finite address/storage Gates; a favorable 1% table is `480.3654 TiB`.
Cardinality counting ceases to reject a global dictionary at 2,020,682 bit
forms or 494,026 favorable 64-bit words, but supplies neither rank-one
alignment nor a decoder. No model or hardware action occurred.

Decision:

```text
KEEP_ALIGNED_SPARSE_FUNCTIONAL_DICTIONARY_UNCONSTRUCTED
NO_SURVIVING_CANDIDATE
```

Authority: `docs/research/E0_SPARSE_FUNCTIONAL_DICTIONARY_FRONTIER.md`.
