# VORTEX Proof-First Contract

## Purpose

Prevent local, synthetic, replay, small-model, projected, CPU-only, future-aware, or oracle-selected evidence from being promoted into claims about the real 405B/8 GiB/4B-class target.

## Fixed target

VORTEX is complete only at E7 after a real unmodified 405B-class dense Hugging Face model runs end-to-end with:

- peak total GPU VRAM <=8 GiB;
- no target retraining, distillation, fine-tuning, LoRA, or user-authored target-specific adapter;
- original declared output/quality contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 on the same machine;
- p95 <=1.5x baseline;
- pinned code/checkpoint and independent reproduction.

## Current environment boundary

GitHub Actions CPU cannot measure:

- real 405B execution;
- real target VRAM;
- CUDA kernels;
- PCIe traffic;
- target SSD behavior;
- target TTFT or tokens/second;
- physical multi-position weight reuse;
- total target/draft hot-state fit.

Those fields remain `NOT TESTED` and `UNVERIFIED` until Phase D.

## Validation phases

### Phase A — theory and structure

Required:

- mathematical statement and claim scope;
- causal inputs and future-information audit;
- correctness/error and fallback contract;
- memory/traffic/compute equations;
- failure conditions and strongest counterexamples;
- 405B symbolic resource model;
- explicit assumptions.

No large-model performance claim is permitted.

### Phase B — synthetic/reference

Required:

- independent slow reference;
- optimized or alternate implementation where meaningful;
- randomized/property tests;
- adversarial/boundary cases;
- fault injection;
- deterministic replay;
- raw logs and checksums.

### Phase C — small real-model falsification

Required for E2 operation-replacement claims:

- unmodified pinned real checkpoints;
- actual operation replacement during generation, not offline observation only;
- disjoint build/evaluation prompts;
- held-out task families;
- future-information and selector audit;
- exact target/draft/solver call counts;
- token/logit agreement and fallback;
- CPU/RAM measurements;
- at least three target sizes before scaling claims.

Offline hook analysis, exact-reference variant selection, or a one-block observation remains E1.

### Phase D — target hardware

Required:

- target GPU constrained to <=8 GiB total usable VRAM;
- actual target checkpoints including 70B/405B;
- same-machine native 4B Q4 baseline;
- CUDA/PCIe/SSD/power profilers;
- raw reproducible wall-clock and quality evidence.

Current status: `NOT TESTED`.

## Evidence scale

```text
E0 idea/equation
E1 synthetic/reference or offline small-checkpoint falsification
E2 small real-model operation replacement
E3 held-out generalization with measured causal coverage
E4 measured representative-hardware improvement
E5 medium/large scaling validation
E6 target model under <=8 GiB total VRAM
E7 real dense 405B at declared 4B-class performance
```

## Provenance

Every result separates:

```text
MEASURED
DERIVED
PROJECTED
UNVERIFIED
```

A projected 405B byte count is not measured traffic. CPU lookup time is not PCIe latency. Tiny-checkpoint token agreement is not 405B quality. A reference-selected proposal source is not a deployable selector.

## Architecture Gate A-1 — candidate efficiency and target-scale upside

Before Architecture Gate A0 and before an experiment branch is opened, apply `docs/RESEARCH_EFFICIENCY_CONTRACT.md`.

The proposal must preregister:

```text
optimistic fully charged operation fraction
optimistic fully charged traffic fraction
optimistic storage fraction
ultimate route toward the approximately 1.185% target-equivalent fraction
new mechanism or new evidence versus closed families
reason the effect survives or strengthens with model scale
cheapest decisive falsification
implementation stage authorized if the cheap Gate survives
```

A candidate is rejected or classified as auxiliary before implementation when its favorable ceiling is only incremental, when it is a nearby variant of a closed family without new evidence, or when its gains disappear after selector, metadata, intermediate, verification, correction, fallback, RAM, SSD, PCIe, or VRAM costs.

The primary research track is not a catalogue of decompositions. Mathematical testability, novelty of notation, or adjacency to the previous experiment is insufficient.

Required progression:

```text
resource/information upper bound
-> exact certificate or favorable oracle upper bound
-> pinned small-real-checkpoint measurement
-> minimal operation replacement
-> backend/kernel
-> target hardware
```

A decisive negative theorem, lower bound, or oracle ceiling terminates the candidate. No optimized backend or physical kernel may be built merely to reconfirm it.

Promotion from cheap screening requires population-level p50/p90 evidence, a route to actual Transformer operation replacement, full current-level cost accounting, and expected information gain that justifies the next implementation stage.

## Architecture Gate A0 — direct objective connection

Before implementation answer:

1. What target operation is skipped, replaced, or amortized?
2. What new information source permits this without target future tokens?
3. What does proposal/selection cost?
4. How are wrong proposals/skips detected?
5. What is exact correction/fallback?
6. Why is the output contract never silently violated?
7. How does saving scale with model and block size?
8. Why can the next decision be made without all sequential target work?
9. What moves between SSD/RAM/VRAM?
10. What are the minimum 405B bytes/operations?
11. What gap remains to the 4B target?
12. What is the strongest universal and empirical falsification?

A proposal failing this gate is auxiliary, not core.

## Architecture Gate A1 — resource closure

Define at minimum:

```text
M_total = M_target_hot + M_draft_hot + M_kv + M_work + M_metadata + M_fallback
B_total/token = B_proposal + B_verify + B_selector + B_correction + B_fallback
C_total/token = C_proposal + C_verify + C_selector + C_correction + C_fallback
```

Required target conditions:

```text
M_total <=8 GiB
B_total/token <=1.2 * B_4B
C_total/token <=1.2 * C_4B
```

When overlap is claimed:

```text
T_token >= max(B_total/effective_bandwidth,
               C_total/effective_throughput,
               serial_latency_floor)
```

Hardware terms may be labeled MEASURED only in Phase D.

## Architecture Gate B — correctness and falsification

Before Phase C:

- independent reference agrees with exact correction/fallback;
- randomized/adversarial/fault tests pass;
- malformed or non-finite state fails closed;
- deployable future information is absent;
- every target/draft/solver/selector cost is charged;
- exact-reference oracle selection is labeled;
- success/rejection thresholds were committed before the run;
- no silent wrong accepts occur in the corpus.

For probabilistic contracts report mathematical delta and empirical wrong accepts separately.

## Architecture Gate C — real operation replacement

A real-model result must replace the actual operation during complete generation and report:

- target/draft revisions;
- held-out prompts;
- exact output tokens/logits;
- target/draft/solver pass counts;
- proposal prefix and rejection distribution;
- selector and fallback;
- CPU/RAM and build costs;
- target-size trend.

Replay, one-block observation, or reference-selected best variants remain auxiliary E1 evidence.

## Architecture Gate D — scaling ladder

Required sequence when resources permit:

```text
1B–3B
7B–8B
30B–34B
70B
405B
```

Each rung executes the same protocol and publishes raw evidence. Larger-rung success may not be inferred solely from smaller checkpoints.

## Universal-claim rule

The fixed objective says arbitrary unmodified dense target. A valid adversarial target within the declared interface can reject a universal mechanism even when average checkpoints show limited improvement.

Restricted-family success must state the restriction and may not be promoted into the arbitrary-model claim.

## Communication rules

Forbidden before matching evidence:

- `405B runs in 8 GiB` before E6;
- `405B reaches 4B speed` before E7;
- `final solution complete` before E7;
- `measured` for projections;
- `exact` for probabilistic contracts without qualification;
- `generalizes` from replay, oracle selection, or three tiny models;
- `deployable` for future-aware or reference-selected conditions.

Research-efficiency reporting must also state whether a result improves feasibility, merely closes a family, or only preserves auxiliary machinery. A long sequence of low-upside negative tests must not be described as increasing target feasibility.

## Current classification

```text
Governance/provenance system: implemented
research-efficiency candidate Gate: implemented by this contract and docs/RESEARCH_EFFICIENCY_CONTRACT.md
mmap/DAG/index and exact verifier/certifier machinery: bounded auxiliary components
raw prefix, enumerative advice, AIG/BDD, fixed-point, external draft, layer-tail, exact repetition, exact sparsity, low-rank, displacement, output-row, and Kronecker core families: rejected under their committed scopes
EXP-065 exact Kronecker rank: rejected as core; tensor certifier retained auxiliary
EXP-066 through EXP-070: rejected as core under frozen scopes
EXP-071 registered lower bounds: insufficient for a model-wide impossibility claim
EXP-072A self-contained exact Q4 hot artifact: rejected as universal core
restricted arithmetic-DAG synthesis: auxiliary only
EXP-073 sanitized target calibration: next prerequisite, NOT TESTED
Phase D: NOT TESTED
E6/E7: not achieved
```

<!-- EXP-072A-CURRENT-CLASSIFICATION -->
## Current classification after EXP-072A

EXP-066 through EXP-070 are rejected as core under their frozen scopes. EXP-071 prohibits an unqualified impossibility claim from the registered lower-bound papers. EXP-072A rejects a self-contained exact Q4 artifact as a universal 8 GiB hot core because the worst-case artifact must retain `188.98828125 GiB` of coefficient information before overhead.

Cold-backed online execution remains logically open but has no surviving constructive mechanism. EXP-073 is authorized only to calibrate a sanitized target-machine resource envelope; calibration cannot promote a runtime or change Phase D from `NOT TESTED` without the declared measurements.

<!-- EXP-074-CURRENT-CLASSIFICATION -->
## Current classification after EXP-074

MTP-1 plus expert paging is rejected as a 1B-class core for the registered
Qwen3.5 surrogate under an optimistic E1 reference Gate. A longer
weight-stationary block remains revised, not promoted, pending causal accepted
prefixes and real router locality. No checkpoint, server command, or hardware
runtime was used. The Qwen-specific MoE result cannot validate the arbitrary
dense 405B mission, and Phase D/E6/E7 remain not achieved.

<!-- EXP-075-CURRENT-CLASSIFICATION -->
## Current classification after EXP-075

The pinned official Qwen3.5-0.8B checkpoint statically exposes one native MTP
layer and 15 registered MTP tensors, and pinned vLLM source exposes a matching
loader/runtime surface. This passes only the metadata prerequisite. No weight,
proposal, acceptance, rollback, quantized path, or runtime execution was tested.

EXP-076 may perform a pinned causal small-checkpoint accepted-prefix
falsification. It must not be called E2 unless it actually replaces the
operation during complete generation with exact fallback. Qwen-specific success
cannot promote the arbitrary dense target. Phase D/E6/E7 remain not achieved.

<!-- EXP-076-CURRENT-CLASSIFICATION -->
## Current classification after EXP-076

The unchanged pinned Qwen3.5-0.8B native-MTP reference passed every registered
causality, exact-acceptance, cache, and rollback control, but failed the
accepted-prefix Gate. At build-selected `K=4`, held-out p05/p50/p95 was
`0/4/4`, below required p05/p50 minima `9/11`, with two zero-accept cases.

The long-block native-MTP surrogate is rejected under this scope. No 35B-A3B
route trace or physical scheduler is authorized. This E1 observation neither
proves vLLM runtime equivalence nor validates arbitrary dense 405B execution.
Phase D/E2-E7 remain not achieved, and there is no surviving core candidate.

<!-- EXP-077A-CURRENT-CLASSIFICATION -->
## Current classification after EXP-077A

The unchanged target replay passed all 192 registered causal decisions, but the
free activation-informed oracle preserved only `71.5278%` held-out top-1 at a
realized `9.988839%` MLP fraction. Mean/p95 KL was
`0.884161/3.080752`; all family Gates failed. Even 20% reached only
`83.3333%` top-1.

This is a valid E1 favorable-ceiling rejection of the registered Fractal MLP
score, not E2 replacement and not a universal dynamic-sparsity impossibility
result. No deployable selector or physical performance was measured. There is
no surviving core candidate; Phase D/E2-E7 remain not achieved.

<!-- EXP-078A-PREREGISTERED -->
## EXP-078A proof-first boundary

The frozen tangent-macroblock candidate composes the complete anchor-conditioned
MLP contribution and therefore does not rescue EXP-077A's channel score. Direct
construction, exact anchor, hot application, quality lifetime, and right
censoring are registered before the run. The seven-position E1 trace can reject
an early failure but cannot demonstrate the thousands-to-hundreds-of-thousands
of positions required by the charged cost equation.

No physical macro matrix, sentinel, repair, cache rollback, larger checkpoint,
or target-server command is authorized. E2-E7 remain not achieved.

<!-- EXP-078A-AUTHORITATIVE-FINAL -->
## Current classification after EXP-078A

The frozen complete anchor-conditioned MLP map failed at the first reuse token
for every held-out case. Zero of 126 later-token top-1 decisions matched, versus
the required 99%; mean/p95 KL was `14.898423/25.335417`. The baseline and
contract controls passed.

This is an E1 rejection of unchanged prior-token operator reuse, not an
impossibility proof for every causal correction scheme. No construction kernel,
sentinel, repair, larger model, target hardware, or E2-E7 evidence exists.

<!-- EXP-079A-PREREGISTERED -->
## EXP-079A proof-first boundary

The DCT block-zonotope candidate derives a small pilot image automatically and
keeps every omitted down-projection residual as a cold exact dependency with a
local correlated L2 enclosure. The p50/p95 final fractions, dual favorable
oracles, local-radius threshold, quality thresholds, target-shape equations,
fallback semantics, and stop rule are committed before model execution.

The E1 runner may reject the concrete local enclosure but cannot establish a
complete token certificate. A pass authorizes only nonlinear propagation and a
fully charged selector/fallback Gate. No Q4 fidelity, physical I/O, CUDA, peak
VRAM, wall-clock, 122B/405B execution, target-server action, or E2-E7 claim is
authorized.

<!-- EXP-079A-AUTHORITATIVE-FINAL -->
## Current classification after EXP-079A

The unchanged control and final logical byte/operation budgets passed. The
dual-oracle DCT block-zonotope candidate nevertheless achieved only `6/144`
held-out p50 top-1, mean/p95 KL `7.544862/12.616226`, and minimum local
sound-radius p50/p95 `48.663918x/57.778748x`.

This is an E1 rejection of the registered fixed pilot and row-block correlated
enclosure. It is not a universal proof against cold-backed online executors.
Nonlinear proof propagation, deployable selection, fallback, physical Q4/CUDA,
8 GiB peak, wall-clock, 122B/405B, target hardware, and E2-E7 remain absent.

<!-- EXP-080A-AUTHORITATIVE-FINAL -->
## Current classification after EXP-080A

The synthetic/reference exact-arithmetic control passed `80/80`. Under a free
perfect-future oracle, standard recursive Strassen passed logical traffic and
the incomplete favorable workspace equation but consumed `36.111580%` of dense
arithmetic at the best registered block. This exceeds the final p50 allowance
by `30.469145x`.

This is an E1 rejection of one constructive arithmetic engine, not an
impossibility proof for all fast rectangular multiplication. The exponent-only
oracle is not proof of an implementation, and no causal future-block source was
tested. Exact BF16/Q4 semantics, physical kernel behavior, peak VRAM, latency,
122B/405B execution, target hardware, and E2-E7 remain absent.

<!-- EXP-081A-PREREGISTERED -->
## EXP-081A proof-first boundary

The only exact fast-path claim is finite-field syndrome recovery followed by an
independent fingerprint. A fingerprint collision is union-accounted; a failed
or unavailable verification mandates original multiplication or abort. A low
floating residual, favorable SVD correction, or successful synthetic code does
not authorize exact commit.

The first real Gate observes only six pinned small-model projections and cannot
claim model-wide replacement. BF16/Q4 equivalence, downstream token quality,
physical traffic, peak VRAM, latency, 122B/405B, and E2-E7 remain absent.

<!-- EXP-081A-AUTHORITATIVE-FINAL -->
## Current classification after EXP-081A

The algebraic reference passed all 321 exact/fault controls, but the necessary
real residual-code premise failed. Held-out coverage was `8.681672%` versus
`99.75%`, and favorable corrected relative-L2 p50/p95 was
`0.351269/1.213828`. With observed fallback, derived traffic is
`92.244986%` of dense rather than `1.185185%`.

This is an E1 rejection of the frozen nonlinear lookup plus rank-eight exact
residual code. It is not a universal online-MatVec lower bound. No integer
operation replacement, downstream token-quality run, BF16/Q4 proof, physical
kernel, target hardware, 122B/405B execution, or E2-E7 evidence exists.
Structurally valid conditions were established. Large-model performance
remains unverified.

<!-- E0-CAUSAL-RESIDUAL-ATLAS-SOURCE -->
## Current classification after the causal information-source search

Exact committed-prefix `(x, W x)` pairs instantiate a concrete Coded Causal
Cold Source: a Causal Residual Atlas evaluates the exact-real prefix subspace
image and retains all out-of-span work as a bounded, progressively revealed
cold residual or exact fallback. Ten focused reference tests validate the
single-projection algebra, safe bounds, faults, and exact completion.

The favorable registered rank-16 screen has `1.085025716%` 64-token amortized
traffic and `0.557958575%` operations, but only by requiring
`99.899840530%` certificate coverage. No real causal trace, end-to-end bound,
native numerical contract, operation replacement, physical I/O, 8 GiB peak,
or 122B/405B result exists. The class is authorized only for preregistration
of its cheapest pinned real-weight Gate; it is not a Surviving Candidate or a
positive milestone.

<!-- CAUSAL-RESIDUAL-ATLAS-CHEAPEST-GATE -->
## Preregistered first real-weight Atlas boundary

The cheapest Gate is frozen before assigning an experiment number. It uses 18
held-out prompts and one first post-prefill decode call, with prompt-only
top-16 bases and layer-11 `q_proj`/`down_proj`. All 16/56 contiguous pages are
enumerated under an illegal native-anchored exact center; every other model
operation remains dense and free.

The exact-reference oracle prefers unchanged top-1 then minimum KL. Promotion
requires 18/18 token states, 3/3 in every family, 36/36 projection branches,
and mean/p95 KL `<=0.02/0.05`. The derived `99.899840530%` frontier allows no
failure in this finite population. This is a necessary favorable ceiling only:
no selector, legal pair center, outward-rounded certificate, simultaneous
operation replacement, physical performance, or scale evidence is present.

Failure rejects the frozen rank-16/page-64 per-projection path and prohibits
rank/page/layer/prompt/tolerance rescue. Passing authorizes only a separately
preregistered legal-center and bound-propagation Gate. E2-E7 and Phase D remain
not achieved.

<!-- EXP-083A-AUTHORITATIVE-FINAL -->
## Current classification after EXP-083A

The frozen favorable one-page oracle passed and reproduced on the unchanged
pinned Qwen3.5-0.8B checkpoint: 18/18 tokens, 3/3 per family, 36/36 branches,
mean/p95 KL `0.007225545/0.037660753`, and zero invalid controls. Independent
verification rebuilt the identical deterministic core
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`
for both complete model runs.

This is E1 evidence that a favorable page exists for every registered branch,
not evidence that a causal executor can find or certify it. Native dense
centers, exhaustive target-logit selection, and dense remaining work were
granted. The obvious post-hoc minimum-radius selector fails two tokens. Legal
pair construction, outward bounds, simultaneous replacement, physical
performance/VRAM, 122B/405B, and E2-E7 remain unverified. Only the next legal
causal-pair and bound Gate is promoted.
