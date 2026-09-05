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
EXP-066 exact TT/MPO bond rank: bounded cheap-kill next Gate, NOT TESTED
exact MPO reconstruction/runtime/kernel: prohibited before EXP-066 lower-bound survival
Phase D: NOT TESTED
E6/E7: not achieved
```
