# VORTEX Proof-First Contract

## Purpose

Prevent local, synthetic, replay, small-model, projected, or CPU-only evidence from being promoted into claims about the real 405B/8 GiB/4B-class target.

## Fixed target

VORTEX is complete only at E7, after a real unmodified 405B-class dense Hugging Face model runs end-to-end with:

- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored model-specific adapter;
- original declared output/quality contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 baseline on the same target machine;
- p95 <=1.5x baseline;
- pinned code/checkpoint and independent reproduction.

## Current environment boundary

GitHub Actions CPU runners cannot measure:

- real 405B execution;
- real target VRAM;
- CUDA kernels;
- PCIe traffic;
- target SSD behavior;
- target TTFT or tokens/second.

Those fields remain `NOT TESTED` and `UNVERIFIED` until Phase D.

## Validation phases

### Phase A — theory and structure

Required:

- mathematical statement;
- causal inputs;
- correctness/error contract;
- fallback proof;
- memory/traffic/compute equations;
- failure conditions and counterexamples;
- 405B symbolic resource model;
- explicit unverified assumptions.

No actual large-model performance claim is permitted.

### Phase B — synthetic/reference

Required:

- independent slow reference;
- optimized implementation;
- randomized/property tests;
- boundary/adversarial cases;
- fault injection;
- deterministic replay;
- measured scaling trend;
- raw logs and checksums.

### Phase C — small real-model falsification

Required:

- unmodified downloadable real checkpoints;
- real operation replacement, not offline hook-only observation for E2;
- disjoint build/evaluation prompts;
- held-out prompts and task families;
- future-information audit;
- forward/layer/tile call counts;
- token/logit agreement;
- fallback;
- CPU time and RAM;
- at least three model sizes before a scaling claim.

Purpose: early falsification only.

### Phase D — target hardware

Required:

- real target GPU constrained to <=8 GiB;
- actual target checkpoints including 70B/405B;
- same-machine 4B baseline;
- CUDA/PCIe/SSD/power profilers;
- raw reproducible wall-clock and quality evidence.

Current status: NOT TESTED.

## Evidence scale

- E0 idea/equation;
- E1 synthetic/reference validation;
- E2 small real-model operation replacement;
- E3 held-out generalization with measured causal coverage;
- E4 measured accessible representative-hardware improvement;
- E5 medium/large scaling validation;
- E6 target model under 8 GiB VRAM;
- E7 405B at declared 4B-class performance.

## Provenance

Every result separates:

```text
MEASURED
DERIVED
PROJECTED
UNVERIFIED
```

A projected 405B byte count is not a measured byte count. A CPU lookup is not a measured PCIe latency. A TinyLlama token match is not 405B quality evidence.

## Architecture Gate A0 — direct objective connection

Before implementation, answer:

1. What original Transformer operation is skipped/replaced?
2. How is that decided without future tokens?
3. What does the selector cost?
4. How are wrong skips detected?
5. What is the fallback?
6. Why is the declared output contract never silently violated?
7. How does the saving scale?
8. Why can the next decision be made without all weights?
9. What moves between SSD/RAM/VRAM?
10. What are the 405B minimum bytes/operations?
11. What gap remains to the 4B target?
12. What is the strongest falsification?

A proposal failing this gate is auxiliary, not core.

## Architecture Gate A1 — resource closure equations

Define:

```text
M_total = M_hot + M_kv + M_work + M_fallback
B_total/token = B_selector + B_normal + r_fallback * B_fallback
C_total/token = C_selector + C_normal + r_fallback * C_fallback
```

Required target conditions:

```text
M_total <= 8 GiB
B_total/token <= 1.2 * B_4B
C_total/token <= 1.2 * C_4B
```

When overlap is claimed:

```text
T_token >= max(B_total / effective_bandwidth,
               C_total / effective_throughput,
               serial_latency_floor)
```

Use measured hardware terms only in Phase D. Before then label them PROJECTED/UNVERIFIED.

## Architecture Gate B — correctness and falsification

Before Phase C:

- independent reference agrees with exact fallback;
- randomized and adversarial tests pass;
- malformed state and fault injection trigger fallback or rejection;
- future information is absent;
- selector and metadata costs are charged;
- success and rejection thresholds were committed before the run;
- no silent wrong accepts occur in the test corpus.

For probabilistic certification, report mathematical `delta` and empirical wrong accepts separately.

## Architecture Gate C — real operation replacement

A real-model result must replace the actual operation during generation. It must report:

- checkpoint revision;
- held-out prompts;
- output tokens/logits;
- forward/layer/tile counts;
- accepted certificates;
- fallback counts;
- CPU/RAM measurements;
- model-size trend;
- build/compile cost.

Replay of stored traces alone remains auxiliary evidence.

## Architecture Gate D — scaling ladder

Required sequence when resources permit:

1. 1B–3B;
2. 7B–8B;
3. 30B–34B;
4. 70B;
5. 405B.

Each rung must execute the same protocol and publish raw evidence. No larger rung may be inferred solely from a smaller checkpoint.

## Communication rules

Forbidden before the corresponding evidence:

- `405B runs in 8 GiB` before E6;
- `405B reaches 4B speed` before E7;
- `the final solution is complete` before E7;
- `measured` for projected or unverified values;
- `exact` for a probabilistic contract without qualification;
- `generalizes` from duplicate or same-trace replay.

Allowed example:

> E1, Phase B: the optimized tile certificate matched the independent reference on the tested synthetic cases and fell back exactly on adversarial cases. Real-model coverage, 405B scaling, and target hardware performance remain unverified.

## Current classification

- Existing mmap/DAG/index work: auxiliary E1/E2 bounded components.
- Raw exact-prefix scaling: rejected.
- EXP-047 CPTC: E0, Phase A/B active.
- Phase D target validation: NOT TESTED.
