# Next Experiment

## Closed Gate — EXP-049

Authoritative evidence:

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

MEASURED:

```text
3 models × 6 families = 18 cases
1,458 fixed trajectory rows
exact verifier mismatches 0
future information in S1/S2 0
unhandled numerical failures 0
oracle-best S1/S2 p50 exact prefix 4.5
oracle-best S1/S2 maximum exact prefix 6
oracle-best S1/S2 p90 target-equivalent fraction 168.778596%
S0 hard Jacobi p50 prefix after 4 passes 4
S2 Anderson p50 prefix after 4 passes 1
S2/S0 improvement 0.25x
triangular transcript indistinguishability true
one-new-exact-position-per-round barrier observed true
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

The exact reference was allowed to choose the best fixed S1/S2 trajectory per case. Even this non-deployable favorable upper bound failed the 16-token/10% early Gate. Hidden triangular models also refuted a universal faster-than-one-position-per-round target-only guarantee.

## EXP-050 — Target-Independent External Draft Advice Gate

### Mechanism change

EXP-050 imports a new causal information source rather than repeatedly querying the target:

> use another already published, unmodified small causal model to generate a long proposal with its own KV cache, then execute one exact target block pass and commit only the longest target-matching prefix plus first-mismatch correction.

No target checkpoint is modified or trained. No target-specific LoRA, adapter, distillation, calibration, or future target token is allowed. Draft checkpoints are part of the runtime and their complete sequential compute/weight-stream cost is charged.

This is different from EXP-048 B3:

- B3 reused early target layers and reread the target LM head for every proposal token;
- EXP-050 uses a separate complete draft checkpoint whose weights may remain resident independently of the target stream;
- the exact target verifier remains unchanged.

### Universal claim warning

For any deterministic target-independent draft rule `D(prompt)`, an arbitrary target model can choose a different first greedy token for the same prompt. Therefore a fixed external draft cannot guarantee even one exact proposal token for every arbitrary target.

EXP-050 must keep two claims separate:

1. **universal worst-case claim:** subject to the first-token counterexample;
2. **average practical checkpoint claim:** measured using a favorable fixed draft pool.

A valid universal counterexample independently rejects fixed target-independent drafting as a universal exact solution. Practical evidence still determines whether the component deserves auxiliary or restricted-family use.

### Conditions

#### E0 — target-independent first-token counterexample

Implement finite deterministic draft and target oracles with the same vocabulary/prompt interface:

```text
draft first token = a
target first token = b != a
```

Extend the target into a valid causal chain. Confirm exact block verification commits only target correction `b` and proposal matching prefix zero.

Repeat under randomized draft rules by conditioning the adversarial target on a seed-independent token outside the draft's support when possible, or state the probabilistic failure probability explicitly.

#### E1 — cross-checkpoint single drafts

Use the pinned unmodified TinyStories checkpoints as both targets and external drafts, never using a target checkpoint as its own draft:

```text
Target 1M <- drafts {3M, 8M}
Target 3M <- drafts {1M, 8M}
Target 8M <- drafts {1M, 3M}
```

All checkpoints share the pinned GPT-Neo tokenizer. Generate draft proposals causally with each draft model's KV cache.

#### E2 — favorable fixed draft-pool oracle

For every target/prompt/block, record every eligible draft and additionally choose the exact-reference-best draft only as a non-deployable upper bound.

Selection order:

1. longest exact target-matching proposal prefix;
2. lowest fully charged target-equivalent fraction;
3. smaller draft parameter count;
4. lexical draft model ID.

A negative result under this favorable selector rejects the fixed pool without building a selector. A positive result does not promote a runtime until a causal target-independent selector is committed.

#### E3 — same-model future oracle

Use exact target future tokens only to validate proposal alignment and exact verifier arithmetic. This condition is future-aware, non-deployable, and excluded from external-draft aggregates.

#### E4 — proposal tree

Forbidden unless E2 survives the early Gate. Every branch, draft token, target-scored node, and tree-selection operation would be charged. Tree expansion may not rescue a pool whose single paths almost always diverge at position zero.

### Pinned corpus

Reuse:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Held-out families remain English narrative, Korean, code, mathematics, structured JSON, and brittle identifier continuation.

Proposal block sizes:

```text
K in {64, 128, 256}
```

Generate one 256-token exact target reference and one 256-token continuation from every eligible external draft. Prefix metrics for shorter blocks are derived from the same causal continuation. States exceeding context limits are recorded as excluded.

### Exact correctness contract

For each proposal:

1. generate draft tokens using only the prompt and the external draft's own prior tokens/KV state;
2. execute one exact target teacher-forced block pass;
3. compare left to right;
4. commit only the matching proposal prefix plus the exact target token at first mismatch;
5. discard all later proposal/target states;
6. compare committed tokens with exact target greedy reference;
7. fail the run on any unexplained mismatch, future-target leakage, revision mismatch, malformed proposal, or non-finite accounting.

### Traffic accounting

For target parameter bytes `P_t`, draft parameter bytes `P_d`, proposal length `K`, and exact committed tokens `A`:

```text
actual_small_model_target_equivalent_fraction =
    (K * P_d/P_t + 1 exact target verification stream) / A
```

This counts one logical full draft weight stream per sequential draft token. CPU elapsed time, draft KV bytes, and target verification time are reported separately.

Final-target normalized projection for a 4B draft and 405B target:

```text
draft ratio = 4/405 = 0.0098765432 target streams/proposal token
normalized_fraction = (K * 4/405 + 1) / A
```

PROJECTED target requirement:

```text
normalized_fraction <=0.01185185185
```

With a completely correct K-token proposal, the 4B draft itself consumes `4/405` per committed token. The one target verification stream therefore requires:

```text
K >= ceil(1 / (0.01185185185 - 4/405))
  = 507 exact proposal tokens
```

This is stricter than the zero-cost proposal minimum of 85. Proposal blocks up to 256 can only pass the early Gate, not the final promotion Gate.

### Required measurements

MEASURED:

- exact model/tokenizer revisions and file hashes;
- target/draft parameter counts and bytes;
- prompt hashes and context limits;
- exact target and draft token continuations;
- per-draft matching prefix for K=64/128/256;
- exact committed tokens and verifier mismatch;
- draft sequential forward count and CPU time;
- target verification count and CPU time;
- draft KV peak estimate and process RSS;
- actual small-model target-equivalent fraction;
- future-information audit;
- model/family trend.

DERIVED:

- favorable-pool p50/p90 exact prefix;
- draft-selection oracle label;
- universal first-token counterexample verdict;
- 4B/405B normalized traffic fraction;
- dynamic exact-prefix requirement.

PROJECTED:

- 405B Q4 target stream;
- 4B Q4 draft stream;
- target-equivalent traffic under measured exact-prefix distributions.

UNVERIFIED:

- a deployable draft selector;
- physical concurrent residency/overlap;
- 8 GiB combined target/draft/KV state;
- 70B/405B exact-prefix behavior;
- CUDA/PCIe/SSD/TTFT/tokens per second.

### Pre-registered early rejection Gate

Reject the fixed external-draft pool as a core path if any condition holds:

```text
exact verifier mismatch >0
future target information in E1/E2 >0
favorable-pool p50 exact matching prefix <16
favorable-pool p90 4B/405B-normalized fraction >10%
any held-out family has zero non-correction proposal acceptance in every case
draft-pool acceptance materially worsens with target size
universal first-token counterexample succeeds
```

The universal counterexample is independently sufficient to reject the mechanism as a solution for the fixed arbitrary-model exact contract. Empirical results may still support auxiliary use in a restricted target family.

Failure decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

### Promotion Gate

A later restricted-family or revised-claim runtime may advance only if:

```text
zero exact mismatch
zero target future information
causal deployable draft selector
p50 exact proposal prefix >=507 for a 4B-normalized draft
p90 normalized fraction <=1.185185%
nonzero useful acceptance in every held-out family
non-degrading target-size trend
combined target/draft/KV hot-state plan <=8 GiB before Phase D
claim explicitly excludes the universal counterexample or supplies target-dependent advice
```

Passing restricted-family evidence may not be described as satisfying the current arbitrary-model mission.

### Strongest counterexamples

- target first token deliberately differs from every fixed draft;
- two targets sharing tokenizer and prompt but opposite greedy continuations;
- code identifiers and random-looking suffixes;
- Korean/English domain mismatch;
- EOS or JSON boundary at proposal position zero;
- larger draft that is confidently different from the smaller target;
- pool oracle chooses a different draft for every prompt, exposing selector impossibility;
- proposal prefix long on narrative but zero on at least one required family.

### Evidence boundary

```text
Phase: A/B with small-checkpoint observation
Evidence ceiling: E1 until a fixed causal selector and complete generation replacement exist
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec: NOT TESTED
Phase D: NOT TESTED
```

### Next exact action

After PR #59 merges:

1. create `research/exp-050-external-draft-advice`;
2. commit the first-token no-free-lunch reference and tests;
3. implement cached cross-checkpoint proposal generation and exact verifier integration;
4. run the pinned 18-case fixed-pool Gate;
5. freeze raw per-target/per-draft evidence and update durable state;
6. do not implement E4 trees unless E2 survives the early Gate.
