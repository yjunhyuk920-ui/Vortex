# EXP-050 — Target-Independent External Draft Advice Gate

## Status

```text
Implementation branch: research/exp-050-external-draft-advice
Gate registration: COMMITTED BEFORE REAL-CHECKPOINT RUN
Scientific result: PENDING
Phase: A/B with small-checkpoint observation
Evidence ceiling: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

## Mechanism change

EXP-049 established that a small number of target-only synchronous causal rounds does not universally reveal a long exact future block. EXP-050 imports information from another already-published, unmodified causal checkpoint:

```text
prompt -> external draft cached greedy continuation
       -> one exact target teacher-forced block verification
       -> longest target-matching proposal prefix
       -> exact first-mismatch correction
```

No target training, target modification, LoRA, adapter, distillation, calibration, target future token, or reference continuation enters deployable E1/E2 rows.

## Universal boundary

For a deterministic target-independent draft whose first proposed token is `a`, an arbitrary target can choose a different greedy token `b` for the same prompt. The exact matching proposal prefix is then zero.

This counterexample is universal in the fixed arbitrary-target mission and independent of average checkpoint behavior. Practical fixed-pool results remain useful only for a restricted-family assessment.

## Pinned target/draft pool

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Eligible external drafts:

```text
Target 1M <- {3M,8M}
Target 3M <- {1M,8M}
Target 8M <- {1M,3M}
```

A target never drafts for itself.

Held-out families: English narrative, Korean, code, mathematics, structured JSON, and identifier boundary.

## Conditions

### E0 — executable first-token counterexample

A fixed deterministic draft proposes token `a`. A causal adversarial target chooses token `b != a`, then continues validly. The retained exact verifier must report matching prefix zero and commit only correction `b`.

### E1 — every cross-checkpoint draft

For every model/prompt:

1. generate an exact 256-token greedy continuation for every checkpoint using its own KV cache;
2. treat another model's continuation as the external proposal;
3. execute one exact target pass over the 256-token proposal;
4. derive K=64/128/256 verification rows from the causal prefix of that single target pass;
5. charge 256 sequential draft forwards for K=256 and K forwards for each derived K accounting row;
6. compare every committed token with the exact target greedy continuation.

One 256-token target block pass is sufficient for all shorter K rows because a causal target output at position `i` is independent of later proposal positions. The single target verification stream is charged in every alternative row; alternatives are offline comparisons, not simultaneous runtime work.

### E2 — exact-reference favorable pool selector

For each target/prompt, select the best fixed external draft and K using:

1. longest exact proposal prefix;
2. lowest 4B/405B-normalized target-equivalent fraction;
3. smaller draft parameter bytes;
4. lexical draft ID;
5. smaller K.

This selector uses the exact reference and is explicitly non-deployable. It is permitted only to create a favorable falsification upper bound.

### E3 — exact target future oracle

Use exact target future tokens as proposal and verify K=256 in one pass. Future information is true; deployable false. This validates alignment and verifier arithmetic only.

### E4 — proposal tree

Forbidden unless E2 survives the early Gate. Every draft branch and target-scored node would be charged.

## Correctness contract

- draft continuations use only prompt plus that draft's own prior tokens/KV;
- target future tokens are absent from E1/E2;
- target and draft revisions are exact pinned SHAs;
- one target block pass verifies each target/draft 256-token proposal;
- shorter rows are causal prefixes of the same pass;
- only longest matching proposal prefix plus exact correction is committed;
- committed output must equal exact target greedy reference;
- malformed tokens, self-draft, undercharged forward count, revision mismatch, empty output, or non-finite accounting aborts;
- every target/draft pair and family remains in raw evidence even when prefix zero.

## Accounting

Actual small-checkpoint logical fraction:

```text
(K * draft_parameter_bytes/target_parameter_bytes + 1) / committed_tokens
```

Final normalized projection:

```text
(K * 4/405 + 1) / committed_tokens
```

Required final fraction:

```text
<=0.011851851851851851
```

A completely correct 4B-draft proposal requires:

```text
4/405 + 1/K <=0.011851851851851851
K >=507
```

The current maximum K=256 can only test the lenient early Gate.

## Required measurements

MEASURED:

- exact revisions/file hashes;
- target/draft parameters and bytes;
- prompt and continuation hashes;
- cached-generation forward count, elapsed CPU, and final KV bytes;
- target verification elapsed CPU;
- per-target/draft/K matching prefix and committed tokens;
- exact-output mismatch;
- future-information audit;
- actual small-model fraction;
- RSS and context exclusions;
- useful acceptance by family/target/draft.

DERIVED:

- favorable-pool p50/p90 prefix and traffic;
- selected draft distribution;
- target-size trend;
- family coverage;
- universal first-token counterexample verdict;
- normalized 4B/405B fraction;
- 507-token dynamic requirement.

PROJECTED:

- target/draft Q4 byte streams;
- 405B traffic under measured exact-prefix distributions.

UNVERIFIED:

- deployable target-independent draft selector;
- complete multi-cycle generation replacement;
- combined target/draft/KV fit in 8 GiB;
- physical overlap/residency;
- 70B/405B exact-prefix behavior;
- CUDA/PCIe/SSD/TTFT/tokens per second.

## Pre-registered early rejection Gate

Reject the fixed external-draft pool as core if any condition holds:

```text
exact verifier mismatch >0
target future information in E1/E2 >0
favorable-pool p50 exact proposal prefix <16
favorable-pool p90 normalized fraction >0.10
any required family has zero matching proposal tokens in every target case
largest-target median prefix <75% of smallest-target median prefix
universal deterministic first-token counterexample succeeds
```

The universal counterexample alone rejects fixed target-independent drafting as a solution for the arbitrary-model exact mission. Empirical pool results can only justify restricted-family auxiliary research.

Failure decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

## Promotion Gate

A restricted-family continuation would still require:

```text
zero exact mismatch
zero target future leakage
causal fixed selector
p50 exact proposal prefix >=507 for 4B-normalized draft cost
p90 normalized fraction <=0.011851851851851851
useful acceptance in every family
non-degrading target-size trend
combined hot-state plan <=8 GiB before Phase D
explicit claim restriction excluding arbitrary adversarial targets
```

## Claim boundary

```text
405B execution: NOT TESTED
8 GiB VRAM: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens per second: NOT TESTED
complete real operation replacement: false
Phase D: NOT TESTED
```

## Commands

```bash
python -m pytest -q tests/exp_050
bash experiments/exp_050/run_current_env.sh
bash experiments/exp_050/reproduce.sh
```
