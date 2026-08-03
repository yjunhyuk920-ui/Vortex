# Next Experiment

## Closed Gate — EXP-050

Authoritative evidence:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact 8852817664
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

MEASURED:

```text
3 pinned models
18 target/prompt cases
36 cross-target draft pairs
108 K=64/128/256 rows
exact mismatches 0
target future information uses 0
favorable-pool p50 exact proposal prefix 0.5
favorable-pool maximum prefix 3
favorable-pool p90 normalized fraction 163.20987654%
matching prefix zero in 72/108 rows
Korean useful acceptance false
structured JSON useful acceptance false
target median prefixes 1.0 / 0.0 / 0.5
universal first-token counterexample matching prefix 0
```

Decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested fixed pool also failed every practical early Gate except exactness/causality. Proposal trees are not continued from this pool.

## EXP-051 — Oracle Layer-Finalization and Tail-Skip Gate

### Mechanism change

EXP-051 no longer tries to predict many future tokens. It asks whether the **current exact target token** becomes final after only a small prefix of Transformer layers.

For every exact greedy generation state:

```text
exact committed prefix/current input token
        |
        v
embedding and target blocks 1...L
        |
        +--> hidden state h_d after each depth d
        |
        v
target final normalization + target LM head
        |
        v
intermediate token prediction z_d
```

Let `z_L` be the exact final target token.

Definitions:

```text
first_match_depth = min d such that z_d == z_L
suffix_stable_depth = min d such that z_j == z_L for every j >= d
```

`suffix_stable_depth` is the favorable oracle depth relevant to tail skipping. It is non-deployable because it uses all later layers to know that no later flip occurs.

### Why this differs from EXP-048 B3

EXP-048 B3 recursively used partial layers to generate future proposal tokens. A first draft error changed every subsequent input.

EXP-051:

- always uses the exact target greedy prefix;
- analyzes only the current next-token decision;
- evaluates every target layer depth;
- separates transient early matches from suffix-stable finalization;
- measures the strongest possible layer-tail skip before designing a selector or certificate.

### Conditions

#### L0 — exact full-depth baseline

Generate 64 exact greedy tokens per held-out target/prompt using the normal target KV cache. Record exact final logits/tokens and full target parameter/CPU accounting.

#### L1 — first-match oracle

At every exact token state, record the earliest block depth whose final-norm/LM-head argmax equals the final target token. This may match, flip away, and return later; it is diagnostic only.

#### L2 — suffix-stable oracle

Record the earliest depth after which every later intermediate argmax equals the final target token. This is the primary favorable non-deployable tail-skip upper bound.

#### L3 — fixed-depth corpus oracle

For each pre-registered fixed depth fraction:

```text
0%, 12.5%, 25%, 50%, 75%, 100% of target blocks
```

report exact token agreement across every target/family. This shows whether a single target-independent depth selector could work without reference knowledge.

#### L4 — exact-reference per-state depth selector

Choose `suffix_stable_depth` independently per token using the final reference. This selector is non-deployable and deliberately favorable. It supplies the early rejection metric.

#### L5 — late-decision residual-chain adversary

Construct a finite residual network whose intermediate logits select token `a` after every layer except the final layer, where a residual update flips the exact token to `b`.

Required properties:

```text
first_match_depth = final depth
suffix_stable_depth = final depth
all fixed early depths fail
exact final output remains valid
```

This demonstrates that no universal target-independent early-exit depth can preserve every arbitrary target exactly.

#### L6 — sound tail certificate

Forbidden unless L2 survives the early Gate. A certificate must bound the effect of every omitted nonlinear attention/MLP residual on the final top-1 decision without executing the skipped target layers. Exact-reference suffix stability is not a certificate.

### Pinned corpus

Reuse:

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Six held-out families remain English narrative, Korean, code, mathematics, structured JSON, and identifier boundary.

Generate exactly 64 target tokens per target/prompt: 3 × 6 × 64 =1,152 exact token states unless a pinned context limit forces an explicit exclusion.

### Hidden-state alignment contract

For GPT-Neo targets:

- `outputs.hidden_states[0]` is embedding output;
- `outputs.hidden_states[d]` for `1 <= d <= L` is output after block `d` before final `ln_f`;
- apply the original target `transformer.ln_f` and tied/original `lm_head` to the final position of every depth;
- final-depth probe argmax must equal `outputs.logits[:, -1].argmax` and the exact generated token;
- abort on mismatch, missing hidden state, non-finite logit, or revision error.

No intermediate hidden state is interpreted as final without the target final norm/head.

### Traffic accounting

For each target checkpoint compile logical bytes:

```text
B_embed_row       current token and position rows only
B_blocks[1..L]    all parameters belonging to each target block
B_final_norm      target final normalization
B_lm_head         full output projection logical read
B_full            B_embed_row + sum(B_blocks) + B_final_norm + B_lm_head
```

Favorable oracle tail-skip fraction at stable depth `d`:

```text
(B_embed_row + sum_{j<=d} B_blocks[j] + B_final_norm + B_lm_head) / B_full
```

This assumes the oracle knows the correct depth and pays only one LM-head probe. Actual selector/certificate probes would add cost and can only be worse.

Also report:

- block-depth fraction `d/L`;
- LM-head share of full logical bytes;
- first-match versus suffix-stable gap;
- transient token flips;
- CPU time for full exact baseline and offline probes;
- peak RSS.

### Required measurements

MEASURED:

- exact revisions and file hashes;
- 1,152 token states or explicit exclusions;
- final-depth/logit reconstruction mismatch;
- intermediate token at every depth;
- first-match and suffix-stable depth;
- token flips after first match;
- oracle stable-depth logical byte fraction;
- fixed-depth token agreement by model/family;
- layer/head parameter bytes;
- margins at every depth;
- CPU time and RSS;
- late-decision adversarial result.

DERIVED:

- p50/p90 stable depth and traffic fraction;
- model/family trend;
- oracle savings upper bound;
- universal fixed-depth counterexample verdict;
- gap to 1.185185% target fraction.

PROJECTED:

- 405B Q4 logical bytes under observed fractions;
- gap to 4B-class target.

UNVERIFIED:

- causal deployable early-exit selector;
- sound nonlinear tail certificate;
- real skipped-layer operation replacement;
- target CUDA/PCIe/SSD/TTFT/tokens per second;
- 70B/405B finalization depths;
- 8 GiB execution.

### Pre-registered early rejection Gate

Reject layer-finalization/tail skipping as a core path if any condition holds:

```text
final-depth reconstruction mismatch >0
future generated token use >0
suffix-stable oracle median logical byte fraction >10%
suffix-stable oracle p90 logical byte fraction >25%
suffix-stable oracle median block-depth fraction >10%
any required family has median stable depth >50% of blocks
largest-model median stable fraction >1.25x smallest-model median
late-decision adversary succeeds
```

The thresholds are deliberately much looser than the final 1.185185% requirement. A universal late-decision target independently rejects fixed-depth exact early exit for the arbitrary-model mission; empirical oracle results determine whether restricted adaptive certification deserves continuation.

Failure decision:

```text
REJECT_LAYER_FINALIZATION_TAIL_SKIP_AS_UNIVERSAL_CORE
```

### Promotion Gate

Only if the oracle survives may L6 and actual Phase-C replacement be built. Promotion still requires:

```text
zero final reconstruction mismatch
zero future information
sound causal selector/certificate
real skipped target blocks during complete generation
p90 fully accounted fraction <=0.011851851851851851
nonzero useful savings in every family
non-degrading model-size trend
claim scope consistent with late-decision counterexample
```

### Strongest counterexamples

- final residual layer flips an otherwise stable token;
- early token matches, flips multiple times, and returns only at final layer;
- near-tied logits whose sign changes in late blocks;
- Korean/code/JSON states finalizing later than narrative;
- LM-head bytes dominate even depth zero;
- fixed depth works on one target but fails a larger one;
- exact-reference depth is very shallow but no sound causal certificate can know it.

### Evidence boundary

```text
Phase: A/B with small-checkpoint observation
Evidence ceiling: E1 until real target blocks are causally skipped
complete real operation replacement: false
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec: NOT TESTED
Phase D: NOT TESTED
```

### Next exact action

After PR #60 merges:

1. create `research/exp-051-layer-finalization-tail-skip`;
2. implement intermediate-depth reconstruction and late-flip adversary tests;
3. run pinned 1,152-state oracle audit;
4. freeze every depth/token/margin row and byte equation;
5. reject before selector work if L2 misses the lenient Gate;
6. implement L6 only if the oracle survives.
