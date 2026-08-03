# EXP-048 — Causal Block Verification Amortization Gate

## Status

```text
Implementation branch: research/exp-048-causal-block-amortization
Scientific result: NOT YET RUN
Evidence ceiling: E1
Complete real operation replacement: false
Phase D: NOT TESTED
```

## Mechanism change

EXP-047R rejected scalar range-based partial-sum certification. EXP-048 instead tests whether one exact target weight stream can verify many proposed future positions at once.

Deployable B3 proposal tokens are generated causally from the current exact prefix using only early layers, final normalization, and the LM head of the same unmodified checkpoint. No training, adapter, calibration, or future target token is used.

## Exact block verification contract

Given exact prefix `p`, proposal `q_0...q_{K-1}`, and one exact causal target pass over `p + q`, align target predictions:

```text
t_i = argmax target_logits[prefix_length - 1 + i]
```

Find the first mismatch `m`.

- if every `q_i == t_i`, commit all `K` proposal tokens;
- otherwise commit `q_0...q_{m-1}` and exact correction `t_m`;
- discard all later proposal state and target predictions.

The correction is exact because every token before position `m` matched the exact greedy path. Predictions after the first mismatch are never committed.

## Conditions

### B0 — exact sequential baseline

Generate 96 exact greedy tokens using the target KV cache. Charge one logical full target stream per token.

### B1 — perfect future-token oracle

Use the exact 96-token B0 continuation as the proposal and verify it with one full causal target pass.

```text
logical target-equivalent fraction = 1 / 96 = 1.041667%
```

B1 uses future generated tokens, is non-deployable, and may prove only verifier exactness and the best possible amortization upper bound.

### B2 — Jacobi control

Initialize a 32-token block with a fixed fill token, repeatedly execute exact target block passes, and commit only an inductively stable prefix. Charge every iteration and failed position. If no stable prefix appears within four iterations, commit only the exact first target prediction from the final pass.

### B3 — causal partial-layer self-draft

For each held-out prompt, sequentially draft 32 tokens using the same checkpoint's first 1, 2, and 4 layers where available:

```text
current exact prefix + prior draft tokens
    -> embeddings
    -> first N target layers
    -> target final normalization
    -> target LM head
    -> next draft token
```

Then run one exact full-target block verification and commit only the safe prefix plus correction.

## Accounting

For every B3 variant charge:

- one full target verification stream;
- one early-layer parameter stream per sequential draft step;
- one full LM-head and final-norm stream per draft step;
- gathered token/position embedding row elements for every reference draft forward;
- all 32 proposed/scored positions;
- rejected positions and correction behavior;
- CPU elapsed time separately from logical stream accounting.

The logical target-equivalent stream fraction is:

```text
(1 target stream
 + draft layer-equivalent streams
 + draft LM-head/norm/embedding-equivalent streams)
/ exact committed tokens
```

Attention/KV arithmetic is not relabeled as weight traffic. It remains visible in CPU elapsed time and UNVERIFIED hardware cost.

## Pinned checkpoints

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

The six held-out families are English narrative, Korean, code, mathematics, structured JSON, and brittle identifier continuation.

## Exactness and causal requirements

- B0/B1/B2/B3 outputs must agree with the exact greedy reference wherever committed;
- B3 proposal generation may read only the exact current prefix and its own prior draft tokens;
- B1 future information must be recorded separately and cannot enter deployable aggregates;
- first-mismatch correction and all malformed contracts fail closed;
- every exact target pass and draft step is counted;
- all checkpoint and tokenizer revisions and files are hashed;
- raw per-case rows, aggregate, environment, logs, and checksums are retained.

## Pre-registered early rejection Gate

Reject partial-layer self-draft as the core path if any condition holds on the best fixed pre-registered B3 variant per case:

```text
exact output mismatch >0
future information in B3 >0
p50 exact committed tokens per target verification <16
p90 target-equivalent stream fraction >10%
p90 target-equivalent stream fraction >= sequential B0 fraction 100%
relative median acceptance drop from smallest to largest model >25%
```

Variant selection is limited to the committed 1/2/4-layer set and minimizes fully accounted target-equivalent fraction. This is a fixed sweep, not post-hoc unrestricted tuning.

Failure decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

## Promotion Gate

Passing the early Gate only permits a complete multi-cycle Phase-C implementation. Final promotion still requires:

```text
zero exact mismatches
zero future information
p50 committed tokens per target verification >=85
p90 target-equivalent stream fraction <=1.185185%
nonzero success across all held-out families
non-degrading model-size trend
```

## Projection boundary

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
zero-cost perfect-proposal minimum: 85 accepted tokens/full target stream
```

These are PROJECTED parameter-count values. This experiment cannot measure physical target-weight reuse, accelerator kernels, 8 GiB VRAM, 405B, CUDA, PCIe, SSD, TTFT, or tokens/second.

## Strongest counterexamples

- Korean and English tokenization shift;
- code indentation and identifiers;
- arithmetic dependency;
- structured JSON boundary;
- immediate early-layer/final-layer disagreement;
- a mismatch at the first proposal position;
- LM-head-dominated small-checkpoint draft cost;
- worsening acceptance with depth/model size.

## Commands

```bash
python -m pytest -q tests/exp_048
bash experiments/exp_048/run_current_env.sh
bash experiments/exp_048/reproduce.sh
```
