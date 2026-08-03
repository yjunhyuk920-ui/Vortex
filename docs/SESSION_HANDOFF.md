# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or architecture-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the target is solved or proven feasible.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, branch, workflow, PR comments, and raw result JSON

Before a user-facing progress/completion answer after meaningful work, commit the ledger and this handoff. This rule is permanent.

## Current repository state

No research PR is promoted. The latest closed sequence is:

```text
PR #36  causal semantic-state signed program routing       rejected
PR #37  prompt-compiled Hankel decision program            rejected
PR #38  perfect-oracle sparse Hankel repair                rejected
PR #40  prompt-only nonlocal exact decision memory         rejected
```

Authoritative evidence heads:

```text
PR #36 research/semantic-state-program-routing
       499e5001c21d782adf79fba69ce6f2d445c0cb5e

PR #37 research/prompt-hankel-decision-program
       12f859e4ec288f0d38b29d8b71e494bdc29f6586

PR #38 research/oracle-sparse-hankel-repair
       13e3f60876199e4b06577ca51e9fd71f575cb134

PR #40 research/nonlocal-exact-decision-memory
       91b3e3f062d33087005ae38bbf94b357012f0ccd
```

Authoritative successful workflows:

```text
PR #36 Semantic state program routing gate  30778002226
PR #37 Prompt Hankel decision program gate   30778715832
PR #38 Oracle sparse Hankel repair gate      30779062125
PR #40 Nonlocal exact decision memory gate   30780847944
PR #40 corrected full CI                     30780847954
```

## Latest decisive result — PR #40

Experiment 039 stored prompt-only exact decision blocks:

```text
key_i   = normalize(P^T (h_i - mean_prompt_hidden))
block_i = prompt_token_ids[i+1 : i+1+L]
```

No continuation token or hidden state entered memory construction. Evaluation charged one exact first continuation token as the block-boundary anchor. Replay began after the anchor:

```text
query_t  = continuation_hidden_states[t]
target_t = continuation_token_ids[t+1]
```

The first workflow attempt incorrectly counted the anchor as replay. It is invalid evidence. The anchored alignment was corrected, locked by a unit test, and rerun. Only workflow `30780847944` and evidence head `91b3e3f062d33087005ae38bbf94b357012f0ccd` are authoritative.

The experiment measured:

- nearest hidden-state retrieval;
- top-4/top-16/top-64 future-token oracle among nearest entries;
- an impossible global future-token oracle that ignored hidden retrieval and searched every prompt suffix.

Corrected frontier over 256 post-anchor decisions:

| Prompt | Best rank | Nearest max | Top-64 oracle max | Global oracle max | Global first |
|---|---:|---:|---:|---:|---:|
| algorithm-runtime | 32 | 74 | 75 | 75 | 0 |
| distributed-database | 16 | 27 | 28 | 28 | 0 |
| korean-plm-governance | 16 | 4 | 5 | 5 | 4 |

Required exact replay horizon:

```text
>=247 tokens after the charged boundary anchor
```

The global oracle saw the future continuation and searched every stored prompt suffix, yet reached only 75, 28, and 5 tokens. Therefore no key rank, ANN index, distance metric, top-k width, or router can make prompt-only exact suffix memory meet the target.

The 405B metadata budget was not the obstruction. At 65,536 entries, rank 128, block length 256, fp16 keys, 32-bit token IDs, and 25% index overhead:

```text
keys: 16 MiB
blocks: 64 MiB
index: 20 MiB
total: 100 MiB = 0.09765625 GiB
full-hidden projection plus brute-force lookup: 0.02097152 GFLOP/query
```

Conclusion: prompt-derived exact decision content does not recur at the required horizon.

## Decisive interpretation

The tested prompt-derived reusable objects are now exhausted:

```text
semantic-state program: mean reuse about 1 token
prompt recurrence: maximum autonomous exact prefix 2 tokens
perfect-token recurrence repair: exact target on 68%–89% of tokens
prompt exact suffix replay: impossible global-oracle maxima 75 / 28 / 5
required strong reuse: 247 tokens
```

Do not recreate candidates that only change:

- static basis rank, block size, or semantic state count;
- norm precision, neuron ordering, or local/global refinement allocation;
- Hankel rank, order, ridge, or polynomial/bilinear lift;
- recurrence detector thresholds;
- nearest-neighbor rank, index, distance metric, or top-k width;
- speculative block length while dense target arithmetic remains per verified position.

## Current frontier — Experiment 040 Exact Dense-Operator Lower Bound

The next work must test the internal consistency of all four fixed requirements together:

```text
arbitrary dense checkpoint
exact original decisions
8 GiB resident memory
4B-class warm-decode traffic and compute
```

The candidate is an executable information/read lower-bound certificate, not a claim of impossibility without proof.

### Core adversarial lemma

For a dense affine operator:

```text
y = W x
```

consider an exact runtime that neither reads a weight degree of freedom `W[i,j]` nor stores an exact representation from which that degree of freedom's effect can be recovered. If `x[j] != 0`, construct two checkpoints `W` and `W'` that are identical on every observed/stored degree of freedom but differ only at `W[i,j]`. The runtime receives identical observations for both checkpoints, while exact output `y_i` differs. With a suitable downstream logit margin, the exact top-1 token can also differ.

Therefore a universal exact runtime must place every decision-relevant degree of freedom into one of:

```text
resident exact information
cold information read for the interaction
lossless metadata carrying equivalent information
```

Compression may change representation but cannot remove arbitrary checkpoint information while preserving exact behavior for every possible checkpoint and input.

### Experiment 040 proof obligations

1. Formalize the indistinguishable-checkpoint adversary for dense matrix-vector multiplication.
2. Extend it to top-1 decisions through a downstream linear margin.
3. Implement an executable adversarial checker that constructs `W/W'`, an input `x`, an inspection mask, and two different exact winners while the simulated runtime observations remain identical.
4. Derive resident/read/lossless-metadata inequalities for the 405B target.
5. Separate worst-case universality from empirical structure in real checkpoints.
6. State exactly which fixed assumption must be relaxed if the lower bound closes:
   - arbitrary checkpoint universality;
   - bit-exact/original-decision preservation;
   - 8 GiB residency;
   - 4B-class latency.

### 405B starting quantities

Use the established optimistic proxies:

```text
full Q4-equivalent target information: about 188.9883 GiB
full dense interaction compute: about 811.6985 GFLOP
resident VRAM: 8 GiB before KV/workspace
required full-interaction amortization: about 247 tokens
```

Do not hide skipped weights in uncharged metadata. Any automatic first-run representation must be counted by exact information content, host storage, transfer, and construction cost.

## Exact next steps

1. Create `research/exact-operator-lower-bound` from the updated `main`.
2. Add `docs/EXPERIMENT_040_EXACT_OPERATOR_LOWER_BOUND.md` with theorem assumptions and 405B inequalities.
3. Implement a small adversarial dense-operator simulator and tests.
4. Add a workflow producing machine-readable counterexample/lower-bound JSON.
5. Test both exact output and exact top-1 indistinguishability failures.
6. Record whether the fixed universal target is mathematically compatible, conditionally compatible only under checkpoint structure, or contradicted by the proven worst-case lower bound.
7. Update the ledger and this handoff before the next progress response.

## Correct communication

Use wording equivalent to:

> Experiment 039 rejected prompt-only exact decision-block memory even under a future-aware global oracle: maximum reusable blocks were 75, 28, and 5 tokens versus the required 247. Prompt-derived static, dynamic, repaired, and nonlocal execution programs are now exhausted. The 405B objective remains unchanged and unsolved. Experiment 040 now tests, with an executable adversarial proof, whether arbitrary dense checkpoints and exact decisions can coexist with 8 GiB residency and 4B-class traffic.
