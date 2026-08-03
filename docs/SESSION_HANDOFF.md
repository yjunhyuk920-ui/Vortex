# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or architecture-specific adapter authoring;
- original-model quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence is below E4. Do not claim the target is solved or proven feasible.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, branch, workflow, PR comments, and raw result JSON

Before a user-facing progress/completion answer after meaningful work, commit the research ledger and this handoff. This rule is permanent and merged into `main` through PR #30.

## Current repository state

No research PR is promoted. The latest experimental sequence is closed with raw evidence:

```text
PR #36  causal semantic-state signed program routing       rejected
PR #37  prompt-compiled Hankel decision program            rejected
PR #38  perfect-oracle sparse Hankel repair                rejected
```

Latest branch evidence heads:

```text
PR #36 research/semantic-state-program-routing
       499e5001c21d782adf79fba69ce6f2d445c0cb5e

PR #37 research/prompt-hankel-decision-program
       12f859e4ec288f0d38b29d8b71e494bdc29f6586

PR #38 research/oracle-sparse-hankel-repair
       13e3f60876199e4b06577ca51e9fd71f575cb134
```

Successful experiment workflows:

```text
PR #36 Semantic state program routing gate  run 30778002226
PR #37 Prompt Hankel decision program gate   run 30778715832
PR #38 Oracle sparse Hankel repair gate      run 30779062125
```

All branch-owned tests, Python 3.10/3.12 CI, validation, real-model measurements, aggregation, artifact upload, PR reporting, and evidence commits completed.

## Latest decisive results

### PR #36 — semantic program routing

A causal router used only the previous completed token's final hidden state. The best held-out point was `K=8`, rank 4:

```text
active program: 0.9901 GiB
host bank: 7.9211 GiB
mean run length: 1.364 tokens
program switching: 0.7261 GiB/token
activation perpendicular mean: 42.28%
dual perpendicular mean: 53.91%
activation p95: 99.50%
dual p95: 99.84%
```

Increasing state count shortened reuse to about 1.07 tokens and raised switching traffic.

Conclusion: close precompiled semantic-state program banks at the tested scale.

### PR #37 — prompt-compiled dynamic recurrence

Prompt final hidden states were compiled into controlled Hankel recurrences with linear, quadratic, bilinear, and full lifts. Projected 405B program memory/compute was small; rank32/control16/order2/full used approximately:

```text
program memory: 0.00673884 GiB
hot compute: 0.008217664 GFLOP/token
prompt projection build: 201.73 GFLOP
```

Real autonomous continuation:

```text
algorithm-runtime: best exact prefix 1 / 256
distributed-database: best exact prefix 1 / 256
korean-plm-governance: best exact prefix 2 / 256
required: at least 247 / 256
```

Best teacher-forced points remained far below the gate. Higher-order lifted recurrences sometimes became non-finite; non-finite evidence is stored as JSON null rather than clipped.

Conclusion: close prompt-only low-rank hidden-trajectory recurrence. Do not tune only rank, order, ridge, or lift.

### PR #38 — perfect-oracle sparse repair

The repair oracle knew the exact target token before deciding whether to accept the recurrence or charge a full exact repair. It is strictly stronger than any deployable causal detector.

One optimistic 405B exact repair was charged as:

```text
188.9883 GiB traffic
811.6985 GFLOP
required mean repair interval >=247 tokens
```

Best points over 256 tokens:

```text
algorithm-runtime:
  repairs 226
  accepted 11.72%
  mean interval 1.133
  maximum interval 3
  traffic 166.84 GiB/token
  compute 716.58 GFLOP/token

distributed-database:
  repairs 229
  accepted 10.55%
  mean interval 1.118
  maximum interval 3
  traffic 169.06 GiB/token
  compute 726.09 GFLOP/token

korean-plm-governance:
  repairs 174
  accepted 32.03%
  mean interval 1.471
  maximum interval 3
  traffic 128.45 GiB/token
  compute 551.70 GFLOP/token
```

Conclusion: close Hankel recurrence plus sparse exact repair, including every weaker detector. Exact repair is effectively tokenwise.

## Decisive interpretation

The exact decision state changes too quickly for every tested low-dimensional tokenwise object:

```text
static semantic program reuse: about 1 token
prompt dynamic recurrence: at most 2 exact tokens
perfect-oracle recurrence repair: exact execution on 68%–89% of tokens
```

This is stronger than a failed implementation. It rejects the tested representation family even under optimistic or impossible oracles.

Do not recreate candidates that only alter:

- static basis rank/block size/state count;
- norm precision or neuron ordering;
- equal versus global refinement;
- Hankel rank/order/ridge/polynomial lift;
- recurrence detector thresholds;
- ordinary speculative block length while dense target arithmetic remains per position.

## Current frontier — Nonlocal Exact Decision Memory

The next candidate must stop extrapolating the transformer hidden trajectory token by token.

First Gate:

```text
build keys only from exact prompt positions
store exact token blocks following those prompt positions
query with a held-out continuation state's causal final-hidden signature
retrieve nearest nonadjacent prompt states
measure exact future-block agreement without recurrence
```

This is an optimistic test of whether exact decision blocks recur nonlocally at all.

Required equations and thresholds:

```text
M_keys + M_blocks + M_index + M_KV + M_work <= 8 GiB
lookup + validation <= 4B-class compute envelope
held-out exact reusable block >=247 tokens
no continuation token or hidden state in memory build
reject trivial EOS/repeated-token traces
```

If prompt-only nonlocal memory also yields one- or two-token reuse, prompt-derived execution programs are exhausted and work must confront the full exact operator lower bound directly.

## Exact next steps

1. Merge the current documentation PR into `main` after full CI.
2. Create a fresh branch from the new `main` for Experiment 039.
3. Derive 405B storage for key ranks, memory entries, and stored decision-block lengths.
4. Implement nearest-state nonlocal block reuse on the same three exact 256-token traces.
5. Keep prompt build and continuation evaluation temporally disjoint.
6. Commit raw JSON, close/promote the PR from measured evidence, then update this file and the ledger again.

## Correct communication

Use wording equivalent to:

> E2 research has now rejected static semantic programs, prompt-compiled low-rank dynamics, and even perfect-oracle sparse repair on real TinyLlama continuations. The best recurrence preserved two exact tokens, while the impossible repair oracle still invoked exact target execution on most tokens, projecting at least 128 GiB/token and 552 GFLOP/token. The 405B objective remains unchanged and unsolved. The next Gate tests nonlocal exact decision-block reuse rather than tokenwise state extrapolation.
