# Experiment 045 — Bounded Exact Decision-Index Compiler Gate

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and purpose

This is an E1/E2 bounded real-checkpoint compiler Gate. It uses an unmodified TinyLlama checkpoint with no training, compiles a declared finite grammar and horizon, exports exact greedy transitions to the Experiment 044 compact40 VM, and replays them without model execution.

It is not a universal compiler, does not cover arbitrary prompts, does not run a 405B model, and does not establish target GPU latency.

Experiment 044 proved that a host-indexed exact pointer VM is functionally viable. The unresolved question is construction:

> Can useful exact next-token records be generated automatically from an unmodified checkpoint without enumerating an intractable context space?

## Finite grammar and exact denominator

The checked grammar is stored in:

```text
experiments/decision_index_compiler_grammar.json
```

It contains:

```text
symbols: A, B, C
counts: 8, 12
templates: 2
full combinations: 12
compiled combinations: all A/B combinations = 8
held-out combinations: all C combinations = 4
maximum continuation horizon: 8
measured horizons: 2, 4, 8
```

Each prompt asks TinyLlama to repeat one capital letter with single spaces and no explanation. The model still determines the exact greedy output; instructions are not treated as truth.

The exact compiled-state denominator at horizon `H` is:

```text
compiled_prompt_paths * H
```

A duplicate control repeats one compiled prompt under a different ID. It is excluded from the unique grammar denominator but included in a separate exact-deduplication check.

## Exact state key

The sound v1 state key is the complete token prefix:

```text
state_key = exact_chat_prompt_token_ids || exact_generated_token_ids_so_far
```

Two states merge only if these integer sequences are byte-for-byte identical. No hidden-state distance, approximate hash, semantic label, or learned router may merge states.

This key is intentionally conservative. It makes exact replay sound but is expected to expose memoization growth.

## Original-checkpoint compilation

For every compiled and held-out grammar prompt:

1. apply the checkpoint tokenizer chat template;
2. run exact greedy prefill and decode for eight tokens;
3. continue after EOS so every path has the declared fixed horizon;
4. record the exact prompt token IDs and generated token IDs;
5. count one prefill and one decode interaction per generated token boundary;
6. record EOS position, distinct-token count, and repetition diagnostics.

No continuation token or state is used before the original checkpoint generates it.

## Decision graph

For each compiled prompt and step `t`:

```text
node key     = prompt_ids || continuation_ids[:t]
record token = continuation_ids[t]
successor    = node at t+1, or terminal at horizon
```

Nodes are assigned deterministic addresses after all exact keys are collected. Duplicate-control starts must point to the same first node and reuse the same path.

Growth is measured at horizons 2, 4, and 8:

```text
path records
unique exact-prefix nodes
exact duplicate records removed
deduplication ratio
nodes added per additional token
```

No extrapolation beyond measured horizons is treated as proof.

## compact40 token codebook

Experiment 044 compact40 stores a four-bit output code. Experiment 045 therefore builds a global exact token codebook from compiled continuations:

```text
codebook index: 0..15
value: full TinyLlama token ID
```

The VM record stores the four-bit codebook index. A separate manifest stores the exact full token IDs and prompt/start mapping.

Promotion requires:

```text
compiled continuation distinct token IDs <= 16
```

If the model produces more than 16 distinct compiled tokens, compact40 export fails honestly. The compiler must not silently truncate or remap collisions.

## VM export and replay

The maximum-horizon graph is exported through the Experiment 044 compact40 builder. The output bundle contains:

```text
compact40 VM file
JSON manifest
model identifier
grammar hash
token codebook
prompt ID to VM start index mapping
horizon
node counts
checksums
```

Every compiled and duplicate-control path is replayed through `DecisionVMReader` without model execution. Four-bit codes are decoded through the manifest codebook and must reproduce every original generated token exactly.

## Held-out coverage and fallback

Held-out C-symbol compositions are not compiled. For each held-out exact prefix, the evaluator checks whether the exact state key exists in the compiled graph.

Report:

```text
held-out state denominator
compiled hits
fallback tokens
coverage
first miss position
```

A missing state requires an original-model fallback. The evaluator may use already collected exact held-out traces only as ground truth; they are never inserted into the compiled graph.

Exact-prefix identity is not expected to generalize across changed prompts. A zero held-out coverage result is a valid rejection of exact-prefix memoization as a broad compiler, not a failed implementation.

## Required evidence

```text
original checkpoint and revision when available
grammar combinations and split
model calls and generated tokens
build time
unique nodes by horizon
exact duplicate reuse
codebook size and token IDs
VM and manifest bytes
compiled replay exactness
held-out coverage and fallback
EOS and repetition diagnostics
```

## Promotion and rejection

The bounded compiler implementation passes if:

- all eight compiled grammar paths are generated from the original checkpoint;
- duplicate-control exact states deduplicate completely;
- the codebook fits 16 entries;
- compact40 build and checksum verification pass;
- every compiled path replays exactly without model execution;
- held-out coverage/fallback is measured without contaminating the graph.

The architecture advances beyond memoization only if measured state growth or held-out coverage shows nontrivial reuse not explained by exact duplicate prompts.

Expected classification possibilities:

```text
bounded compiler correct, growth linear, held-out coverage zero:
    accept implementation; reject exact-prefix memoization as universal mechanism

bounded compiler correct with substantial exact reuse:
    advance to larger grammar/horizon and certified state equivalence

codebook or replay failure:
    repair implementation; do not interpret as model evidence
```

## Strict scope

A complete eight-prompt grammar is not arbitrary language. Exact compiled replay does not preserve quality outside the domain. Build calls and grammar enumeration are mandatory costs.

Do not report:

- bounded grammar coverage as universal coverage;
- duplicate prompt reuse as semantic generalization;
- a TinyLlama result as 405B evidence;
- generated trace memoization as a replacement for the checkpoint on unseen prompts;
- CI build time as target-machine build time.

The next direction after this Gate depends on evidence. Linear growth and zero held-out coverage would move research toward certified state quotienting or adaptive on-demand compilation, while successful exact reuse would justify a broader compiler.
