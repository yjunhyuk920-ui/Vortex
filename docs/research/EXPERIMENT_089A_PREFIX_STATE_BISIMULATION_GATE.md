# EXP-089A — Prefix-State Bisimulation Gate

## Status

```text
FROZEN BEFORE PUBLIC-CHECKPOINT RESULT
Phase: A/C constructive state-contract validation
Evidence ceiling: E2 state witness
Performance core: false
TARGET-W / 8 GiB / 4B-class latency: NOT TESTED
```

## 1. Purpose

EXP-088B showed that retaining or zeroing cross-layer checkpoint pages does not create a smaller exact successor-state program: only the full mask was exact. The corrected EXP-087A result then showed that a compact degree-two finite-context program can interpolate all 96 build MLP states but produces zero exact unseen MLP vectors, even under a target-seeing oracle.

Those experiments required exact intermediate or KV tensors from the candidate. The fixed project contract, however, also permits a `STATE-BISIMILAR` executor state. EXP-089A tests a stricter and simpler behavioral state:

```text
compiled state = exact token prefix + exact RNG state
```

The Gate does not claim that replaying the original model is fast. It establishes whether future research may target exact next-token decisions without requiring candidate hidden/KV equality after every step.

## 2. Reference transition

For immutable checkpoint `W`, pinned runtime ABI `A`, token input `u_t`, and RNG state `r_t`, let

```text
(y_t, S*_{t+1}) = F_{W,A}(S*_t, u_t, r_t)
```

where `S*_t` contains the official incremental cache and all runtime state needed by the declared decoder.

Let the complete token prefix at step `t` be

```text
p_t = prompt || y_0 || ... || y_{t-1}.
```

Define `Replay_{W,A}(p_t)` as the exact official execution schedule:

1. prefill the frozen prompt once;
2. feed each generated token in `p_t` one at a time with the returned official cache;
3. preserve the declared dtype, attention implementation, sampling algorithm, and RNG state.

## 3. Bisimulation relation

The compiled state is

```text
S_hat_t = (prompt_ids, generated_ids, rng_state).
```

Define

```text
S_hat_t ~ S*_t
```

iff all of the following hold:

- `Replay_{W,A}(prompt_ids || generated_ids)` produces the same official cache bytes as `S*_t`;
- it produces the same next-token logits bytes as `S*_t`;
- its position equals the prefix length;
- its RNG state is byte-identical to the reference decoder RNG state.

The relation is checkpoint- and ABI-specific, but it does not store a hidden-state lookup table or a response table.

## 4. Proof obligations

### Initialization

For the frozen prompt, both sides execute the same official prefill, so the relation holds before the first generated token.

### Output preservation

If the relation holds, reference and replay logits plus RNG bytes are equal. The deterministic greedy or frozen sampled selector therefore returns the same token.

### Transition preservation

Both states append the same token and advance the same RNG state. Replaying the longer prefix uses the same official prefill and one-token update sequence as the incremental reference, so the relation holds at the next step.

### All suffixes

Initialization plus output and transition preservation imply by induction that every legal future suffix produced by the declared decoder is identical.

### Finite precision

The Gate compares raw logits and cache bytes, not tolerances. A cache/logit/RNG byte mismatch invalidates the witness.

## 5. Executable interface

The Gate implements the required constructive object:

```text
compile_checkpoint(reference_model) -> pinned identity manifest
initialize_state(prompt, rng) -> PrefixCompiledState
decode_step_by_replay(model, PrefixCompiledState)
    -> next_token, next_PrefixCompiledState, resource_trace
```

`decode_step_by_replay` deliberately reconstructs the official state from the prefix. This is a correctness witness and baseline only. It is prohibited from being reported as a fast executor.

## 6. Frozen public checkpoint

```text
model        HuggingFaceTB/SmolLM2-135M
revision     93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader       transformers.LlamaForCausalLM.from_pretrained
reference    official eager BF16 CPU path
```

The model weights are unchanged.

## 7. Frozen population

Four disjoint cases are used:

- English greedy decode;
- Korean greedy decode;
- code greedy decode;
- structured-output sampled decode with frozen seed, temperature, and top-k.

Each case executes 16 consecutive transitions. This Gate validates a state relation, not a performance component, so it does not claim the repository's 128-step performance-promotion boundary.

## 8. Exact controls

For every transition record:

```text
prefix digest
reference/replay logits SHA-256
reference/replay cache SHA-256
reference/replay RNG SHA-256
reference/replay selected token
first byte mismatch if any
reference target calls
replay target calls and replayed token positions
```

Additional controls:

- changing one generated token changes the prefix-state digest;
- changing one cache byte changes the cache digest;
- two samplers restored from the same RNG bytes select the same token and successor RNG bytes;
- malformed empty prompts, invalid sampling parameters, or unsupported cache structures fail closed.

## 9. Gate

Success requires:

```text
zero logits-byte mismatches
zero cache-byte mismatches
zero token mismatches
zero RNG-state mismatches
all configured transitions complete
checkpoint identity and runtime pins pass
```

Pass decision:

```text
ACCEPT_PREFIX_STATE_BISIMULATION_WITNESS_FOR_TOKEN_DECISION_RESEARCH
```

Failure decision:

```text
INVALID_PREFIX_STATE_BISIMULATION_WITNESS
```

A pass does not promote replay as a runtime. It authorizes the next compiler boundary:

```text
ExactTokenDecision(W, prefix_state) -> token, proof/resource trace
```

The successor state is then obtained by appending the exact token rather than by preserving candidate hidden/KV tensors.

## 10. Resource boundary

Prefix storage is small, but full replay is deliberately expensive:

```text
replay work through step t = prompt prefill + t incremental official calls
```

All replay calls and replayed token positions are measured. No speedup, bandwidth saving, 8-GiB result, or 405B projection is derived from this Gate.

## 11. Next research class

After a pass, future constructive research must attack exact next-token determination over the prefix state. Approximate hidden states are permitted only when accompanied by a sound token-decision certificate or an exact token program. Dense replay remains a reference oracle, not an admissible hidden fallback for the final core.
