# EXP-089A latest result — Prefix-State Bisimulation Gate

## Identity

- source commit: `f271e52ad99ade62f39c73fe670015ff30d713cd`
- raw result: `results/exp_089a/f271e52ad99ade62f39c73fe670015ff30d713cd/result.json`
- public checkpoint: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- checkpoint tensor SHA-256: `ee567358e6d45900612a1a276fd068d1c5fd999a451cd67bae55cb2c01446308`
- deterministic core: `a3d2a7a79fc59c0f1c18e22d001c052f2bd49c0568e22bfaf917d11238d2f820`

## Authoritative decision

`ACCEPT_PREFIX_STATE_BISIMULATION_WITNESS_FOR_TOKEN_DECISION_RESEARCH`

## Measured result

- configured exact token decisions: `64`
- token mismatches: `0`
- logits-byte mismatches: `0`
- cache-byte/digest mismatches: `0`
- RNG-state mismatches: `0`
- official incremental reference target calls: `68`
- replay target calls: `612`
- replayed token positions: `3791`
- model-free fault controls: `True`

## Meaning

The accepted relation, when the Gate passes, is:

```text
compiled state = exact prompt/generated token prefix + exact RNG bytes
```

Replaying that state through the pinned official checkpoint reproduces the official incremental logits and complete KV cache bytes. This establishes a checkpoint/ABI-specific behavioral state witness. It does **not** make full replay an efficient executor.

## Next boundary

A pass authorizes research on:

```text
ExactTokenDecision(checkpoint, prefix_state) -> exact token + proof/resource trace
```

The first promoted Gate is a hot-prefix early-release oracle: measure whether a fixed shallow prefix of the actual decoder can identify or tightly rank the final exact token for long blocks, while the unchanged suffix remains the exact state constructor. No replay, dense suffix, or target-output oracle may be hidden in the final selector.

TARGET-W, a cheap exact token program, 128-step performance promotion, 8-GiB residency, and 4B-class p50/p95 remain `NOT TESTED`.
