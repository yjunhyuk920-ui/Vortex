# EXP-076 — Qwen3.5-0.8B Native MTP Accepted-Prefix Gate

## Question and evidence ceiling

Does the native MTP head in the pinned unchanged official Qwen3.5-0.8B
checkpoint causally produce a population-wide exact prefix long enough to keep
the revised EXP-074 weight-stationary surrogate alive after proposal, LM-head,
rejected-suffix, verification, correction, and fallback work are charged?

This is a Phase C small-real-checkpoint causal observation with an E1 ceiling.
It is not E2 unless complete generation actually replaces target operations.
It cannot validate vLLM latency, GPU behavior, 35B/122B routing, arbitrary dense
checkpoints, or 405B.

## E0 efficiency triage

The amortized operation is one cold target/expert weight stream across multiple
verified positions. The new information source is the unchanged checkpoint's
native MTP layer, already confirmed by EXP-075. A long accepted block can in
principle provide more than a tenfold traffic change, while a short prefix kills
the only retained EXP-074 branch before any scheduler or large-model download.

The mechanism is Qwen-specific and therefore auxiliary to the arbitrary dense
mission. Its value is the cheap falsification of the proposed surrogate rung,
not a universality claim. Scale transfer remains unverified even if this Gate
passes.

## Frozen inputs

Checkpoint:

```text
Qwen/Qwen3.5-0.8B
revision 2fc06364715b967f1860aea9cf38778875588b17
weight file bytes 1,746,942,600
weight SHA-256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
registered download bytes 1,769,904,871
```

Runtime:

```text
Python 3.12.13
PyTorch 2.6.0+cpu
Transformers 5.12.0
eager attention, bfloat16, CPU, 8 PyTorch threads
vLLM semantic reference a07086e4032e66aacae60ac2fc01e738096e9569
```

The full wheel lock is `experiments/exp_076/requirements.lock.json`. The
checkpoint manifest pins every required file name, byte count, repository OID,
and the LFS SHA-256 where exposed. No unregistered checkpoint file may be used.

Prompt authority is `experiments/exp_076/prompts.json`, file SHA-256
`46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a`.
There is one build and three held-out evaluation prompts for each of English,
Korean, code, structured JSON, math, and adversarial low-acceptance families.

## Causal protocol

For each prompt:

```text
exact committed prompt
-> unchanged target computes exact greedy token y0 and final hidden states
-> shift prompt tokens by one and pair them with target hidden states
-> native MTP full-attention layer proposes y1
-> recursively feed only each proposed token and prior MTP hidden/cache state
-> continue one fixed path to K=64
-> unchanged target verifies [y0, y1, ..., y64] on a cache clone
-> longest matching prefix only; ignore any later accidental match
-> commit y0, accepted drafts, and exact first correction/bonus
-> discard proposal/verification caches
-> reconstruct target cache from immutable committed state
```

The proposer function has no target-future-token input. `y0` is the already
computed exact target token supplied by the standard EAGLE/MTP alignment, not an
oracle future token. All registered K values `2,4,8,16,32,64` are prefixes of
the same K=64 path. No K-specific proposal rerun or exact-reference variant
selection is allowed.

Four states remain distinct:

```text
committed target hybrid cache
recursive MTP full-attention cache
target verification hybrid-cache clone
post-rejection target cache reconstructed from exact committed tokens
```

The committed cache is hashed before and after verification. Two independent
exact-commit reconstructions must have identical tensor hashes. Failure aborts
the Gate rather than becoming scientific evidence.

## Accounting and selection

The actual proposal cost is derived from safetensors shapes:

```text
P = numel(all 15 mtp.* tensors) + numel(tied target LM head)
```

The EXP-074 fixed-route target and baseline remain:

```text
T = 10,000,000,000 parameter-equivalent reads per target block
B = 1,000,000,000 parameters per native baseline token
```

For configured K and exact accepted prefix A:

```text
realized_ratio(K,A) = (T + K*P) / (A*B), A > 0
realized_ratio(K,0) = infinite / fail-closed
```

This charges every rejected proposal. Separately report target verification
compute as `(K+1) * target_text_parameter_count`; weight-stationary traffic does
not imply skipped arithmetic.

The ideal shape-derived acceptance minimum assumes `K=A` only for the necessary
threshold:

```text
A_p50 = ceil(T / (1.2*B - P))
A_p95 = ceil(T / (1.5*B - P))
```

If either denominator is nonpositive, the candidate is rejected without a
finite acceptance threshold.

Build prompts select exactly one K by minimizing:

```text
(realized p95, realized p50, zero-accept rate, K)
```

Any zero acceptance makes build p95 infinite. Only the selected fixed K is
judged on the held-out evaluation population.

## Promotion and rejection

Promotion requires all of:

- zero target-future reads, wrong accepts, prefix mutations, committed-cache
  mutations, exact-commit replay mismatches, and rollback recompute mismatches;
- held-out population p50 accepted prefix at least `A_p50`;
- held-out population p05 accepted prefix at least `A_p95`;
- the same p50 and p05 requirements for every required prompt family;
- held-out realized parameter-traffic p50 at most `1.2` and p95 at most `1.5`.

Pass decision:

```text
PROMOTE_TO_QWEN35_35B_A3B_ROUTE_UNION_TRACE_GATE
```

Scientific failure:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

Integrity failure:

```text
INVALID_NATIVE_MTP_EXECUTION_CONTROL_FAILURE
```

Dependency, download, unsupported CPU operation, storage, or timeout failure is:

```text
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

## Stop rule and prohibited continuation

On scientific failure, close the native-MTP long-block surrogate. Do not rescue
it with prompt cherry-picking, K sweeps beyond the registered set, sampling,
quantization, a different small Qwen size, a page scheduler, or a 35B/122B
download. Return the primary research portfolio to a materially different
query-time information source.

Passing authorizes only a separately preregistered 35B-A3B expert route-union
trace. It does not authorize a 122B download, physical scheduler, private target
server mutation, Phase D, E6, E7, or a dense-405B claim.
