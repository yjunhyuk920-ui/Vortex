# Independent Hugging Face bridge implementation review

Date: 2026-09-09
Method: static source-and-artifact review only. No bridge execution, tests,
model run, Git operation, GPU operation, configuration change, or code change
was performed.
Reviewed: `hf_bridge.py`, `scalar_producer.py` where needed to follow the
compiled call, `hf_results_v2/summary.json`,
`hf_results_v3/summary.json`, `hf_results_v3/manifest.json`,
`hf_results_v3/checksums.sha256`,
`PREREGISTRATION.md`, and `SCALAR_INDEPENDENT_REVIEW.md`.

## Verdict

The v3 artifacts are useful positive evidence for one deliberately restricted
CPU fixture. They record a correction of the v2 first-token V-cache stride
mismatch and a pass for the logical BF16 logits/K/V words and the serialized
incremental `DynamicCache` fields on one 40-token trace. They also record eight
matching sampled tokens, identical float32 probability words at each
`torch.multinomial` call, matching CPU RNG states before and after each call,
the same final CPU RNG state, and matching serialized final caches for that
generation trace.

This is not yet a complete exact-successor-state or reproducibility record.
The cache serializer omits storage offsets and alias/view relationships, the
full-prefill comparison checks cache words rather than the full cache state,
and v3 does not contain a direct per-sample generation-logit or full-prefill
logit result. The current `hf_bridge.py` contains new checks for those logits,
but its output schema differs from v3. With no executed-source hash in the v3
manifest or checksums, the current source cannot be retroactively treated as
the source that produced v3.

The cost report is a list of useful logical payload components, not a complete
upper bound or measurement. It omits actual Python/container memory, full
checkpoint residency, temporary tensors, cache allocation traffic, cumulative
`DynamicCache` copying, wall time, verification/evidence cost, SSD/PCIe/HBM/GPU
terms, recovery, and fallback. The result therefore remains restricted O6
evidence only:

```text
CORE_ADMISSION=false
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
O1-O5=OPEN
O6=PARTIAL
```

## What v2 and v3 actually establish

The v2 failure is scientifically useful. At step zero, the candidate and
native V words, dtype, shape, device, and ordinary layout agreed, but the V
stride differed:

```text
candidate: [16, 2, 2, 1]
native:    [16, 2, 16, 1]
```

The result correctly failed instead of accepting word equality as complete
state equality. The current implementation forms V as the native-style
`reshape(...).transpose(1,2)` view. The v3 summary then reports all 40 manual
steps passing its cache-state comparison, including stride. This closes that
specific mismatch for the recorded environment and trace.

Within the v3 summary's exact scope, the following are supported artifact
claims:

* The generated model `state_dict` hash was the same before and after the run.
* Forty incremental steps completed with matching raw BF16 logits and K/V
  logical words and matching fields captured by `cache_state`.
* The native 40-token full-prefill K/V logical words matched native incremental
  K/V logical words.
* Deterministic forward comparison did not change the global CPU Torch RNG
  state.
* One three-token generation prefix followed by eight sampling calls produced
  the same tokens, float32 probability words, per-call CPU RNG before/after
  states, final CPU RNG state, K/V words, and fields captured by `cache_state`.
* The current inspected compiled path calls the two-support `project`, creates
  K/V tensors, updates `DynamicCache`, and returns fixed zero logits. Static
  inspection of this source and `scalar_producer.py` finds no dense matrix
  multiply or full-model call in that compiled path.

These are bounded observations. They are not a proof over every checkpoint in
the generated family, every legal token sequence, or all 64 template positions.
The manual trace is the 32 token IDs once each followed by eight fixed repeats.
The generation trace is one realized continuation. A pass on those traces does
not supply the missing universal quantifiers.

## Finding 1: v3 is not bound to the currently inspected source

This is the first reproducibility obligation because it limits every claim
made from v3.

The v3 summary contains the combined field
`each_sample_logits_probability_RNG_exact` and has no
`native_full_prefill_logits_exact` field. The current source instead emits
separate `each_sample_logits_exact` and
`each_sample_probability_RNG_exact` fields and emits
`native_full_prefill_logits_exact`. Therefore the current source is newer than
the v3 artifact schema.

The v3 checksum file covers only `manifest.json`, `manual_records.json`,
`sampling_records.json`, and `summary.json`. The manifest records library
versions and the Torch build, but it does not hash `hf_bridge.py`,
`scalar_producer.py`, the preregistration, or a source archive. It also lacks
the invoked command, code revision, OS identity, CPU model/selected ISA,
relevant environment variables, and a package lock.

Result-directory overwrite refusal and artifact checksums protect the files
after creation. They do not prove which source created them. A new immutable
result version must hash or package the exact executed sources and record the
full invocation/environment. Until then, credit v3 only for fields actually in
v3; do not credit the newer source's unrecorded logit checks.

## Finding 2: “all cache fields” is broader than the serializer

`cache_state` records a tensor's dtype, device, shape, stride, `requires_grad`,
layout, and logical words after making a contiguous CPU copy. It recursively
records the current Python `vars()` of Transformers cache objects. This is much
stronger than comparing K/V words alone and was sufficient to catch v2's
stride mismatch.

It does not record at least:

* tensor `storage_offset` and backing-storage size;
* base-view and alias relationships among cache tensors and other live tensors;
* whether two fields share storage, or whether a later in-place write would be
  observed through another view;
* tuple versus list identity, because both serialize as a JSON list;
* class/module identity for every nested ordinary container and any
  continuation-relevant property not held in `vars()`.

Pointer identity can reasonably be excluded from a value-level continuation
contract. Storage offset and aliasing are not merely pointer identity: they can
affect legal views, mutation, append behavior, and allocator/copy cost. Either
capture these properties or state and prove a narrower successor-state ABI in
which callers cannot observe or mutate aliases and future `DynamicCache`
behavior depends only on the captured data.

Accordingly, rename the evidence claim to something like
`cache_serialized_fields_exact`, or expand the serializer and give the complete
field contract. `cache_all_fields_including_strides_exact` and
`final_cache_all_fields_exact` currently overstate the comparison.

The full-prefill check is narrower again. `full_incremental_equal` compares
only `cache_words(full.past_key_values)` with the incremental cache words. It
does not compare dtype, device, shape, stride, cache object fields, storage
semantics, or full-prefill versus incremental logits. Its result should be
labeled `native_full_prefill_incremental_KV_words_exact` unless a later bound
run adds the missing comparisons.

## Finding 3: v3 does not directly establish generation logits

The v3 sampling trace records probability words, CPU RNG before and after
`torch.multinomial`, and the sampled token. Exact probabilities and RNG are
good evidence for the eight frozen sampling calls. They do not imply exact
logits: softmax is invariant to adding the same constant to every logit, and
other different inputs can round to the same probability words.

The v3 source version requested `output_logits=True` from `generate`, but the
v3 summary has no separate generation-logit field. The current source now reads
`reference.logits` and compares it against compiled float32 score words; that
is the right additional comparison, but it belongs to a future source-bound
result, not v3. The same applies to the current full-prefill zero-logit check.

The 40-step manual path does compare native and compiled raw BF16 logits and is
valid evidence for that fixed trace. Because the lm-head is all `+0`, all logits
in this fixture are structurally zero. This makes the sampling distribution
uniform and the RNG comparison useful as a protocol check, but it does not test
preservation of nonuniform probabilities or sampling under nonzero logits.

The RNG claim is also specifically the global CPU Torch RNG used by
`torch.multinomial`. It does not cover an explicit `torch.Generator`, CUDA RNG
streams, Python RNG after construction, NumPy RNG, distributed RNG, or other
sampling procedures. Those can remain outside this fixture, but the interface
must say so.

## Finding 4: the prior KV state is stored but not causally consumed

The compiled step uses the cache's sequence length to select one of 64 K
templates and appends K/V. It never reads prior K/V values to calculate logits.
The native checkpoint has zero q/k/o/MLP/lm-head paths, so prior cache contents
also cannot affect the native logits. Nontrivial bounded V words are preserved
as stored successor data, but they are not consumed by a later nontrivial
attention/output path in this fixture.

This does not invalidate the word/state comparison. It precisely limits its
meaning: v3 demonstrates append-and-carry preservation for one inert-output
cache fixture. It does not demonstrate that a compiled executor can consume
an arbitrary original KV state and preserve dependent logits or the next
successor state. That broader causal step remains O2/O3 work.

The 64-position template is also a hard admission limit. The manual trace uses
positions 0 through 39; the generation run exercises a shorter prefix/sample
path. The remaining positions and all token-position combinations are not
covered by the recorded tests. A proof could lift the template construction
over all 64 positions, but test agreement alone does not do so.

## Finding 5: compile validation and failure recovery are incomplete

The compiler checks the exact two-support embedding table, finite bounded
`v_proj` weights, normalization weights equal to one, and all other parameters
equal to `+0`. It then verifies the two-support RMSNorm words for every token
and layer, compiles both V matrices, and constructs the 64-position zero-key
RoPE template. These checks make the fixture restriction visible.

Two implementation details still need closure:

1. RMSNorm word validation uses Python `assert`. Running Python with
   optimization can remove this check. An automatic constructor must use an
   explicit checked failure and record whether optimization was enabled.
2. `DynamicCache` layers are updated one after another. An exception or
   allocation failure after one layer update can leave a partially advanced
   cache. No rollback, transactional replacement, or recovery state is
   defined. Resource failure and fallback cannot be free under the governing
   contract.

`prohibit_dense_forward` blocks `torch.nn.Linear.forward` and the model's
`forward` method during the compiled call. It does not by itself block
`torch.nn.functional.linear`, `matmul`, or another imported dense primitive.
Static inspection of the currently reviewed `compiled_step` and `project`
finds no such escape, which is the substantive evidence. A source hash is
needed to bind that inspection to a run.

## Finding 6: the reported cost is partial payload accounting

The v3 summary reports useful exact logical components for this fixture:

| Reported component | v3 value |
|---|---:|
| Parameters scanned | 14,496 |
| RMSNorm words verified | 2,048 |
| V-project words read/transposed | 1,024 |
| Original plus compiled V-project BF16 payload | 4,096 bytes |
| Row-count payload | 128 bytes |
| Layer-scale payload | 4 bytes |
| 64-position, 2-wide zero-key template | 256 bytes |
| Selected V-project words per token | 64 |
| Row metadata words per token | 32 |
| Input words scanned per token | 64 |
| New K/V payload per token | 128 bytes |
| Prior K/V logical payload named as copied | `128*T` bytes |

These values are consistent as logical payload counts for two layers,
`n=32`, `m=16`, `head_dim=2`, and support two. The input scan is once per layer,
not once per output row. The current producer has no per-row `n`-element leaf
initialization.

The table is not a complete cost bound for the implementation that ran:

* The full model remains resident. From the reported 14,496 BF16 parameters,
  parameter payload alone is 28,992 bytes, before buffers and object overhead.
  The 4,096-byte figure counts only original plus compiled V matrices.
* `build_fixture` retains the separate generated `matrices` tensors after
  copying them into the model, adding another 2,048 BF16 payload bytes plus
  tensor/container overhead during the run.
* `compile_store` represents columns and counts as Python tuples of Python
  integers. Their real RAM is much larger than the compact BF16/u32 payload.
* Construction allocates the embedding expectation, normalized matrix,
  per-token prediction vectors, positions, probe, cos/sin, zero Q/K inputs,
  RoPE outputs, template clone, validation masks/temporaries, Python lists, and
  hashes. Peak live ranges and allocator fragmentation are not reported.
* Each compiled layer/token allocates an `n`-entry Python input list, active
  list/dictionary, per-row selected lists, output integers, a Torch tensor for
  V, an expanded/contiguous K tensor, and a fresh logits tensor. The report
  counts logical words but not bytes moved, Python operations, allocations, or
  deallocations.
* `DynamicCache.update` concatenates existing K/V as the sequence grows. The
  reported `128*T` is old logical payload at prior length `T`, not complete
  memory traffic. Across `L` sequential appends it names
  `128*L*(L-1)/2` bytes of old logical payload. Actual traffic also reads and
  writes that payload, reads/writes the new K/V, allocates the replacement,
  and frees or retains prior storage. Thus the cumulative cache work is
  quadratic for this implementation unless a different paid cache schedule is
  constructed and shown equivalent.
* Logit creation, BF16-to-float score conversion, clone, softmax, multinomial,
  CPU RNG state access, token extraction, and control are unquantified.
* Native-reference full prefill, 40 incremental forwards, generation,
  checkpoint hashing, comparison serialization, JSON/hex expansion, and
  evidence checksums are verification costs. The v3 evidence directory itself
  is 361,002 bytes, but no time, peak RAM, or SSD-write bound is recorded.
* No wall time, CPU cycles, peak RSS, SSD source/evidence traffic, PCIe, HBM,
  GPU allocation, kernel launch, synchronization, or same-machine baseline is
  measured or bounded.
* No finite recovery/rollback cost or paid dense fallback frequency/schedule is
  given. Inputs outside two support or position 64 are simply refused.

The native RMSNorm and RoPE calls used by compilation are finite for this small
fixture and are acknowledged in prose, but their actual computation and
temporary state must be counted. They cannot become a free decoder or free
per-matrix global adviser in a lift to the fixed mission. Likewise, the full
column copy and one count per row are paid matrix-specific data here; their
existence provides no selector or decoder for arbitrary dense hidden states.

At the mission scale, both original and compiled BF16 matrix payloads remain
`4P` bytes for `P` compiled parameters: 405B parameters would require about
1.62 TB decimal before the full model's other state, metadata, caches,
workspaces, SSD staging, transfers, allocator overhead, validation, recovery,
or fallback. Nothing in v3 provides a <=8-GiB single-GPU schedule or native 4B
Q4 latency/TTFT comparison.

## Precise claim that can be retained

A source-qualified version of the result may say:

> On the recorded Windows CPU environment with PyTorch 2.8.0+cpu,
> Transformers 4.55.4, one Torch thread, deterministic algorithms enabled,
> MKLDNN enabled, BF16 dtype, the fixed generated two-support Llama fixture,
> the fixed 40-token manual trace, and the fixed eight-sample generation trace,
> v3 matched the logical BF16 logits/K/V words and the cache fields serialized
> by the v3 harness for the incremental comparison. It also matched the
> recorded float32 sampling probabilities, sampled tokens, and global CPU
> Torch RNG transitions. The v2 first-update V-stride mismatch was repaired in
> v3.

Do not strengthen that statement to all cache state, all generation logits,
all 64 positions, all legal continuations, arbitrary bounded V matrices,
arbitrary Hugging Face models, CUDA, a public unchanged checkpoint, 405B,
<=8-GiB, or target latency.

## Next obligations

1. Create one fresh immutable result version that packages or hashes the exact
   executed bridge, scalar producer, preregistration, dependencies, invocation,
   OS/CPU/ISA/backend environment, and output artifacts. Do not rewrite v3.
2. In that bound result, separately report raw generation logits, full-prefill
   logits, full-prefill versus incremental cache state under the declared
   serializer, probabilities, samples, and every relevant RNG transition.
3. Freeze the successor-state ABI. Either add storage offset/alias information
   and other observable fields, or prove that the legal continuation interface
   cannot observe them. Rename existing “all fields” claims to match what is
   serialized.
4. Replace removable `assert` validation with an explicit refusal. Define
   atomic cache update, rollback, resource-failure behavior, and paid fallback
   or terminal refusal.
5. Give a finite fixture proof for every admitted token sequence and positions
   0 through 63, or keep the claim trace-specific. Treat the cached V values as
   append-only evidence until a nontrivial consumer proof exists.
6. Produce separate measured and analytic ledgers for compile, compiled
   runtime, native reference, verification, evidence storage, recovery, and
   fallback. Include peak CPU RAM, logical and physical bytes moved, cumulative
   cache-copy cost, wall time, SSD, and all allocator/control terms.
7. Return to the unchanged mission separately: arbitrary public unmodified
   405B-class checkpoints and all legal continuations, exact output/logits/RNG
   and successor KV, <=8-GiB single-GPU peak, and the frozen same-machine
   native-4B-Q4 latency/TTFT population. No restricted fixture pass discharges
   those obligations.

The existing rejected full-reconstruction, free-witness, row-replay,
full-scan, catalog, free-decoder, global-advice-per-matrix split, sparse-delta,
and arbitrary-dense-change assumptions remain rejected. This bridge is one
continued restricted E1 producer artifact, not three new qualifying execution
principles and not a route around the fixed mission.
