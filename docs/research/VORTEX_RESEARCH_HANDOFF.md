# VORTEX research handoff

Status date: 2026-08-17

This document is the authoritative compact handoff for the current VORTEX research line. It does not replace the raw experiment records, root ledgers, or validation contracts. It records what later sessions must preserve, what must not be promoted, and the next constructive target.

## 1. Fixed mission

Build an executor that supports an arbitrary publicly released, unmodified Hugging Face dense autoregressive Transformer, including a real 405B-class checkpoint, under all of the following conditions:

- executor/runtime replacement only;
- no retraining, distillation, fine-tuning, LoRA, model editing, or prompt-specific training;
- declared output contract preserved;
- one GPU with peak VRAM at or below 8 GiB;
- batch size 1;
- same-machine warm p50 time/token at or below 1.2x a native 4B Q4 baseline;
- same-machine warm p95 time/token at or below 1.5x that baseline;
- CPU RAM, SSD, PCIe, preprocessing, auxiliary artifacts, synchronization, and cold-start costs charged rather than hidden;
- no remote or external accelerator and no dense-executor fallback disguised as success;
- independent reproduction from pinned code, checkpoint revisions, raw logs, and hashes.

The mission is not reduced when a candidate fails.

## 2. Output-contract scopes

The repository distinguishes the following scopes. They must never be silently mixed.

### EXACT-E

For a fixed checkpoint, tokenizer, numerical ABI, prompt, decoding algorithm, and RNG tape, the executor produces the same token sequence as the declared reference. A dynamic executor must also produce the exact reference successor state or a state proven behaviorally bisimilar for all legal future continuations.

### QUALITY-Q

Approximate execution evaluated under preregistered perplexity, benchmark, generation-quality, and latency gates. QUALITY-Q evidence is not EXACT-E evidence. No QUALITY-Q threshold may be invented after observing results.

### Model-family labels

- `ALL-WEIGHTS`: adversarial legal weight assignments for an architecture class.
- `PUBLIC-FIXED`: named public pretrained checkpoints that the final mission must support.
- `SINGLE-TARGET`: one frozen checkpoint revision.
- `TOY-CONSTRUCTION`: a bounded construction used for proof or falsification.

A theorem for ALL-WEIGHTS does not automatically describe one public checkpoint. Structure found in one public checkpoint does not automatically generalize to all checkpoints.

## 3. Honest current acceptance state

```text
actual public 405B end-to-end execution        NOT TESTED
peak GPU VRAM <= 8 GiB                         NOT TESTED
same-machine 4B p50/p95 latency gate           NOT TESTED
full target correctness suite                  NOT TESTED
complete dynamic executor core                 NONE
target E7 acceptance                           NOT ACHIEVED
```

No equation, toy family, reconstructed local tree, local-only commit, or small-model result changes this state.

## 4. Provenance correction

The public research baseline verified for this handoff is:

```text
repository: yjunhyuk920-ui/Vortex
branch:     research/exp-082a-differential-spanning-tree
base SHA:   03737a8e4aa76bd3c23a0c38478a9079c27e649d
open PR:    #83, draft
```

The following previously reported SHAs were reconstructed local commits and must not be cited as remote VORTEX history:

```text
6153a73c6b3d17dbf000a0c6592ae49ee323ce9a
a8a9cfc32b6f2faa046094d5794ffe0c26a28f34
f34124884fc6cc74518150eb62c3d1e3c71ca581
```

Useful claims from those local bundles may be reimplemented and recommitted only after review against the real remote branch. Their existence alone is not remote evidence.

## 5. Research operating principles already established

Later sessions must preserve:

- proof-first and cheapest-kill-first research;
- fixed pass/fail thresholds before observing results;
- explicit assumption register and mechanism fingerprint;
- separation of `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED` evidence;
- raw logs, deterministic regeneration, and checksums;
- failure as permanent project data rather than a reason to lower the mission;
- no synthetic or small-model result promoted to 405B performance;
- no operation-count ratio promoted to wall-clock performance;
- no hidden movement of costs into RAM, SSD, PCIe, preprocessing, artifact size, batching, another user, or dense fallback;
- no new name treated as a new mechanism;
- no user-facing research report before meaningful repository changes are committed.

## 6. Earlier exact-structure and reuse experiments

The repository contains a long sequence of exact reuse, structure, coding, factorization, causal-selection, and residual-source experiments. The canonical details remain in the root ledgers and experiment records.

Important established examples include:

- exact cached K/KV reuse did not produce useful exact duplicate coverage and did not reduce warm work or reads;
- exact Q4 row/prototype and sparse-delta mechanisms did not provide a model-wide saving;
- exact Kronecker/tensor and low-rank routes did not produce a universal core under their frozen scopes;
- Causal Residual Atlas and related bilinear-span/codebook routes were rejected by their preregistered gates;
- routing among materialized linear leaves supplies no independent answer information;
- renaming, rank sweeps, prompt sweeps, tolerance changes, and nearby factorization variants do not reopen a closed mechanism family.

Read `FAILED_APPROACHES.md`, `FAILED_APPROACHES_RECENT.md`, `DECISION_LOG.md`, and `NEXT_EXPERIMENT.md` before revisiting any of these mechanisms.

## 7. Breakthrough-frontier sweep: retained results and limits

### 7.1 Exact three-lane signed-Q4 packed dot

A bounded-word primitive was constructed for three signed-Q4 integer pairs. With biased radix packing, the middle convolution digit recovers the exact three-term dot product. The reported exhaustive domain was all `16^6 = 16,777,216` input combinations, with zero mismatch in the local evidence bundle.

Retain only this claim:

```text
EXACT_FOR_SIGNED_Q4_INTEGER_TARGET_ONLY
```

Do not promote it to BF16/FP16 model exactness. It omits scale/zero point, BF16 multiplication, FP32 reduction order, FMA/rounding semantics, Transformer state, and token-sequence equivalence. Prepacked layouts expand payload, and no target-GPU wall-clock saving was measured.

### 7.2 Carry-free giant-integer contraction

The required radix width expands checkpoint storage before giant-product work and workspace are charged. It is rejected as a primary core because it worsens the dominant data footprint rather than removing it.

### 7.3 Exact multi-token activation-difference forest

The identity `W x_child = W x_parent + W(x_child - x_parent)` is exact. Its acceleration requires extremely sparse exact activation deltas. Adversarial dense deltas return the mechanism to dense work. It is not an unconditional core.

### 7.4 Full separator/function tabulation

Direct materialization of a full dense input separator or truth table has exponential artifact growth. This closes that tabulation arm, not every possible exact circuit representation.

### 7.5 Causal low-rank premise

A constructive finite-word family showed that a broad dense autoregressive class need not have an automatically low-dimensional causal query family. The claim must not be expanded into a measured statement about the reachable query dimension of a named 405B pretrained checkpoint.

## 8. ROOT-ESCAPE audit: corrected claims

The audit corrected several misleading statements:

1. The signed-Q4 packed primitive is not a BF16/FP16 exact primitive.
2. `4/6` abstract operation counting is not a GPU performance ratio.
3. Derived ratios such as `56.25x` from unlike source-level operations are not measured latency.
4. `84.375 coefficient contributions per physical operation` is not a universal necessary condition for every exact algorithm.
5. A software-visible `STATIC_LINEAR(operator_id, x)` interface does not eliminate internal weight storage, reads, or MACs when its emulator performs the same dense operation.

The replacement physical lower-bound ledger is resource-specific:

```text
T_token >= max(
    storage_bytes / storage_bandwidth,
    PCIe_bytes / PCIe_bandwidth,
    VRAM_bytes / VRAM_bandwidth,
    instruction_count_i / throughput_i,
    dependency_time,
    communication_time,
    synchronization_time
)
```

Each term requires its own assumptions and evidence.

## 9. Self-contained artifact bound

For a finite checkpoint family whose behavior map is injective, a runtime that never reads any other checkpoint-dependent cold data and whose only checkpoint-dependent object is a self-contained artifact `A(W)` requires at least `log2 |F|` artifact bits.

This pigeonhole result is valid in its stated scope. It does not rule out a large cold artifact queried adaptively at runtime, and it is not a 405B latency theorem.

## 10. Decisive cold-backed gap: retained results and limits

### 10.1 Cold-backed model

The research separated persistent artifact space, hot checkpoint-dependent state, logical word size, adaptive probe count, logical replies, physical pages/bytes, compute, preprocessing, and wall-clock latency. These quantities must not be collapsed into a single information count.

### 10.2 ALL-WEIGHTS inner-product tradeoff

For an ALL-WEIGHTS exact GF(2) inner-product query family with `H` hot bits, `S` arbitrary nonlinear cold cells of `w` bits, and at most `p` adaptive probes, the communication reduction gives:

```text
H + p (w + ceil(log2 S)) >= n
```

Retain the theorem only for its explicit ALL-WEIGHTS full-query scope. Conversion from logical probes to pages, bytes, or latency requires separate physical assumptions.

### 10.3 Scalable Llama-operator hard family

A scalable mathematical construction using Llama-like standard operators was derived for an exact inner-product/parity family. Its parameter cost was `O(n^2 log n)`. At a 405B parameter budget it encoded too little independent behavior information to exceed 8 GiB of hot state. Therefore this construction does not establish a useful 405B/8-GiB lower bound.

The official `transformers.LlamaForCausalLM.from_pretrained()` path was implemented in a local-only workflow but was not actually executed in the reported environment. Do not mark it measured until a real run exists.

### 10.4 One-probe selector witness

A 257-bit indexed selector showed that checkpoint information or behavior injectivity alone does not imply a large token-time cold-probe requirement: a query that supplies the address can be answered with one word probe.

This validly rejects the generic inference:

```text
large checkpoint information => every token must read a large fraction
```

It does not implement an autoregressive dynamic executor. Its state update was zero and it does not address successor-state generation.

## 11. Meaning of the former open-lemma label

Do not use `ONE_EXACT_OPEN_LEMMA_REMAINS` to imply that the project is nearly complete. The statement attached to that label was essentially the original mission restated as a constructor-or-lower-bound disjunction.

The correct handoff is:

```text
Static information-counting routes have been narrowed.
The constructive problem is the exact dynamic execution of a named public checkpoint,
including its successor state and its physical resource ledger.
```

## 12. Current constructive frontier

The active research target is a checkpoint compiler and dynamic executor:

```text
compile_checkpoint(reference_model) -> artifact
initialize_state(artifact, prompt) -> compiled_state
decode_step(artifact, compiled_state, input, rng_state)
    -> next_token, next_compiled_state, resource_trace
```

For reference transition

```text
(y_t, S*_{t+1}) = F_W(S*_t, u_t, r_t)
```

the compiled executor must produce the same token and either:

- `STATE-EXACT`: the exact reference successor state; or
- `STATE-BISIMILAR`: a state with a defined and tested relation preserving every legal future output transition.

A one-step token lookup without a successor state is not a core executor.

## 13. Minimum constructive promotion gates

A mechanism is not a surviving core until it demonstrates all applicable items:

```text
G1  actual named public checkpoint loaded through the official runtime
G2  official reference forward/generation executed
G3  at least one complete real Transformer layer replaced
G4  >=128 consecutive decode transitions with token agreement
G5  successor state exact or behaviorally bisimilar
G6  existing CPU/GPU ISA lowering; no oracle opcode
G7  physical bytes or measured latency reduced after every artifact and transfer is charged
G8  full-model scaling path and complete <=8 GiB ledger
G9  actual remote commit, raw evidence, hashes, and reproducible CI/run record
```

Small-model success remains development evidence, not TARGET-W success.

## 14. Forbidden loop after this handoff

Do not return to the following as the primary next result:

- another ALL-WEIGHTS hard family;
- another generic communication/counting lower bound;
- another parity or indexed-selector toy;
- a `state_update = 0` constructor;
- a custom Llama-like forward without official runtime verification;
- a synthetic checkpoint presented as public-model evidence;
- a hypothetical instruction that hides the same internal reads/MACs;
- another packed-arithmetic variant with no full-layer path;
- another project name or open-lemma declaration;
- a local reconstructed commit reported as remote progress.

## 15. Environment and infrastructure failure policy

An unavailable package, GPU, network path, or local shell is an infrastructure result, not a scientific result. It must not reject a mechanism. It also must not end the entire work session when repository or CI tools are available.

Required fallback order:

1. use the connected GitHub connector for repository reads and writes;
2. create a real branch from a verified remote SHA;
3. commit the environment harness, pinned workflow, directive, and state updates;
4. use GitHub Actions for CPU/dependency/reference checks that fit hosted runners;
5. record target-GPU-only metrics as `NOT TESTED` without fabricating them;
6. continue every constructive task that does not require the unavailable resource;
7. finish with a remote commit and an executable next gate, not an uncommitted environment report.

Do not require a local `git fetch`, local `gh`, or a local DNS path when the connected GitHub writer can perform the equivalent repository operation.

## 16. Immediate next experiment

The next experiment is constructive, not a new lower-bound sweep:

1. freeze one actual public development checkpoint with revision/config/file hashes;
2. execute the official Hugging Face loader and reference path in a pinned GitHub Actions job or another real environment;
3. record a deterministic reference transition trace;
4. audit actual checkpoint tensors for an automatically compilable, lossless structure not already closed by the ledgers;
5. implement one checkpoint-specific compiler transformation using existing ISA operations;
6. replace at least one complete Transformer layer;
7. test consecutive token and successor-state transitions;
8. record physical artifact/byte/compute costs;
9. reject or promote the mechanism at the first decisive gate;
10. commit and push every result, including a negative result.

## 17. Research-completion truth

A research round may produce a useful negative result. It may not claim repository progress until the corresponding code, documents, raw summaries, and updated state are present in a verified remote commit. See `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md` for the fail-closed Git policy.
