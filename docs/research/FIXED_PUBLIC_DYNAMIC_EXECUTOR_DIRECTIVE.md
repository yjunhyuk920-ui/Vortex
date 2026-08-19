# Fixed-public dynamic executor directive

Use this document as the highest-priority research instruction after `AGENTS.md` and the repository commit mandate. It supersedes any earlier instruction that allowed an environment preflight failure to terminate the research round.

## Mission

Stop generating generic lower-bound families and static one-query toys. Construct an actual checkpoint-specific dynamic compiler/executor for a named public dense Transformer, preserving the declared reference contract and producing a nontrivial successor state.

The final project target remains unchanged:

```text
arbitrary public unmodified Hugging Face dense checkpoint, including 405B class
executor replacement only
no retraining, distillation, fine-tuning, LoRA, or quality laundering
one GPU, peak VRAM <= 8 GiB
batch size 1
same-machine native 4B-Q4-class p50/p95 latency target
all CPU RAM, SSD, PCIe, preprocessing, artifact, and synchronization costs charged
no remote/external accelerator
no dense fallback disguised as a core
```

## 1. Do not terminate at the environment gate

An unavailable local shell network, missing `transformers`, missing GPU, missing checkpoint cache, or missing local Git credential is not permission to stop.

When local execution is blocked:

1. use the connected GitHub connector for repository reads, branch creation, commits, PRs, and status checks;
2. commit pinned GitHub Actions workflows for CPU/reference validation that fit hosted runners;
3. commit dependency manifests, model-free unit tests, artifact schemas, compiler code, and fail-closed runners;
4. record target-GPU-only items as `NOT TESTED`;
5. continue every constructive step that does not require the missing resource;
6. finish with a verified remote commit and one executable next gate.

Do not require local `git fetch`, `gh`, DNS, or Gram MCP when an authenticated GitHub connector can perform the repository operation.

Do not fabricate a run. Do not turn honest non-execution into a no-change session.

## 2. Mandatory repository path

Repository:

```text
yjunhyuk920-ui/Vortex
```

Before research:

- verify the real remote base branch and SHA;
- read `AGENTS.md`;
- read `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md`;
- read `docs/research/VORTEX_RESEARCH_HANDOFF.md`;
- read all mandatory root ledgers and active experiment evidence;
- work on a real `research/*` branch;
- never push directly to `main` and never force-push.

Every meaningful result, including a rejection, must be committed and read back from the remote before reporting it.

## 3. Research scope for this phase

This phase attacks only the constructive arm.

Forbidden as the primary result:

- a new ALL-WEIGHTS hard family;
- a new generic communication/counting lower bound;
- checkpoint-entropy arguments;
- parity or indexed-selector toys;
- `state_update = 0` constructors;
- synthetic weights presented as public-model evidence;
- custom Llama-like forward as the only runtime validation;
- hypothetical instructions or oracle APIs;
- another packed arithmetic variant without a complete-layer path;
- another VORTEX name;
- `ONE_EXACT_OPEN_LEMMA_REMAINS` or an equivalent restatement of the whole mission;
- an environment report with no repository change.

The required object is:

```text
compile_checkpoint(reference_model) -> artifact
initialize_state(artifact, prompt) -> compiled_state
decode_step(artifact, compiled_state, input, rng_state)
    -> next_token, next_compiled_state, resource_trace
```

## 4. Reference transition and state obligation

Define the reference transition:

```text
(y_t, S*_{t+1}) = F_W,R(S*_t, u_t, r_t)
```

The new executor must produce:

```text
(y_hat_t, S_hat_{t+1}) = D(A(W), S_hat_t, u_t, r_t)
```

Required output:

```text
y_hat_t = y_t
```

Required state contract: one of

### STATE-EXACT

```text
S_hat_{t+1} = S*_{t+1}
```

### STATE-BISIMILAR

Define a relation `~` and prove/test:

```text
initialization preservation
transition preservation
output preservation
finite-precision compatibility
all legal future suffixes produce the same declared output behavior
```

A next-token lookup without a successor state is not a core candidate.

## 5. Freeze actual checkpoints

Freeze two named checkpoints:

### TARGET-W

The actual 405B-class public target, including repository ID, immutable revision, config hash, tokenizer hash, and tensor-index hash. If inaccessible in the current environment, retain `NOT TESTED`; do not synthesize missing weights.

### DEV-W

A named public small checkpoint with the closest feasible operator graph, normalization, attention, MLP, tokenizer, and dtype/reference ABI. Freeze its immutable revision and hashes.

DEV-W is for implementation and falsification. DEV-W success is not TARGET-W success.

## 6. Official runtime gate

The official development reference must use the real supported loader and model class, such as:

```python
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained(
    frozen_model_id_or_path,
    revision=frozen_revision,
    local_files_only=as_required,
    torch_dtype=frozen_dtype,
    attn_implementation=frozen_attention_backend,
)
```

No custom forward may be the sole reference.

If local dependencies cannot be installed, add and commit a pinned GitHub Actions workflow. The workflow must:

- pin Python, torch, transformers, safetensors, tokenizers, and pytest;
- verify import and official load path;
- record exact versions and environment;
- execute a deterministic reference test that fits the runner;
- upload raw summaries and hashes;
- fail closed when model access or expected evidence is missing.

The lack of a completed Actions run does not permit fabricated `PASS`, but the workflow and tests must still be committed.

## 7. Actual-checkpoint compiler search

Inspect real DEV-W tensors and graph before choosing a mechanism. Reuse existing audits and do not reopen closed families without invalidating evidence.

Allowed constructive mechanisms include:

- lossless checkpoint-specific program synthesis;
- exact straight-line program synthesis;
- exact common-subexpression elimination;
- exact block dictionaries only when total artifact and decode cost improve;
- lossless coding fused with compute;
- bit-level exact circuits with explicit full-layer cost;
- equality saturation or superoptimization;
- persistent exact state;
- exact state quotient with a real bisimulation relation;
- adaptive cold probes with explicit addressing and page/byte cost;
- certificate-based early determination when every miss remains within the same executor and all verification cost is charged;
- existing-ISA software primitives with explicit lowering.

A mechanism must be generated automatically from the checkpoint. A manually written circuit for one selected tensor is not a PUBLIC-FIXED executor.

## 8. Cheapest decisive gate

Before a broad implementation, preregister:

```text
mechanism fingerprint
replaced reference operation
required assumptions
success threshold
rejection threshold
artifact-size equation
online byte equation
instruction equation
state equation
strongest adversarial input
what result permanently closes the mechanism class
```

Implement the smallest real-checkpoint boundary that can decide the mechanism.

When the first mechanism fails, commit its evidence, register the fingerprint, and move in the same round to a second mechanism whose primitive and dominant bottleneck are genuinely different. Do not rename or tune the failed mechanism.

## 9. Minimum real boundary

Toy scalar dots, one-bit functions, API wrappers, and zero-state lookups do not count.

The minimum implementation boundary is one of:

- a complete real DEV-W MLP block;
- a complete real DEV-W attention block;
- a complete real DEV-W Transformer layer;
- a complete DEV-W generation transition.

Promotion beyond a component prototype requires at least one complete Transformer layer and at least 128 consecutive decode transitions.

## 10. Exact dynamic validation

For each tested prompt and seed, record:

```text
reference token sequence
candidate token sequence
first token mismatch
reference successor-state digest
candidate successor-state digest or bisimulation witness
first state mismatch
128-step continuation result
```

Include:

- greedy decoding;
- frozen sampling algorithm and RNG tape where supported;
- normal prompts;
- code prompts;
- repeated-token prompts;
- long-context cases fitting DEV-W;
- small-margin/adversarial cases available without prompt cherry-picking.

Workloads and thresholds are frozen before results.

## 11. Existing-ISA requirement

Every new primitive must lower to actual current CPU/GPU operations. Report:

```text
load/store operations
bytes by memory level
integer instructions
floating instructions
shift/bitwise instructions
branches
address generation
dependency chain
synchronization
temporary memory
register/local-memory pressure when available
```

A function named `STATIC_LINEAR`, `MAGIC_MATMUL`, `MODEL_ORACLE`, or equivalent is outside the target unless its internals are fully lowered and charged.

## 12. Physical resource ledger

Keep separate:

```text
algorithmic operation count
compiled ISA instruction count
measured wall-clock latency
```

For each replaced boundary and the projected full model, record:

```text
reference checkpoint bytes
compiled artifact bytes
code/constants/metadata bytes
preprocessing time and peak memory
hot VRAM checkpoint-dependent bytes
KV/state bytes
temporary and allocator-reserve bytes
cold RAM/SSD bytes per token
PCIe bytes per token
VRAM traffic per token
decompression/addressing/verification work
CPU and GPU instruction classes
synchronization
TTFT
per-token latency
p50/p95
```

Do not call a cost removed when it moved into artifact expansion, preprocessing, CPU, SSD, PCIe, batching, cold start, another user, or a dense fallback.

## 13. Promotion gates

A core may be reported only at the highest gate actually satisfied:

```text
G1 ACTUAL-RUNTIME
   named public DEV-W loaded through the official runtime

G2 EXACT-TRANSITION
   tokens plus exact/bisimilar successor state verified

G3 REAL-BOUNDARY
   a complete real Transformer layer replaced

G4 CONTINUOUS-DECODE
   at least 128 consecutive transitions verified

G5 PHYSICAL-SAVING
   total physical bytes or measured latency reduced after every cost is charged

G6 FULL-MODEL-PATH
   explicit model-wide resource equation closes the required gap

G7 8-GiB-LEDGER
   checkpoint artifacts, caches, KV, runtime, temporaries, and allocator reserve all fit

G8 TARGET-HARDWARE
   actual same-machine 405B and 4B measurements satisfy p50/p95 gates

G9 REMOTE-REPRODUCIBILITY
   remote commits, raw evidence, hashes, workflow/run records, and independent commands exist
```

Report `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_Gn` only when all lower gates are satisfied.

A failed mechanism is reported as `REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_Gn` only when the mechanism was actually executed or decisively bounded at that gate. Infrastructure failure is not a mechanism rejection.

## 14. Infrastructure fallback deliverables

When an actual model or GPU cannot run in the current process, the round still must produce the maximum executable repository progress, such as:

- frozen checkpoint/revision contract;
- pinned official-runtime workflow;
- deterministic model-free compiler tests;
- artifact schema and versioning;
- resource trace schema;
- reference/candidate comparison harness;
- successor-state digest/bisimulation harness;
- hosted-run fail-closed gate;
- actual remote commit and PR update.

End with the first remaining action that truly requires the unavailable resource. Do not repeat the entire environment inventory as the main research result.

## 15. Commit and PR obligations

Follow `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md`.

At minimum:

1. commit research contract/handoff changes;
2. commit the official reference and CI harness;
3. commit implementation or decisive negative evidence;
4. update every canonical ledger whose truth changed;
5. verify the remote branch head SHA after each final commit;
6. create or update a PR;
7. report actual CI status;
8. leave no unreported change.

A failed experiment is still committed. A chat-only report is not repository progress.

## 16. Required final report

Begin with a Git evidence table:

| Field | Actual value |
|---|---|
| Repository | |
| Base branch/SHA | |
| Working branch | |
| Commit SHA(s) | |
| Remote SHA verified | PASS/FAIL |
| Pull request | |
| CI/checks | |
| Uncommitted remainder | |

Then report:

1. frozen TARGET-W and DEV-W;
2. official runtime execution status;
3. compiler and artifact format;
4. successor-state representation and invariant;
5. real boundary replaced;
6. consecutive transition results;
7. complete physical ledger;
8. first failed mechanism and genuinely different transition, if any;
9. highest gate passed;
10. exact `NOT TESTED` items.

Do not end with an open-lemma slogan or a request that the user choose a direction.

## 17. Immediate execution order

Execute now, without stopping at local-environment preflight:

1. verify the remote VORTEX base and writer permissions through the GitHub connector;
2. create/select the real research branch;
3. commit any missing policy and handoff updates;
4. freeze TARGET-W and DEV-W revisions and hashes obtainable from authoritative metadata;
5. add the pinned official Hugging Face reference workflow and tests;
6. implement the compiler/executor interfaces and resource-trace schema;
7. run every locally possible deterministic test;
8. use hosted CI for dependencies/reference paths that local DNS cannot provide;
9. select one actual-checkpoint mechanism by cheapest decisive gate;
10. implement it at a complete real Transformer boundary;
11. test dynamic successor state and continuous decode;
12. commit either survival or rejection evidence;
13. update canonical ledgers;
14. verify the remote head and PR/CI state;
15. report repository evidence before scientific interpretation.

The operational rule is:

```text
Do not discuss why the local environment cannot finish the whole mission and stop.
Use the available repository and CI control plane to move the constructive executor as far as evidence permits, commit it, and leave the next gate executable.
```
