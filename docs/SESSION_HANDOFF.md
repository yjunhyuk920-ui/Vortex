# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or model-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the physical runtime target is solved or impossible from metadata size alone.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, workflows, PR comments, and raw JSON

Before a user-facing progress/completion answer after meaningful work, update and commit the ledger and this handoff.

## Current repository state

Latest research decisions:

```text
PR #36  semantic-state program routing                    rejected
PR #37  prompt-compiled Hankel decision program           rejected
PR #38  perfect-oracle sparse repair                      rejected
PR #40  prompt-only nonlocal exact decision memory        rejected
PR #42  exact dense-operator information bound            accepted/merged
PR #44  metadata-aware direct/operator top-1 bound        accepted/merged
PR #46  end-to-end Llama final-token metadata bound       accepted/merged
```

Latest merge:

```text
PR #46 main merge: 038d3fa72dbfe91f4d9837d482b9f9c10719a00f
```

Authoritative Experiment 042 evidence:

```text
branch: research/llama-final-decision-routing-bound
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate workflow: 30784848053
Python 3.10/3.12 CI + validation: 30784848049
raw JSON: results/llama_final_decision_routing_bound.json
```

Earlier Experiment 042 workflow comments before the scope correction are not authoritative. Use the raw JSON above.

## Accepted Experiment 042 result

### Executable Llama-style family

The micro-model uses actual:

```text
token embeddings
RMSNorm
causal grouped-query attention
residual connections
SwiGLU MLP
final RMSNorm
linear LM head
```

A legal four-token prompt selects a variable layer/group, payload coordinate, output channel, and query carrier. Two GQA loader layers copy controls into the final query position. Two variable layers carry signed Q4 codes in `up_proj`. The fixed LM head decodes the selected code from the final next-token winner.

Executable certificate:

```text
hidden size: 8
attention heads / KV heads: 4 / 1
KV dimension: 2
total layers: 4
loader layers: 2
variable layers: 2
independent Q4 coefficients: 2
expected functions: 256
observed functions: 256
exact code recovery: true
minimum winner margin: 0.24951063086132308
```

Six primitive tests and full repository CI passed. The tests include nonzero causal GQA loading, exact inactivity of an unselected variable layer, two-layer additivity, final-token recovery, and target projection constants.

### Llama-405B-shaped projection

The projection charges the 1,024-dimensional GQA KV bottleneck and consumes loader layers:

```text
hidden size: 16,384
intermediate size: 53,248
total layers: 126
loader layers: 15
variable layers: 111
groups/layer: 31
neurons/group: 1,717
payload coordinates: 9,508
active intermediate neurons/layer: 53,227
control coordinates: 14,666
vocabulary rows: 42,139
independent signed-Q4 coefficients: 56,175,137,076
```

Complete exact final-decision metadata lower bound for the constructed family:

```text
224,700,548,304 bits
26.158586645498872 GiB
resident allowance: 8 GiB
excess: 18.158586645498872 GiB
```

Accepted theorem:

> A complete checkpoint-specific representation preserving every final next-token decision for the constructed arbitrary Q4 Llama-style family cannot fit entirely inside an 8 GiB resident checkpoint-information allowance.

## Critical scope boundary

The authoritative JSON explicitly records:

```text
actual Llama-style final-token routing: proven for the family
256/256 end-to-end functions: proven
complete exact decision metadata fits in 8 GiB resident: false
per-query external traffic lower bound: not proven
host-indexed random-access escape: not closed
fixed physical runtime target fully contradicted: false
released 405B checkpoint maximum complexity: not proven
real 405B execution: not performed
GPU wall clock: not measured
```

Metadata total size is not per-token traffic. In the Experiment 042 family one legal query selects one Q4 coefficient, so a host-resident indexed table could in principle return a sparse answer. Do not claim that 26.16 GiB must cross PCIe each token.

Current classification:

```text
all-resident arbitrary exact-decision metadata in 8 GiB: contradicted
host-indexed exact-decision representation: open
per-token cell-probe / communication lower bound: open
405B/8 GiB/4B-speed physical runtime: unsolved
```

## Accumulated execution evidence

```text
semantic-state program reuse: about 1 token
prompt recurrence: maximum 2 exact tokens
perfect-token repair: exact target on 68%–89% of tokens
future-aware prompt suffix oracle: 75 / 28 / 5 tokens
required full-stream amortization: about 247 tokens
exact-output Q4 information: 188.9883 GiB
end-to-end final-decision metadata family: 26.1586 GiB total
```

## Prohibited repeats and overclaims

Do not continue by only changing static rank, recurrence order, repair thresholds, ANN settings, speculative block length, or uncharged metadata. Do not:

- relabel metadata size as traffic;
- cite the invalid pre-correction Experiment 042 conclusion;
- claim a released checkpoint has worst-case complexity without measuring it;
- claim physical impossibility without a traffic/latency lower bound;
- claim runtime success without real hardware evidence.

## Current frontier — Experiment 043 Host-Indexed Exact-Decision Probe Gate

The remaining escape is an external checkpoint-specific decision index. Experiment 043 must determine what exact autoregressive execution requires from that index.

### First proof target

Construct an adaptive pointer-chasing decision family:

```text
address_(t+1) = transition[address_t, value_t]
value_t       = host_table[address_t]
```

The next address is unavailable until the current exact value is returned. This prevents parallel look-ahead for the constructed trace and gives a serial cell-probe count.

Required distinctions:

```text
serial probes/token: algorithmic property
bytes/probe: representation property
host/PCIe latency: measured hardware property
4B-speed failure: not proven until the above are jointly charged
```

### Experiment 043 required work

1. Add `docs/EXPERIMENT_043_HOST_INDEXED_CELL_PROBE_GATE.md`.
2. Implement a deterministic adaptive Q4 decision table and exact pointer-chasing decoder.
3. Construct indistinguishable table pairs proving that skipping the current addressed cell can change the next token and every later address.
4. Verify serial adaptivity over multiple steps and reject prefetch strategies that do not know the current value.
5. Derive memory, address, probe, and transferred-byte equations.
6. Add a host-memory prototype using explicit random accesses and record functional probe counts separately from machine-specific timing.
7. If timing is measured in CI, label it nonrepresentative and do not project it to the target GPU.
8. State whether the Gate proves only one serial probe/token, a stronger number of probes, or no useful latency lower bound.
9. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.

## Correct communication

> Experiment 042 proved an end-to-end final-token metadata lower bound of 26.1586 GiB for a constructed Q4 Llama-style family and therefore closed an all-resident 8 GiB exact-decision representation for that family. It did not prove 26.1586 GiB of traffic per token. A sparse host-indexed representation remains open, so the physical 405B/8 GiB/4B-speed objective remains unsolved. Experiment 043 now tests unavoidable serial host probes and their charged communication path.
