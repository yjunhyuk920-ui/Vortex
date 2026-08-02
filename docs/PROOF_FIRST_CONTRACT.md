# VORTEX proof-first contract

This document prevents a local prototype result from being promoted into a claim about the flagship target.

## Fixed target

VORTEX is complete only when an unmodified 405B-class Hugging Face dense model runs end-to-end with:

- peak GPU VRAM at or below 8 GiB;
- no user training, distillation, fine-tuning, or model-specific calibration;
- declared original-model quality/correctness mode preserved;
- p50 time/token at or below 1.2x a native 4B Q4 baseline on the same machine;
- p95 time/token at or below 1.5x that baseline;
- a one-command runtime interface.

The target is not reduced when an intermediate design fails.

## What went wrong before this contract

Previous work validated useful local properties, then described them too strongly:

- tiny-model correctness was treated as evidence of 405B scalability;
- same-trace replay was treated as evidence of unseen-prompt generalization;
- a component that avoided one weight read was treated as a sufficient model-wide architecture;
- 405B memory and traffic projection was performed after implementation instead of before it;
- phrases such as `core solution`, `works`, or `can reach the goal` were used before the final resource inequalities were satisfied.

Those promotions are prohibited from this point onward.

## Evidence levels

Every result must use exactly one of these labels.

### E0 — idea

A proposed mechanism with no executable evidence.

### E1 — local primitive

Executable unit behavior on synthetic or tiny inputs. It proves only the tested local property.

### E2 — real-model component

A component replaces the corresponding operation in a real pretrained model and is tested on disjoint build/evaluation traces.

### E3 — scaling trajectory

Measured bytes/token, compute/token, memory, quality, and fallback rates across at least three model sizes show a trajectory compatible with the 405B gate.

### E4 — flagship completion

All gates in `docs/VALIDATION_PROTOCOL.md` pass on a real 405B-class model and the same-machine 4B baseline.

Only E4 may be described as the target being achieved. E1 or E2 must never be described as proof that the full target is feasible.

## Architecture Gate 0 — prove the budget before building the backend

A proposed model-wide architecture may not become the main implementation path until it has a committed feasibility certificate containing the following quantities.

Let:

- `M_hot` be persistent GPU-resident runtime state;
- `M_kv` be KV state at the declared benchmark context;
- `M_work` be activations, temporary tensors, kernels, and allocator reserve;
- `M_repair` be the largest cold repair tile or streamed weight window;
- `B_hot` be normal-path bytes transferred per generated token;
- `B_cold` be bytes transferred by one cold model-weight stream;
- `A` be committed generated tokens amortized by one cold stream;
- `B_4B` be measured native 4B Q4 weight/storage traffic per generated token on the same machine;
- `C_hot` and `C_4B` be corresponding useful compute per token.

The architecture must satisfy, with measured or conservatively bounded values:

```text
M_hot + M_kv + M_work + M_repair <= 8 GiB
B_hot + B_cold / A <= 1.2 * B_4B
C_hot + C_repair / A <= 1.2 * C_4B
```

When transfer and compute overlap is claimed, the certificate must use a measured roofline model:

```text
time/token >= max(bytes/token / measured_bandwidth,
                  operations/token / measured_throughput)
```

A design that fails these inequalities is recorded as a rejected experiment. It is not extended merely because its unit tests pass.

## Architecture Gate 1 — unseen-trace generalization

Any learned, cached, compiled, or atlas-like runtime representation must be built and evaluated on disjoint data.

Required split:

- build prompts and continuations;
- unseen prompts from different task families;
- unseen continuations after the build boundary;
- adversarial or distribution-shift prompts;
- Korean and English;
- code, mathematics, structured output, tool calls, and long-form text.

Required measurements:

- exact-token or declared-quality agreement;
- representation growth per token;
- normal-path hit rate;
- cold streams per token;
- bytes and compute per token;
- peak host and device memory;
- wall-clock relative to exact streamed target and native 4B.

Replaying the same prompt or the same activation trace is only E1 evidence.

## Architecture Gate 2 — real operation replacement

Hook-based offline analysis is insufficient. The candidate operator must replace the real model operation during generation.

A gate result must include:

- end-to-end generated output;
- exact baseline output or quality baseline;
- actual peak VRAM;
- actual token latency;
- actual storage/host/GPU traffic where measurable;
- fallback and repair counts.

## Architecture Gate 3 — scaling ladder

A candidate advances only after passing the same executable protocol on:

1. 1B–3B;
2. 7B–8B;
3. 30B–34B;
4. 70B;
5. 405B.

Each stage must publish raw metrics and compare the observed scaling slope with the flagship inequalities. A later size must not be inferred solely from a tiny checkpoint.

## Required workflow for every new architecture

1. Write the mechanism and its exact correctness/quality contract.
2. Derive its 405B memory, traffic, and compute equations.
3. Insert conservative values and show the flagship inequalities can close.
4. Define a falsification test that can reject the mechanism quickly.
5. Implement only the minimum code needed for that test.
6. Run disjoint-trace, real-operation measurements.
7. Promote, revise, or reject the mechanism based on committed evidence.

Implementation starts only after steps 1–4 are committed.

## Communication rules

The following statements are forbidden before E4:

- `the target is possible`;
- `this is the final solution`;
- `the core problem is solved`;
- `405B will run at 4B speed`;
- any equivalent wording based only on synthetic, tiny-model, same-trace, theoretical FLOP, or projected-byte results.

Allowed wording must state the evidence level and exact scope, for example:

> E1: the operator reproduced one tiny-model projection on a replayed trace without a cold read. Unseen-trace and model-wide scaling remain unproven.

## Current status under this contract

The repository currently contains E1 primitives:

- streamed safetensors access;
- exact LM-head decision refinement;
- tiny-model Jacobi equivalence;
- an exact-on-span `OnlineAtlasLinear` replay path.

No current component has passed E2 on a real operation with disjoint traces, and no architecture has passed Architecture Gate 0 for the full 405B/8GiB/4B-speed target.

The next task is not to add more Atlas features. It is to produce the first model-wide feasibility certificate and a falsification experiment whose measured outputs directly populate the flagship inequalities.