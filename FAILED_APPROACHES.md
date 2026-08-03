# Failed and Demoted Approaches

This file is a permanent anti-repetition register. A rejected family may be revisited only when a new mechanism directly addresses the recorded failure and defines a stronger falsification test.

## F-001 — Static low-rank and generic factorization

Includes:

- global/semantic Kronecker;
- ordinary low-rank factorization;
- activation subspace caching;
- recurrent dictionaries;
- gauge dictionaries;
- functional skeletons.

Observed failure:

Storage sometimes fit a projected envelope, but real-model decision preservation failed or executed reads remained close to the full model stream.

Do not repeat as:

- a new rank schedule;
- a different block shape;
- a renamed basis;
- an uncharged residual stream.

Revisit condition:

A new exact or certified mechanism must show disjoint-trace operation replacement and explicitly charge residual/fallback traffic.

## F-002 — Progressive low precision as the primary path

Observed failure:

Q2/Q3 failed quality. Q4 retained useful teacher candidates in some tests but autonomous exact prefixes were negligible and full-stream amortization required more than one thousand accepted tokens in the target projection.

Do not repeat as:

- only changing quantization bits;
- speculative depth without charged target verification;
- quality claims based on top-k teacher containment.

## F-003 — Independent exact-neuron selection

Observed failure:

Optimistic neuron subsets produced at most two exact tokens while traffic rapidly exceeded the target envelope. Layer allocation did not recover useful continuation.

Do not repeat as:

- a new neuron score only;
- a new global percentage;
- uncharged teacher gradients or adjoints.

## F-004 — Deterministic signed residual refinement

Includes PR #31–#34.

Observed failure:

Signed cancellation was real, but worst-case sound residual bounds required roughly 90–98% refinement and hundreds of GiB/token in projections.

Do not repeat as:

- another partition size;
- another deterministic norm;
- another static residual code.

Allowed new direction:

EXP-047 tests a materially different assumption: time-uniform probabilistic finite-population bounds that exploit observed random-order cancellation. It must still charge selector cost and exact fallback, and must be rejected if useful confidence requires nearly all tiles.

## F-005 — Prompt-derived recurrent programs

Includes semantic-state routing, Hankel recurrence, and related prompt programs.

Observed failure:

Program reuse was approximately one token; autonomous exact prefixes were one or two tokens, far below the amortization requirement.

Do not repeat as:

- a higher recurrence order;
- more state clusters;
- a larger static router bank without a new causal invariant.

## F-006 — Sparse repair, even with impossible oracles

Observed failure:

A future-aware or perfect-token oracle still required repair on most tokens, yielding target projections of roughly 128–169 GiB/token and 552–726 GFLOP/token.

Implication:

Any causal detector weaker than the oracle cannot rescue the same repair architecture.

## F-007 — Prompt suffix memory and nonlocal response replay

Observed failure:

Corrected future-aware suffix reuse was far below the required amortization on diverse prompts.

Do not repeat as:

- a different ANN index;
- a different embedding distance;
- larger prompt-only memory without new behavior equivalence.

## F-008 — Raw exact-prefix decision graph

Observed failure:

On the bounded TinyLlama grammar, eight distinct paths produced 64 unique nodes for 64 records and held-out prompts missed at position zero.

Classification:

Auxiliary exact memoization only. Not a universal execution mechanism.

## F-009 — Future-aware exact suffix DAG as a complete runtime

Positive result:

Exact future suffixes compressed 64 records to 38 nodes.

Failure as a complete runtime:

The construction used complete future continuations; causal held-out start coverage was 0%.

Classification:

Accepted graph-body compression, rejected as a standalone unseen-prompt executor.

## F-010 — Metadata size relabeled as traffic

Failure:

A total metadata lower bound does not imply that the same number of bytes crosses RAM/PCIe per token.

Rule:

Always separate total representation size, logical bytes accessed, physical transaction bytes, and latency.

## F-011 — Probe count relabeled as latency

Failure:

One serial host probe per token can still be small. The explicit pointer experiment forced near-one miss per token but only about 4.86 logical bytes/token.

Rule:

Latency claims require target hardware measurement or a valid bandwidth/latency lower bound.

## F-012 — Small-model evidence promoted to 405B success

Failure:

TinyLlama and synthetic tests do not measure 8 GiB VRAM, 405B execution, PCIe, SSD, CUDA scheduling, TTFT, or tokens/second.

Rule:

Such results remain Phase B/C and at most E3 until the required later phases run.
