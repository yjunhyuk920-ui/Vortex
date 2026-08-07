# Recent Failed and Demoted Approaches

Continuation of `FAILED_APPROACHES.md`. This is a permanent anti-repetition register. Revisit an entry only with a mechanism that directly addresses the recorded failure and a stronger preregistered falsification.

<!-- EXP-066-AUTHORITATIVE-FINAL -->
## F-034 — Exact classical TT/MPO bond-rank core

Frozen real-Q4 rank evidence gave favorable p50 operation/query fractions of 3.8941%/2.9984%, but p50/p90 static storage lower bounds were 11.0524%/22.9883%. The measured small-model extrapolation placed the 405B Q4 lower-bound representation at about 14.315 GB, above 8 GiB before runtime state.

Do not revive classical single-matrix TT/MPO by hiding factor storage, reporting only operation/query savings, or treating unresolved unit-boundary cuts as an exact reconstruction. Retain the MPO rank certifier as an auxiliary.

<!-- EXP-067-AUTHORITATIVE-FINAL -->
## F-035 — Exact joint Q/K/V row and common-right-factor reuse

Across 24 complete Q/K/V groups and 10,752 Q4 rows, exact reusable rows were zero. The common-right rank was 100% of input width at p50/p90; operation fractions remained 100%, while storage rose to 107.41%/114.12%.

Do not reopen with equality/sign/proportional-row variants, larger repeated blocks, or a joint kernel before a genuinely new shared circuit is proven. Retain the group certifier as an auxiliary.

<!-- EXP-068-AUTHORITATIVE-FINAL -->
## F-036 — Exact absolute-unread global demand certificates

Even after granting every preceding Transformer operation and weight read, the winning LM-head row, all metadata, and an independently optimal reveal order for every competitor for free, the output-head-only mandatory lower bound was 13.7697% at p50 and 19.2524% at p90. The p50 target failed before any real scheduler, propagation, fallback, or kernel cost.

Do not reopen with tile-size/order tuning or the same norm/absolute-unread bounds. Retain the bound auditor as an auxiliary.

<!-- EXP-069-AUTHORITATIVE-FINAL -->
## F-037 — Causal exact temporal-span replay

Exact dyadic modular-rank certificates covered 833 warm projection traces. Certified-independent arrivals alone required 100% of dense weight reads and operations at p50/p90; model p50 values were 69.244%/100%/100% for TinyStories-1M/3M/8M. No exact duplicate replay hit occurred, and the favorable basis cache was 391.97% of one Q4 projection-weight population at p50.

Do not reopen with approximate subspaces, numerical tolerances, post-selected longer traces, future/cross-prompt dictionaries, or uncharged coefficient/cache work. Retain the dyadic rank auditor as an auxiliary.

<!-- EXP-070-AUTHORITATIVE-FINAL -->
## F-038 — Exact Q4 short-block local-pattern table circuits

Across all 144 frozen real-Q4 dense projections and 3,024 preregistered width/order plans, the best single joint plan per matrix still required p50/p90 operation fractions of 88.4856%/91.4423%. Exact dictionaries, pattern IDs, offsets, row scales, and routing raised p50/p90 query and static representation fractions to 111.0294%/112.7907%. Even the most favorable matrix had a joint worst-axis fraction of 105.4244%.

Do not reopen by adding block widths or column orders after observation, reporting arithmetic without bytes, hiding dictionaries/IDs/routing/scales, using selected matrices, or approximately merging patterns while claiming exactness. Retain the block-pattern analyzer only as a conditional auxiliary for models with independently measured repetition.

<!-- EXP-071-AUTHORITATIVE-FINAL -->
## F-039 — Unqualified impossibility claims from online matrix-vector lower bounds

The Boolean/F2 reduction was exact in 1,052,740 exhaustive cases, but the strongest registered succinct theorem did not cover any Llama-405B tensor family under the full 8 GiB side-information allowance. For the largest valid square subproblem, 8 GiB is 1,024x above the theorem's `n^2/4` redundancy ceiling. Neither registered source supplies the required direct-sum theorem for 884 jointly preprocessed matrices, and all displayed bounds hide asymptotic constants.

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

Do not divide the 8 GiB state by tensor count, sum per-matrix asymptotic bounds, set hidden Omega constants to one and call the result certified, equate one cell probe with one GPU/PCIe/SSD transaction, or claim that all exact software executors are impossible. Retain the theorem/reduction auditor as a guardrail. This entry does not establish feasibility.

<!-- EXP-072A-AUTHORITATIVE-FINAL -->
## F-040 — Self-contained exact Q4 arithmetic DAG as universal hot core

For an exact self-contained artifact and fixed interpreter, standard-basis queries recover every Q4 coefficient. The artifact is therefore an injective encoding of the matrix. Finite validation found 272 unique signatures for 272 matrices with zero collision or control failure.

At the registered 405B count, worst-case Q4 information is `188.98828125 GiB`, while the complete hot allowance is 8 GiB (`4.2331%`). The information gap is `23.62353515625x` before scales, biases, opcodes, alignment, interpreter state, or workspace. Even the former 10% static threshold requires `18.898828125 GiB`.

Do not build pair-frequency CSE, beam/SAT/SMT synthesis, model-wide transcoding, or CUDA kernels as a universal self-contained hot core. Retain a bounded synthesizer only for explicitly restricted structured matrices. A future cold-backed design must introduce a materially different query-time information source and charge every original/lossless cold-data probe.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## F-041 — MTP-1 plus expert paging as a 1B-class surrogate core

For the registered Qwen3.5-122B-A10B structure, MTP-1 retained `10.0x` native
1B weight-equivalent traffic even after granting a zero-cost proposal and a
perfectly fixed routed-expert set. The p50 allowance was `1.2x`.

The failure is not repaired by fitting the 81 GB checkpoint on disk or paging
one layer at a time. Residency is not token latency, and the active 10B work
must still be reduced or amortized. Do not reopen this narrow path by omitting
draft/LM-head/verification/fallback cost, assuming expert identity across
tokens, relabeling block GEMM utilization as reduced arithmetic, or treating a
MoE result as dense-405B evidence.

Retain the deterministic block-accounting reference and exact longest-prefix
verifier. A materially different continuation must causally produce long exact
blocks and measure real router union locality under the fully charged Gate.

<!-- EXP-076-AUTHORITATIVE-FINAL -->
## F-042 -- Native MTP long blocks as a 1B-class surrogate core

The pinned unchanged Qwen3.5-0.8B native MTP reference selected `K=4` on the
registered build split. Held-out accepted-prefix p05/p50/p95 was `0/4/4`, while
shape-derived p05/p50 minima were `9/11`. Two of 18 evaluation cases accepted
no first proposal token, so the registered traffic accounting failed closed.
All causality, exact-acceptance, committed-cache, and rollback controls passed.

Do not reopen this branch by sweeping K after observation, selecting prompt
families, adding sampling, converting to quantization, testing another nearby
small Qwen size, tracing 35B routes, or building an expert pager. These changes
do not supply the missing accepted prefix and cannot turn Qwen-specific MTP
into arbitrary dense-405B evidence.

Retain only the causal proposal/verification/rollback reference as auxiliary
falsification machinery. A continuation must introduce a materially different
query-time information source and independently pass the final-fraction E0
Gate.

<!-- EXP-077A-AUTHORITATIVE-FINAL -->
## F-043 -- Activation-norm individual-channel Fractal MLP

The free non-deployable oracle saw every post-SiLU SwiGLU intermediate and
ranked channels by absolute activation times unchanged down-column norm. Even
with selector, full gate/up reads, score construction, and error measurement
charged free, the `9.988839%` path preserved only `71.5278%` of held-out target
top-1 decisions. Mean/p95 KL was `0.884161/3.080752`; every family failed. The
20% arm reached only `83.3333%` top-1.

Do not reopen by sweeping fractions after observation, selecting layers or
prompts, replacing individual channels with nearby blocks under the same score,
training a target-specific router, adding a sparse kernel, or moving directly
to 35B/122B. None supplies the missing logit contribution, and all real selector
costs would make the favorable ceiling worse.

This entry does not reject all dynamic sparse execution. Revisit only with a
materially different information source or exact/declared-quality correction
dependency and a new fully charged E0 route.
