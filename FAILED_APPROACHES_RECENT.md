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

<!-- EXP-078A-AUTHORITATIVE-FINAL -->
## F-044 -- Frozen exact-anchor tangent macroblock reuse

The complete causal anchor-conditioned SwiGLU operator avoided EXP-077A's
channel omission, but it did not remain valid for even one later token. Across
18 held-out cases and 126 reuse positions, top-1 agreement was `0%`; valid-prefix
p05/p50/p95 was `0/0/0`; mean/p95 KL was `14.898423/25.335417`. All six families
failed. The unchanged target control had zero mismatch across 192 decisions.

Direct materialization was already unfavorable: the 0.8B path required
`13,821/6,249` p50/p05 reuse tokens, and the nine-path 122B screen required
`115,299/26,354`. The measured first-token failure is therefore decisive before
sentinel, repair, rank approximation, physical construction, or cache-fallback
costs are charged.

Do not reopen with a longer trace, selected prompts/layers, rank sweeps around
the same frozen anchor, dense macro construction, or an uncharged sentinel. A
causally delta-updated operator is a different mechanism only if it supplies a
new cheap information source and fully charges delta construction, detection,
repair, fallback, and cold traffic.

<!-- EXP-079A-AUTHORITATIVE-FINAL -->
## F-045 -- Fixed DCT pilot plus correlated cold row-block proof balls

The candidate passed its logical byte Gate only because every non-down operator,
two mutually favorable oracle selectors, nonlinear propagation, and failed
fallback were free. At p50 it retained 23/256 row blocks per down projection,
yet held-out top-1 was `6/144 = 4.1667%`, mean/p95 KL was
`7.544862/12.616226`, and MLP relative-L2 p50/p95 was
`0.713266/0.750627`.

The independently optimized minimum sound L2 enclosure was not close to a token
proof: its p50/p95 radius was `48.663918x/57.778748x` the exact MLP-output
signal. Increasing to the p95 allowance reached only `14/144` top-1 and left a
`46.993979x` median radius.

Do not reopen with DCT/Hadamard/random basis swaps, nearby pilot ranks, row-block
sizes, layer/prompt selection, fraction sweeps, or an uncharged nonlinear
propagator. These do not supply the missing dense residual information. This
entry rejects only the registered fixed-pilot/block-zonotope interface, not all
cold-backed online data structures or every possible correlated proof system.

<!-- EXP-080A-AUTHORITATIVE-FINAL -->
## F-046 -- Standard recursive Strassen as a Hyperblock core

Even with exact future activations, a zero-cost proposal, one target sweep, and
the best classical leaf width granted for free, no registered block length met
the final traffic and arithmetic fractions together. At `K=16,384`, traffic was
only `0.006103516%` and the favorable workspace equation was `7.539063 GiB`,
but fully charged multiplication/addition/padding/accumulation work remained
`36.111580%` of dense execution. That is `30.469145x` above the p50 allowance
before kernel and numerical overhead.

The unit-constant `omega=2.371552` pass is non-constructive and supplies no
causal future activations. Do not reopen with larger `K`, nearby leaf/tile
sweeps, an uncharged tensor-core packing factor, or rejected Jacobi/draft
sources. Revisit only with an explicit exact algorithm that closes the measured
constant gap and a materially new causal block source. This entry rejects
standard recursive Strassen under the frozen interface, not all exact fast
rectangular multiplication.

<!-- EXP-081A-AUTHORITATIVE-FINAL -->
## F-047 -- Nonlinear lookup plus tiny exact residual syndrome code

The exact field mechanism was sound in 321/321 controls, and its favorable
metadata-complete target shape fit the pre-fallback operation, traffic, and
storage ceilings. The population premise failed: held-out exact residual-code
coverage was `8.681672%` instead of `99.75%`; every family was only
`6.7164%-10.7143%`; favorable corrected relative-L2 p50/p95 was
`0.351269/1.213828` instead of `0.01/0.05`.

All six projections produced the same `54/622` exact fraction, consistent with
common causal template-prefix reuse rather than a general low-dimensional
error code. Charging the `91.318328%` fallback population leaves logical
traffic/operations at `92.244986%/91.533212%` of dense.

Do not reopen with tree depth/leaf, residual-rank, field, layer, projection,
prompt, tolerance, or kernel sweeps. Those enlarge or optimize the same failed
information source. Revisit only if a new causal dependency predicts the
otherwise missing residual or if the proof target changes materially, for
example to an end-to-end exact token-decision certificate. This entry rejects
the frozen lookup/syndrome composition, not all conditional online MatVec data
structures.

## F-048 -- Output-head-only and source-free proof-carrying execution

EXP-068 already closes absolute-unread output-head certificates, and the whole
registered 405B head is only `0.520459999%` of non-embedding coefficients.
Random linear proofs make checking a supplied full trace cheap, but a local
dense proposer plus verifier remains `100.058120260%` operations and
`100.266838924%` traffic. An external prover is added compute, not a local
execution reduction.

Do not reopen with more verifier challenges, a different polynomial
commitment, GKR/SNARK packaging, or an uncharged claimed trace. Revisit only
when an independently specified causal local generator produces the exact
trace inside the final budget; that generator, not the proof, is the core
invention. Exact correlated output-head indexing remains auxiliary.

## F-049 -- Exact terminal-only row/column Hamming spanning trees

EXP-082A granted the cheaper of row and column trees and certified their cost
without constructing either one. On 21 pinned Q4 projections, even the very
favorable one-operation-per-different-32-value-block lower bound was
`1.562367394%`, exceeding the final `1.185185185%` allowance by `1.318247489x`.
Every omitted metadata, delta-value, activation, tree-build, traffic, and
fallback term is nonnegative. All 72 exact controls passed.

Do not reopen with block size, layer/family selection, a different MST
algorithm, quantizer, traversal order, sparse kernel, or isolated favorable
matrix. Those cannot reduce the certified whole-population lower bound below
itself. Revisit only with a mechanism that invalidates the terminal-only tree
model, such as useful synthetic intermediate vectors, and only after proving
novelty against EXP-053/054/072 and fully charging discovery, circuit edges,
intermediate state, traffic, storage, build, and fallback.

## F-050 -- Static synthetic-intermediate Hamming trees and linear DAGs

Adding coefficient vectors that are not original rows or columns does evade
EXP-082A's terminal-only lower bound, but it does not define a new execution
class. Every exact synthetic-tree edge is a linear straight-line recurrence,
and general sharing is already the archived EXP-072B synthetic arithmetic-DAG
class. Saving the same program to SSD changes residency only.

Do not reopen with a Steiner heuristic, alternative tree topology, more beam
search, addition-chain/CSE synthesis, compressed opcodes, a cold circuit file,
or free offline compilation. A static runtime still stores a lossless program,
executes and/or reads its used gates and deltas every query, and has no
registered operation/traffic closure below `1.185185185%`. Revisit only with a
genuinely Query-Adaptive Cold Source whose causal selector, page probes, misses,
execution, state, verification, and fallback are fully charged. Authority:
`docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

## F-051 -- Raw Q4 page selection without an omitted-contribution source

Reading a query-selected subset of raw checkpoint pages does not by itself
determine exact arbitrary `W*x`. If the executor has no checkpoint-derived
information about an unread coefficient, two matrices can agree on all read
pages and differ at that coefficient, producing different output for a causal
activation with a nonzero matching coordinate.

Even granting a free selector, decoder, hit execution, and build, mean dense
fallback must be below `1.185185185%`, so exact coverage must be at least
`98.814814815%`. The retained verifier raises it to `99.081653739%`, while the
selected information must amplify at least `84.375x` or `108.891389x`
respectively.

Do not reopen by renaming page IDs, changing page size, adding an uncharged
index, or assuming omitted coefficients are irrelevant. Revisit only with a
concrete causal checkpoint-derived code or certificate that determines the
omitted contribution and pays selection, decode, state, verification, misses,
fallback, and build. This rejects raw-page omission as a source, not all coded
query-adaptive data structures. Authority:
`docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`.
