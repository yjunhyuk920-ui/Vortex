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

## F-052 -- Exhaustive tables or Boolean absorption as the numerical source

Preprocessed finite-semiring MatVec is exact, but the known
`n^2/(epsilon log n)^2` query scheme pays
`n^(2 + epsilon log_2 K)` preprocessing. At `n=16,384`, reaching the bare
`84.375x` operation factor requires `epsilon >=0.656113324`; even granting
`K=16` to the complete arithmetic gives exponent `4.624453296` and roughly
`1.1496e11` table states in the relevant factor before stored output vectors.

The deterministic Boolean cell-probe upper bound also does not provide an
`84x` physical source. Against an already 64-bit-packed Boolean matrix its
finite `n=16,384`, `w=64` traffic factor is only about `1.414214x`. Its crucial
OR witness/zero-rectangle saturation returns one bit and does not reconstruct
signed counts, Q4/BF16 sums, or Transformer hidden vectors.

Do not reopen with a different table block width, free exponential
preprocessing, coefficient-operations-versus-word-probes accounting, or an OR
oracle relabeled as a numerical dot product. Revisit only if a new algebra
preserves the declared Transformer computation and supplies a fully charged
finite representation. This entry does not reject committed-prefix residual
coding. Authority: `docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`.

<!-- EXP-083B-AUTHORITATIVE-FINAL -->
## F-053 -- Legal one-page Causal Residual Atlas with a common spectral ball

EXP-083B used committed-prefix pairs only, two-pass causal MGS, a verified
`beta_W=1.326752041578861`, maximum residual-energy page selection, complete
BF16/native arithmetic enclosures, and a strict final RMSNorm/LM-row top-1
certificate. The first untouched row reached rank 16 and preserved the native
winner in the candidate, but its certificate was unresolved and exact fallback
fired. That single fallback violates the required `99.908521%` coverage and
24/24 zero-fallback Gate.

The frozen down radius was `23.4205200666`: unread common-spectral residual
`16.3298572850`, pair-image term `6.2376633562`, and the remaining charged
terms. Actual down error was `3.5125591929`, but a sound global ball still
cannot certify the output. The actual final-hidden separation was
`38.9079080403`, over `9.2686x` the ideal top-two row-margin radius limit even
with numerical rounding removed. All 19 controls and independent replay
passed; this is a scientific rejection, not an infrastructure failure.

Do not reopen with rank, page width, layer, position, prompt subset, another
residual score, spectral slack, tolerance, or tighter RMS implementation
constants. Those retain the same missing unread-contribution premise or cannot
repair the observed row-margin necessary condition. Revisit only with a
materially new causal information source that provides paid, lossless
directional/correlated information about the unread contribution and first
closes a fully charged E0 equation. Authority: `results/exp_083b`.

## F-054 -- Constructible post-Atlas decision-dual sources

A signed decision query has the exact decomposition
`v^T W x = v^T(WQ)a + (W^T P b)^T u + r^T W u`. Forward-only causal pairs
therefore remain the Atlas normal form. Adding exact dual images is new
information, but their prompt-dependent constructor is not free: one fresh
full-model dual build costs `1.5625%` over the registered 64-token service life,
above the complete `1.185185185%` target before all other costs.

The static alternative also fails early. Precomposing every vocabulary
direction through one last down projection requires `12.720703125 GiB` under
an unrealistically favorable two-byte exact grant and a full scan costs
`6.765979992%` of registered Q4 bytes. A few-directions plus row-norm screen
left all `248,319` competitors unresolved on the frozen EXP-083B first row,
even with the observed distance substituted post hoc.

Do not reopen with more forward pairs, a free transpose/backpropagation pass,
a full vocabulary scan, approximate MIPS under an exact contract, or an
uncharged proof trace. This closure does **not** prove that every exact
bilinear-query/MIPS data structure is impossible. Revisit only with a concrete
checkpoint-derived lossless source for the Bilinear Cross Residual `r^T W u`
whose finite E0 equation charges construction, representation, query work,
state, verification, misses, and fallback. Authority:
`docs/research/E0_POST_ATLAS_CAUSAL_INFORMATION_SOURCE_AUDIT.md`.

## F-055 -- Matrix-local separable linear residual codes

Replacing the small Atlas bases with arbitrary exact left/right linear
covering codes does not close the Bilinear Cross Residual under the declared
all-query coefficient-probe model. A code pair with dimensions `a,b` needs
`a*n + m*b - a*b` independent checkpoint-image bits and leaves a raw cross
support equal to the product of its two covering radii.

The finite witness `rate=3/10`, `radius=3/16` satisfies
`0.3 + H_2(3/16) < 1`. Even granting every one of the fixed `8 GiB` hot bits to
the binary reduction, weighted side-state accounting forces at least
`6,141,198,336` cross-coordinate probes, or `1.52104775688%` of dense. That is
`1.28338404487x` the entire p50 allowance before lookup, cached terms,
verification, fallback, KV state, or physical movement.

Do not reopen with a larger Atlas rank, a different linear code, a
nearest-codeword decoder, or separately optimized row/column covering radii.
The proof is scoped: it assumes matrix-local images and independently
selectable per-matrix query pairs. Revisit only with nonlinear/global shared
advice that pays cancellation and probes, a word-packed structure with full
physical accounting, or a causal query restriction established on unchanged
real-checkpoint traces. Authority:
`docs/research/E0_BILINEAR_CROSS_RESIDUAL_SEPARABLE_CODE_BOUND.md`.

## F-056 -- Free cross-matrix projection reuse and overstated direct sums

One global advice space may project to high dimension in every matrix block,
but those dimensions are not independently usable. In the systematic linear
model, a block-local query can use global advice without outside raw probes
only through `U intersect V_i`; every other-block component must be canceled.
For fixed outside support `T`, each charged coordinate buys at most one local
dimension. The exact diagonal counterexample `U={(x,x)}` has two full
projections but zero shortened dimension in either block.

The valid global span-to-cover consequence is only `6,407,133` coefficient
uses, `0.001586914271%` of dense and `746.8489x` below the available p50
allowance. Do not divide advice by matrix count without a theorem, sum
projection dimensions, ignore outside cancellation, or promote this weak
bound into a general impossibility claim.

This entry does not close a concrete globally mixed code, nonlinear or
data-dependent cell probes, or a causal restriction on real Transformer
queries. Reopening the global interface requires a full constructor/query
equation; the next permitted proof path is a leakage-free causal-query
restriction certificate. Authority:
`docs/research/E0_CROSS_MATRIX_ADVICE_LOCALITY.md`.

## F-057 -- Calibration-built full-factor Causal Bilinear Span Ledger

EXP-084A built a dimension-23 ledger from the first exact-rationally
independent residual outer products in the frozen six-prompt/four-position
build order. All 24 build rows were independent modulo each registered prime.
On the untouched population, the first five rows were all exact nonmembers;
each also raised the fixed-ledger field rank from 23 to 24. Zero hits and five
misses exceeded the four-fallback traffic allowance. All 114 controls and the
zero-forward independent verifier passed.

Do not reopen with another build subset, post-hoc basis, dimension, side rank,
prime, prompt, position, decision direction, tolerance, or modular-only hit.
Dimension 24 already fails the registered traffic equation, and changing the
population after observing five misses is not evidence.

The run stopped at held-out rank five, so this entry does not prove that the
best post-hoc 23-dimensional subspace misses five rows or that the complete
36-row population has rank 28. It also does not reject nonlinear, implicit, or
non-factor-scanned exact query data structures. Reopening the Bilinear Cross
Residual frontier requires a materially different construction with a fully
charged E0 equation, not a linear-ledger parameter sweep. Authority:
`results/exp_084a` and
`docs/research/EXPERIMENT_084A_CAUSAL_BILINEAR_RANK_GATE.md`.

## F-058 -- Trace-built query-adaptive exact linear code unions

Routing among separately built exact factor ledgers avoids scanning their
concatenation, but it does not amplify independent cached answers. Leaves with
total dimension `A` hit at most `A` members of a globally independent query
population. Under the registered build and selected-leaf equation, the best
direction ceiling is 87,958 at leaf dimension 1, only `0.43979%` of 20M
queries; full dense fallback instead demands `99.999992826%` coverage and
leaves traffic at `85.0039x` target.

Do not reopen this class with a perfect or trained router, more leaves,
different leaf dimensions, shared redundant bases, or a partition of the
EXP-084A build rows. All five frozen evaluation rows are outside even the full
24-row build span under all three primes, so every such partition has zero
stored hits.

This entry does not reject repeated-query caching, causal populations proven
to have low total union dimension, or an implicit nonlinear checkpoint-derived
data structure that answers without materialized trace basis directions. Such
a structure may reopen research only with a concrete constructor and complete
E0 equation. Authority:
`docs/research/E0_QUERY_ADAPTIVE_CODE_UNION_BOUND.md`.

## F-059 -- Exact-field nonlinear branching as a distinct bilinear source

For a bounded exact algebraic query program, some branch/probe path has a
full-dimensional open cell. Freezing that path gives one rational circuit that
agrees with `f_W(r,u)=r^T W u` on an open set and therefore identically.
Baur--Strassen then computes `gradient_r f_W=W u` with at most four times the
unit-cost arithmetic. The alleged nonlinear adaptive source is consequently
contained by the static arithmetic-DAG family archived in EXP-072B.

Do not reopen with algebraic decision trees, rational gates, or a fixed
checkpoint index whose addresses are locally constant on continuous query
cells. The current twin-width route also supplies no new class: its unordered
algorithm is F-049 Hamming predecessor traversal, while rectangle and grammar
evaluation are F-050 static circuits.

This is containment, not a finite impossibility result. It does not cover
native BF16/Q4 rounding, bitwise/floor/modular operations, discontinuous word
addresses, exhaustive finite domains, or general cell probes. Reopening now
requires one non-exhaustive finite-word constructor with a complete target
equation. Authority:
`docs/research/E0_IMPLICIT_NONLINEAR_BILINEAR_SOURCE_AUDIT.md`.

## F-060 -- Local finite-word rank-one answer tables and named shortcuts

A Block Rank-One Truth Table is a genuine finite-word discontinuous source:
each matrix block stores all two-sided GF(2) query answers and the current
left/right patterns address one physical word. It buys small query traffic
only with exponential preprocessing. The 64-bit 2.5% layout needs 25-by-26
blocks, a table 8.660768514174031e11 times the Q4 checkpoint, 78-bit
addresses, and 173,215 dense-equivalent builds per service token. The tighter
8/675 layout needs 37-by-37 blocks and a table 3.449500717947148e18 times the
checkpoint.

Do not reopen with a different local block aspect ratio, one-bit ideal payload
instead of physical word traffic, omitted table construction, uncharged
wide addresses, Q4/BF16 answers treated as one bit, Mailman/Four-Russians,
broadword packing that still scans all payload, a Boolean nonemptiness oracle
called a numerical sum, or finite rounding state called a free transition
table.

This entry rejects those constructions, not every globally nonlocal nonlinear
bounded-word data structure. The general rank-one cell-probe model remains
open because current linear/direct-sum bounds do not cover it under the global
8 GiB advice grant. Authority:
docs/research/E0_FINITE_WORD_DISCONTINUOUS_BILINEAR_SOURCE_AUDIT.md.

## F-061 -- Boolean zero rectangles and limited-independence theorem lifts

The Larsen--Williams all-zero-rectangle list is globally nonlocal and
nonlinear, but its answer is one Boolean nonemptiness bit. Its proof works
because every intersection with an all-zero rectangle contributes zero. A
summary that instead answers every exact numerical subrectangle without raw
probes must answer singleton rectangles, recover every cell, and store at
least the raw block information. Do not reopen the direct lift with parity,
counts, signed sums, bit planes, Q4, BF16, or native rounding named as though
they had the same hereditary property.

Korten--Pitassi--Impagliazzo 2025 also cannot be cited as a target lower bound.
The complete GF(2) rank-one query family has a three-query XOR dependency and
is at most pairwise independent; its premise `k>t*w+1` fails before numerical
substitution. A whole-MatVec lower bound must be divided by `n` to infer one
scalar-call lower bound, and even an ideal `n^2` premise then misses `n^2/40`
by a factor 409.6 at `n=16,384`. The 2026 Multiphase bound is dynamic and
polylogarithmic.

This entry rejects those lifts only. It does not reject a different adaptive,
globally coupled, bounded-word exact numerical structure under 8 GiB advice.
That general gap remains open. Authority:
`docs/research/E0_GLOBAL_NONLINEAR_RANK_ONE_FRONTIER.md`.

## F-062 -- Williams finite-semiring preprocessing graph as a 2.5% core

The published lookup graph genuinely preprocesses an arbitrary finite-semiring
matrix and obtains subquadratic one-vector steps. Its physical information
tradeoff is not free: with block length `b` and alphabet `K`, the favorable
direct graph reads `1/b` of semantic raw matrix bits but stores `K^b/b` times
that raw payload in output-pattern edges.

Do not reopen by quoting only `O(n^2/(epsilon log n)^2)`, treating one pointer
lookup as zero bytes, hiding node values in addresses, omitting the `K^b`
catalog, or taking `b>14` as though the displayed `epsilon<1` theorem and
finite work remained unchanged. One registered Boolean square at `b=14`
already needs `36.616085 GiB` of favorable edges. Model-wide one-bit semantics
need `55,292.57 GiB`; Q4/BF16 one-query payloads are `2.629552%` and
`10.518207%` of DFloat before omitted costs.

Do not call native BF16 or FP32 rounded accumulation a semiring. Explicit
three-term witnesses give different left- and right-associated results.
Also do not infer scalar rank-one or 32-token shared-probe bounds from the
paper's full one-vector MatVec theorem.

This entry rejects this graph and direct relabelings only. A scalar-specific
global adaptive numerical structure, a finite transition construction with a
fully charged representation, or a covering lower bound remains open.
Authority:
`docs/research/E0_FINITE_SEMIRING_PREPROCESSING_FRONTIER.md`.

## F-063 -- Naive global-advice division and CKL tile direct sum

Do not divide the global 8 GiB nonlinear advice by matrix count and insert the
quotient into a single-matrix rank-one lower bound. The exact XOR witness
stores one `N`-bit checksum for `m` independent `N`-bit matrices, yet that
checksum and the other matrices recover any selected matrix. Conditional
information can therefore be synergistic and need not add up to advice
entropy.

Do not sum separately selected per-tile worst-case queries. The registered
1,386-square split is invalid, outside the displayed `n^2/64` proof regime,
and even its hidden-constant-one diagnostic is 12,553.84x below the requested
bound. The finite pigeonhole lemma supplies only one separate hard query for
one under-described tile; even its illegal global sum is 9.829856x short.

This entry rejects the proof shortcut, not CKL's single-matrix theorem and not
all globally nonlinear exact data structures. Reopening requires a theorem
that jointly charges cross-matrix probes used to unlock shared advice, or a
concrete complete constructor. Authority:
`docs/research/E0_GLOBAL_ADVICE_SYNERGY_FRONTIER.md`.

## F-064 -- All-linear Fourier spanning used as a rank-one target bound

The Fourier-fiber inequality is valid for the complete all-linear query
family and correctly charges arbitrary nonlinear global advice plus adaptive
cross-block probes. Do not reuse its 26.2% registered witness after replacing
all characters with Transformer rank-one characters. The full-character span
of every advice fiber is the theorem's decisive premise.

For all registered matrices, one independently selectable rank-one tuple has
at most `2^39,254,528` descriptions and 32 tuples at most
`2^1,256,144,896`. The low-degree character-dimension method is already large
enough at 114,194,991 probes (`0.028137293%`), 42.12x below the registered p50
budget. Do not call this ceiling an upper-bound algorithm.

This closes query cardinality and full-Walsh spanning as the missing rank-one
proof. It leaves a lower bound exploiting the actual restricted character
matrix, or a concrete globally nonlinear scalar constructor, open. Authority:
`docs/research/E0_FOURIER_FIBER_DIRECT_SUM_FRONTIER.md`.

## F-065 -- Native-exact local reuse and low-VC/Pollard relabelings

Do not reopen exact rounding absorption, natural activation runs,
same-coordinate history caches, exact product dictionaries, or perfect
opposite-product pairing as the universal core. On the pinned real final-MLP
tensor, deliberately favorable elimination uppers range from `0.1674%` to
`5.6920%`, while the compute-only 405B/native-4B envelope requires
`98.8173%` before positive costs.

Do not rename the Anand--van den Brand--McCarty low-VC/Pollard constructor as
a new route around EXP-082A. Its Boolean mechanism is the already rejected
row-difference spanning tree. Its exact non-Boolean theorem carries a `T`
factor for unique column values. The pinned BF16 tensor has at least 653
values per column and 1,276 per row; even the ideal `d=1` orientation costs
at least `35.6027%` of dense work, versus a `1.1827%` allowance.

This entry is a real-checkpoint necessary-condition rejection, not a universal
impossibility result. It does not reject a genuinely global nonlinear exact
answer source. Authority:
`docs/research/E0_NATIVE_EXACT_SHORTCUT_FRONTIER.md` and
`results/e0_native_exact_shortcut_frontier`.
