# Recent Failed and Demoted Approaches

## Native full-rank pure-copy lift — local rejection

[Proof and fixed native controls](experiments/codex_native_fiber_20260909/REPORT.md): the all-nonzero full-rank
I+11^T BF16 family has a native collision set of size 256^(n-1). Arbitrary
bijective input/output encodings cannot turn it into an injective full-coordinate
copy. A binary full-rank proof-only corollary has the same obstruction.
Rejected premise: algebraic full rank implies native bijectivity. Not rejected:
global cross-wire state, nonbijective continuation encodings, different native
operators, or side-state lifts with their actual paid cost. Not a universal
impossibility proof, and no HF activation reachability is inferred.

## CODEX-GAUGE-TRANSPORT-20260909 — native schedule and rejection scope

Do not claim a native-exact dense column permutation from `W'=Pout W Pin^-1`
alone: original logical leaves, tree/order and rounding points must move too.
The declared balanced FP32 `[2^24,1,-2^24,0]` witness returns 0 originally and 1
in a naive permuted tree. A finite transported-schedule reference fixes that
specific defect but retains 100% original dense work.

Do not infer a universal cheap-transformed-gate impossibility from the raw
coordinatewise field automorphism lemma or summed individual output ranks.
An explicit native opaque-bit butterfly conjugation has a paid O(n log n)
gate decoder; the arbitrary native dense projection is still unconstructed.
This is a scope correction, not a rejection of all global encodings or an
admitted new core. [Evidence](experiments/codex_gauge_transport_20260909/REPORT.md).

Continuation of `FAILED_APPROACHES.md`. This is a permanent anti-repetition register. Revisit an entry only with a mechanism that directly addresses the recorded failure and a stronger preregistered falsification.

## IMPLICIT-PROGRAM-CARRIER-20260909 — routing metadata, Patricia program, and paid rank-normal nonlinearity

Do not reopen an address-only selector by claiming logical addresses are free.
The implemented exact GF2 alias router does reduce the `16,384` square
query-table plus row-XOR arithmetic to `5.3116608411%` of the favorable leaf+add
slots, but its checkpoint is exactly the logical-alias-to-pattern relation. The
current 64-bit descriptor realization needs `300.749874 GiB` across the 883
registered matrices, `6.398601x` the raw binary source, and cannot live inside
8 GiB. A new address carrier must actually beat that metadata/translation cost;
renaming descriptor/page/TLB state as hardware does not remove it. This is not a
general nonlinear-address lower bound.

Do not reopen the word-label Patricia synthesizer by counting only its parity
events. The exact arbitrary-GF2 program reaches a `2.3436546326%` favorable
edge+leaf event fraction, but the preregistered one-64-bit-label-per-edge square
realization reads `536,838,144` checkpoint label bits, `1.99987793x` the raw
binary matrix, before topology and row memberships. A genuinely succinct trie
must provide an arbitrary-worst-case encoding and paid decoder rather than assume
checkpoint pattern compressibility.

Do not claim that independently rank-normalizing dense operators solves encoded
state once the transformed nonlinearity is charged. Deterministic elimination
does construct `A W B=J_r` for every binary rectangular matrix and can reduce an
isolated full-rank square map to `1/n` effects. But for
`a=W1 x`, `b=W2 x`, `c=a AND b`, the maximally favorable encoded projections
force the exact product to evaluate `(W1 z1) AND (W2 z2)`, restoring both dense
maps. The literal frozen micrograph is `1.0000305171x` baseline. This rejects
that per-operator rank-normal implementation, not every globally co-designed
graph encoding.

Authority: `experiments/implicit_program_carrier_20260909/REPORT.md` and
`results/e0_implicit_program_carrier_gate/summary.json`.

## IMPLICIT-DIRECT-QUERY-20260909 — complete image frames, duplicate transitions, and product-safe gauges

Do not reopen the complete one-sided query-image dictionary by merely changing
block widths or atom layout. The implemented arbitrary-GF2 producer is exact and
has checkpoint-independent query-derived addresses, but the registered 10/11-bit
construction needs about `4,817.8189 GiB` transformed images and `4.6992 GiB`
worst simultaneous query payload, `8.4357x` the registered `8/675` line. Moving
to the first equal block width that meets `8/675` (`b=85`) makes the complete
image-table storage ratio about `4.55e23x` source. More generally, a complete
one-sided **linear** image frame with radius `t` can name at most
`sum_{j<=t} C(S,j)` right factors; the registered target-radius lower is about
`1e23x` source storage. This boundary does not cover arbitrary nonlinear cells.

Do not reopen exact per-weight caching as though ordered row accumulation were
free. The implemented value-class executor is exact for the declared finite-word
ordered-leaf ABI, but even one distinct weight per column retains
`16385/32768 = 50.0030518%` favorable scalar operations, high-distinctness
retains 100%, and row membership visits remain `mn`. A stronger online grouping
by identical `(accumulator,weight)` transition also has an explicit FP32-exact
integer adversary: first-column weights `1..m`, later weights `1`, query all
ones. Every row accumulator stays distinct, so all `mn` state transitions remain.
This rejects duplicate transition sharing, not every nonlinear state machine.

Do not claim an encoded-coordinate gauge sparsifies arbitrary dense maps while
leaving coordinatewise Hadamard/AND cheap for free. Every bijection of the
Boolean cube preserving meet/AND is a coordinate permutation. Even allowing
independent invertible linear gauges on the two inputs and output of a product
node, `C(x AND y)=A(x) AND B(y)` forces one aligned coordinate permutation.
Coordinate permutations preserve dense support. A richer transformed nonlinear
operator remains OPEN only if it is explicitly constructed and fully paid.

Authority:
`experiments/implicit_nonlinear_direct_query_20260909/REPORT.md` and
`results/e0_implicit_direct_query_gate_v3/summary.json`.

## DIRECT-GLOBAL-PRODUCER-20260909 — reconstructive code, exact-sum witness, and Gauss-Jordan producer

Do not reopen a global erasure/network code whose runtime first reconstructs the
arbitrary checkpoint and then executes the reference dense kernels. Giving the
complete 8 GiB to information-bearing advice can replace at most `17.020392%`
of the binary source, `4.255098%` of Q4, or `1.063775%` of BF16, while the
declared downstream dense arithmetic remains `100%`.

Do not treat a mathematical exact dot sum as though the remaining native
rounding witness were automatically small. In the declared balanced IEEE-FP32
RNE tree, `[2^24,b,-2^24,0,-b,0,0,0]` has exact sum zero and rounded output
`-b`. Aligned gadgets encode arbitrary `-popcount(W_row&v)` entirely in
rounding error, so a universal witness can inherit the direct MatVec problem.
This scope is the declared balanced tree, not every CUDA reduction ABI.

Do not reopen deterministic Gauss-Jordan/LU row-operation replay as a target
producer while hiding its program. The implemented exact arbitrary-GF2
producer reduces `W` to `R=EW`, computes `Rv`, and reverses the row XORs. On
the fixed width-16,384 adversary this implementation costs exactly
`16385/32768 = 50.0030518%` of dense scalar bit operations and the row-op
metadata lower alone is `447.97 MiB` for one matrix. This rejects the explicit
producer, not every linear circuit or GF(2) data structure.

A new global nonlinear route theorem now permits arbitrary cross-matrix cells
and fully adaptive addresses without dividing advice per matrix. Its registered
necessary floor is only 578,619 words, `0.193471%` of the favorable target word
allowance. Do **not** cite it as mission impossibility. Reopening the direct
producer frontier requires an explicit implicit nonlinear cell compiler/address/
decoder or native equivalent that beats both source reconstruction and quadratic
program fetch. Authority: `experiments/direct_global_producer_20260909/REPORT.md`.

## BOOLEAN-MATVEC-LIFT-20260909 — indirect exact parity from Boolean-semiring products

Do not reopen a succinct Boolean-semiring MatVec structure as the exact numerical
source by adding an unspecified parity/count decoder. In the declared black-box
model, compare `R0=1_T` with each one-deletion `Ri=1_(T\{i})`. A Boolean subset
query distinguishes the pair iff its intersection with `T` is exactly `{i}`;
one query isolates at most one opposite-parity witness. Exact deterministic
conversion therefore needs at least `|T|` complete Boolean products (`n` for
full support). All 37,067 distinct `<n` query sets at `n<=5` confirm the finite
control with zero false determination.

Do not reopen the same route by independently preprocessing rows and queries into
arbitrary nonlinear Boolean features and claiming one OR/AND product computes
parity. The complete `d`-bit GF(2) inner-product truth matrix has exact Boolean
rank `2^d-1`: every all-one rectangle has at most `2^(d-1)` entries, while the
matrix has `(2^d-1)2^(d-1)` ones, and one rectangle per nonzero row meets the
bound. At `d=216` the feature representation is exponentially beyond the source.

Scope: these entries reject the black-box Boolean lift and the one-shot Boolean
feature factorization, **not** every direct GF(2) or nonlinear adaptive
data structure. Revisit only with an explicit direct representation/address/
decoder that pays its own side data, raw probes and native finite-word lift.

## CAUSAL-GLOBAL-20260908 — literal dynamic response-column delta summary

The new standard-causal sign/delta compiler provides legal batch-1 traces whose
successive desired right factors may differ in every coordinate. Therefore a
summary that stores the current exact product `y=W s` and updates literally by

```text
y' = y + W(s' - s)
```

must aggregate every source column on a Hamming-distance-n transition. That is
the previously rejected response-column/time-axis transport family under a new
name, now with an explicit legal causal adversary rather than an abstract query
sequence. Do not reopen by assuming adjacent causal right factors are sparse or
by hiding the changed-column aggregation inside `Update`.

This does **not** reject all dynamic exact summaries. Revisit only with a finite
update law that aggregates arbitrary dense right-factor changes subdensely and
pays its representation/address/decoder/native-state costs.

Also do not promote randomized succinct Boolean-semiring MatVec as the missing
exact producer: it permits error and computes a different algebra. This is an
insufficient primitive, not a universal lower bound against all zero-error
finite-word data structures.

## OUTPUT-ENVELOPE-20260908 — fixed-packet identical-output broadcast

Scope: original SmolLM2-135M revision93efa2f097d58c2a74874c7e644dbc9b0cee75a2,
layers0/15/29 q/gate/up/down;packet256contiguous;8frozen synthetic BF16 inputs/matrix;
declared separateFP32 products/balanced reduction/BF16 output, not official HF ABI.
12matrices/96queries/101376outputs agree, but0of432packets share actual output words.
Thus0accepted_groups and0oracle_broadcastable_groups;original reads100%,extra
coefficient/FPwork0.78%-1.04%. Even exact packet range cannot broadcast unequal
outputs. Do not vary packet/seed/tensor or refine the same certificate as core.
Different heterogeneous output encodings and actual causal inputs are not excluded.
Report/constructor/raw evidence: experiments/output_envelope_20260908/.

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

## F-066 -- ΩROUNDLOCK with Atlas-plus-one-page prediction

ΩROUNDLOCK was a directly designed attempt to solve the accumulated-state
problem: BF16-equal proposed coordinates become exact singletons at each layer
and only unlocked rows are repaired. The oracle Gate grants the true target
words and selector for free. On the frozen EXP-083B row, the down projection
locks only `1/1,024` coordinates; post-residual and post-RMSNorm locations lock
`7/1,024` and `2/1,024`. No complete vector locks.

Do not reopen this predictor with a paid certificate, row-repair kernel, or a
different tolerance. Equality is tested on exact BF16 words and the free
oracle already leaves `99.9023%` dense-row repair at the earliest point. This
does not reject a materially different predictor or a globally coded repair
source. Authority: `docs/research/E0_NATIVE_EXACT_SHORTCUT_FRONTIER.md`.

## F-067 -- OMEGA-SPIKECUT direct masked-factor evaluation

Do not reopen sums of permutation blocks, block-diagonal rank-one pieces, or
spiky matrices as a direct exact scalar evaluator.  Let `L` count every
active row/column factor coordinate over all masked components. A finite
description count includes arbitrary sparse supports, block partitions,
global budget allocation, cross-matrix components, and ignored invalid cross
cells, then applies Warren's theorem to the degree-two coefficient
polynomials. The complete favorable `L=4,800,000,000` model-wide allowance
has representable log2 count at most `184,958,474,578.07`, short of the
registered sign-checkpoint space by `218,789,422,765.93` bits.

This closes only the direct static evaluator.  It does not cover a further
globally nonlinear adaptive data structure over the factors, and it is not a
general cell-probe lower bound.  Authority:
`docs/research/E0_SPIKY_POWER_RANK_ONE_FRONTIER.md`.

## F-069 -- OMEGA-NEARESTPAIR literal representative-answer table

Do not reopen nearest-codeword, nearest-rank-one-query, or arbitrary joint
representative routing with one materialized exact scalar per representative.
The favorable Gate repairs only the exact differing coefficient mask and
grants routing, representatives, addressing work, and every non-read cost
free. A greedy packing of the `r tensor all_ones` subfamily at `N=16,384`
still forces log2 table size `13,741.25484686` and a 13,742-bit address, while
the complete 8 GiB contains only `2^36` bits.

Reusable `Wq`/`p^T W` images of nonlinear codewords are contained by F-055:
their linear spans retain the codewords and cannot worsen covering radius.
This entry does not reject a nonliteral, nonseparable compressed decoder for
the representative answers. Authority:
`docs/research/E0_ADAPTIVE_CODEBOOK_TRAPDOOR_FRONTIER.md`.

## F-070 -- OMEGA-TRAPSHIFT arbitrary-checkpoint masking

Do not cite a fast sampled trapdoored matrix as a compiler for an arbitrary
public checkpoint. Over a field, `Wx=(W+R)x-Rx` is exact and the trapdoor makes
`Rx` fast, but no trapdoor is available for the arbitrary shifted matrix
`W+R`. Computing that term directly retains one complete dense product;
assuming an average-case solver merely renames the missing information source.

The real/finite-field constructions also do not establish native BF16/FP32
accumulation equivalence. This does not reject using a checkpoint that was
originally generated with and semantically defined by a trapdoor; that is not
the arbitrary unmodified-checkpoint objective. Authority:
`docs/research/E0_ADAPTIVE_CODEBOOK_TRAPDOOR_FRONTIER.md`.

## F-068 -- OMEGA-POWERFOLD entrywise-power relabeling

Do not treat an exact integer entrywise power of a low-rank root as a new
runtime information source.  The multinomial theorem expands it into
`binom(r+p-1,p)` ordinary rank-one terms.  Direct exact queries are therefore
the static low-rank family already closed under its committed scope; at the
registered square size only 96 expanded terms fit even before positive costs.

The 2026 EPMF result concerns factorization complexity and nonnegative
magnitudes.  It neither supplies arbitrary signed Transformer weights nor
native accumulation order.  This entry does not reject every entrywise
nonlinear representation.  Authority:
`docs/research/E0_SPIKY_POWER_RANK_ONE_FRONTIER.md`.

## F-071 -- OMEGA-ROWLOTTERY direct average-coordinate source

Do not use the common-row Fourier Gate as if it were the publication's
average-distance premise. Exact answers on a fraction `f` of GF(2) rows plus
baseline guesses on the rest already achieve average accuracy `1/2+f/2`.
The `8 GiB` one-bit fraction gives `58.510196237313%`, so the former inference
was too broad.

The resulting direct constructor still fails. Across all amplifier calls,
exact recovery of an arbitrary `n`-coordinate result needs at least `n`
independent encoded row forms. Directly evaluating them reads at least `n^2`
arbitrary coefficients. The registered one-bit payload is
`47.00244140625 GiB`, `5.875305x` the hot grant and one complete source sweep
if cold. The concentrated-row source also has dense, not near-linear, query
time.

This closes direct row lottery, not a succinct nonlinear cold-backed oracle.
Every preprocessing structure, repeated call, probe, decoder, verifier,
fallback, and native numerical lift remains mandatory. Authority:
`docs/research/E0_AVERAGE_ORACLE_AMPLIFIER_FRONTIER.md`.

## F-072 -- OMEGA-FUNCTIONALSPAN linear recovery-set relabeling

Do not reopen functional PIR, functional batch, or functional array codes as
a nonlinear cold oracle. Their encoded symbols and recovery equations are
linear forms, so they are the existing systematic-linear sparse-span and
covering-code model.

A complete binary enumeration of every resident subspace through ambient
dimension six found only the expected parity/covering tradeoff. For a `2 x 3`
matrix, one resident form still needs three of six raw coefficients in the
worst case; two resident forms need two. The witnesses store row parities and
read the requested subset or its complement. The favorable all-linear
`GF(16)` sphere-covering screen with the complete 8 GiB grant has a
`79.09389479630005%` radius root, but this is not a rank-one or native-float
lower bound and must not be cited as one.

This closes only the named linear code family. It does not cover arbitrary
nonlinear advice, adaptive checkpoint probes, or a nonliteral compressed
answer decoder. Authority:
`docs/research/E0_FUNCTIONAL_ARRAY_CODE_NOVELTY_GATE.md`.

## F-073 -- Small nonlinear advice fibers as an immediate rank-one shortcut

Do not infer a new decoder merely from allowing an arbitrary nonlinear advice
fiber and adaptive coordinate probes. Every one of the 65,535 nonempty fibers
of binary `2 x 2` matrices was checked exactly. The minimax rank-one depths by
fiber size are `0` for size one, `1` for sizes two through four, `2` for sizes
five through eight, and the full `4` probes for sizes nine through sixteen.
The power-of-two optima are ordinary affine parity-code fibers and match the
all-linear-query optima.

The `2 x 3` affine control exhausts 26,387 distinct cosets and gives depths
`0,1,1,2,2,3,6` at sizes `1,2,4,8,16,32,64`. This rejects only the claim that
the first nonlinear toy already exhibits a superior adaptive source. It is
not a high-dimensional nonlinear-fiber theorem and does not cover global
advice or native numerical queries. Authority:
`docs/research/E0_NONLINEAR_FIBER_DECISION_DEPTH_SCREEN.md`.

## F-074 -- Exact-cut, standalone lossless-streaming, and attention-gauge cores

Do not reopen a weighted graph cut oracle as a numerical answer source. The
three-cut identity for a bipartite rectangle is exact, but evaluating one cut
contains the original arbitrary quadratic subset aggregate. The transformation
renames the Bilinear Cross Residual; it does not implement it.

Fused register-level lossless decoding is retained as an auxiliary, not a
core. Even granting the 10.6-bit BF16 entropy figure to every registered
parameter and charging all compute as zero, a 32 GB/s sweep takes
`16.8046952448 s`, needs 841 exact tokens to reach 20 ms/token, and costs
`525.1467264 ms/token` at a 32-token block.

Do not reopen attention-only gauge or fusion as a whole-model solution. Even
deleting every Q/K/V/O coefficient in all 126 layers for free removes only
`17.709431388354932%`; `82.29056861164507%` remains against the
`1.18270517%` compute-only allowance. These ceilings do not reject a future
mechanism that couples attention, MLP, embeddings, and cold data globally.
Authority: `docs/research/E0_LOSSLESS_CUT_GAUGE_FRONTIER.md`.

## F-075 -- Literal functional tables and counting-only sparse dictionaries

Do not reopen a full bilinear answer table, a direct local rank-one truth
table, or a standard coordinate-LDC label as an exact source.  The frozen 884
matrix independently rank-one tuple family has
`floor(log2 |Q|)=39,254,527`; one literal answer bit per tuple is
unaddressable.  A favorable square truth table needs `b=10` merely to reach a
1% coefficient-query fraction, but stores 1,046,529 forms per 100 source bits:
`10,465.29x`, or `480.3654 TiB` model-wide before native value widths.

A cold non-systematic linear dictionary is a more general and still open
interface.  With 405,849,243,648 binary source dimensions plus all 8 GiB as
redundancy, query-cardinality counting rejects only through 2,020,681 selected
bit forms, or 494,025 favorable 64-bit words.  High-girth capacity witnesses
already have enough distinct short sums at 2,216,796 forms or 510,961 words.
They do **not** align those sums with rank-one masks, provide a decoder, or lift
to native arithmetic.  Therefore cardinality is neither a constructor nor a
target lower bound.

Do not promote `OMEGA-FUNCDICT` until an explicit implicit-layout dictionary,
sparse exact rank-one decoder, native-order numerical lift, and complete
physical equation exist.  This entry rejects the table/LDC/counting-only
promotions; it does not reject every globally aligned non-systematic linear or
nonlinear cold source.  Authority:
`docs/research/E0_SPARSE_FUNCTIONAL_DICTIONARY_FRONTIER.md`.

## F-076 -- Fixed linear functional-dictionary decoders

Do not reopen invertible basis changes, fixed FFT/tensor inverse transforms, or
fixed linear syndrome representatives as the sparse functional decoder.  If
dictionary atoms are columns of `G` and a fixed linear decoder `H` satisfies
`G H=I`, then `H` has at least `D` nonzero rows.  Every such row is active with
probability at least `1/4` under an independently uniform binary rank-one tuple.

Across 32 independent tuples, the union activity is at least
`1-(3/4)^32 = 0.9998995475742793` in expectation.  Granting all 8 GiB as free
one-bit hot cells and perfect 64-bit packing still forces a worst-case cold
union of `41,874,345,776` bytes.  This is `3.03866665x` the complete registered
block allowance and costs `40.8929 ms/token` at a favorable 32 GB/s before all
other work.

This closes only a fixed linear representative under the independently
selectable 32-query tuple interface.  It does not cover query-dependent
nonlinear minimum-weight syndrome decoding, arbitrary nonlinear hot advice, or
a proved causal restriction on reachable tuples.  Authority:
`docs/research/E0_FIXED_LINEAR_FUNCTIONAL_DECODER_ACTIVITY_BOUND.md`.

## F-077 -- Extension-field rank called a physical probe count

Do not promote a rank-saturating q-system by identifying coefficient rank
`rho` with the number of stored cells read. In the exact characterization, a
rank-one coefficient vector has the form `lambda_j=gamma*mu_j`; the binary
support of `mu` may include every q-system basis coordinate.

Directly materializing one extension-field checkpoint summary per basis vector
of the published full-ambient `rho=1` construction expands the registered
one-bit source to `748.5509965564124 TiB`, or `16,223.57143848x`. Restricting
the system to the actual rank-one query subset removes that expansion but
becomes the raw one-bit source. Across 32 independent directions, even after
the complete 8 GiB hot grant, its expected cold union is `42,141,220,855`
bytes and costs `41.153536 ms/token` at 32 GB/s before all other work.

An exhaustive `2 x 2` search and explicit `2 x 3`/`3 x 3` witnesses confirm
that nonlinear coset leaders can shorten one tiny query, so F-076 must not be
misstated as a nonlinear theorem. Their canonical 32-query unions use at
least `99.4%` of the tiny dictionaries. This is not an asymptotic lower bound.
The remaining interface is a joint nonlinear batch decoder that finds one
small physical atom set spanning every query in the batch. Authority:
`docs/research/E0_EXTENSION_FIELD_RANK_SATURATING_FRONTIER.md`.

## F-078 -- Literal joint-factor catalogs and Segre counting as a solution

Do not cache one exact restriction for every pair of left/right batch factor
spaces. Every 32-query rank-one batch does lie in the exact envelope
`R tensor U` of dimension at most 1,024, but one registered square has more
than `2^1,046,528` independently selectable pairs of 32-dimensional factor
spaces. A literal envelope catalog is therefore more extreme than the already
rejected answer table.

Also do not promote rank-one intersection counting into a target lower bound.
Schaathun's product-code weight hierarchy gives the exact maximum Segre
intersection and matches exhaustive `2 x 2`, `2 x 3`, and `2 x 4` controls.
Even after using that exact structure and granting all 474,568,720,384 global
binary atoms to one 16,384 square, the proof-safe 32-batch count forces only
20,218 selected bit atoms, or 316 perfectly packed 64-bit words. This is far
below the physical target and rejects counting as the resolving method.

The remaining interface is an implicit near-source-size generator that
materializes the requested `R tensor U` restriction on demand. No such layout,
native equation, or decoder exists. Authority:
`docs/research/E0_JOINT_BATCH_COSET_GEOMETRY.md`.

## F-079 -- Linearized-polynomial renaming as a local evaluator

Do not promote the identity

```text
L_W(x) = sum_i a_i x^(2^i)
```

into an I/O shortcut. For an `n x n` binary matrix, the representation has
`n` coefficients in `GF(2^n)`, hence exactly `n^2` base bits. Published fast
linearized-polynomial operations consume this explicit list. Even one
model-wide bit plane swept once for 32 queries costs `49.542144 ms/token` at
32 GB/s with every other cost free.

Do not import a dense ordinary-polynomial evaluation data structure either.
At `n=16,384`, the ordinary degree is `2^16,383`, so its dense coefficient
parameter is exponential rather than the `n` nonzero linearized coefficients.
A new specialized local evaluator would be exactly the original arbitrary
preprocessed binary MatVec gap because the evaluation map is bijective.

This closes the representation renaming, not all nonlinear/adaptive MatVec
data structures. Authority:
`docs/research/E0_LINEARIZED_POLYNOMIAL_LOCALITY_GATE.md`.

## F-080 -- Cartesian bilinear scalars counted as causal tokens

Do not divide a full checkpoint sweep by every entry of `R^T W U`. With 32
left directions and 32 forward states the table does contain 1,024 exact
scalars, but only the 32 columns of `U` are distinct model states. Changing a
left direction asks another decision or competitor question about the same
state; it does not create another KV-bearing successor.

Even granting 10.6 bits per BF16 weight and making decompression, GEMM, KV,
metadata, and scheduling free, one registered sweep is
`16.8046952448 seconds` at 32 GB/s. The valid 32-state denominator is
`525.1467264 ms/token`, while the invalid 1,024-scalar denominator would
appear to pass at `16.4108352 ms/scalar`. At least 841 causally usable tokens
per sweep are required before positive costs.

Retain Cartesian batching for shared scalar certification, but do not reopen
it as token amplification without additional distinct forward states and a
causal path-coverage proof. The result does not reject selective tile reads
or a partial exact information source. Authority:
`docs/research/E0_CARTESIAN_BILINEAR_CAUSAL_UTILITY_GATE.md`.

## F-081 -- Nonlinear stored bits called a new fixed-XOR tabulation source

Do not promote arbitrary nonlinear checkpoint preprocessing when every query
reads a fixed nonadaptive cell set and XORs the returned bits. Expand each
stored bit in its unique algebraic normal form. Exact equality to the query's
linear parity for every checkpoint forces all selected constants and
higher-degree monomials to cancel; the selected degree-one coefficient masks
XOR exactly to the query. Replacing the cells by those linear forms preserves
every recovery set. This is the sparse functional dictionary, not a new
nonlinear mechanism.

Do not invert this scoped lemma into a universal rejection. With the complete
8 GiB advice granted proportionally and every omitted cost free, a `23 x 23`
binary block has 619 atom slots and enough subset names to stop rejecting at
weight six (`6/529`). That capacity is not an aligned code. Uniform random
atoms hit a fixed nonzero rank-one query with union-bound floor exponent
`-483`; the expected total nonzero rank-one hits have floor exponent `-437`.

The unresolved object is an explicit Segre-aligned sparse atom family plus a
succinct decomposer, shared physical 32-query set, and native bounded-word
lift. Adaptive data-dependent addresses and arbitrary word decoders remain
outside F-081. Authority:
`docs/research/E0_TABULATION_SUPERCODE_FRONTIER.md`.

## F-082 -- Raw sparse-subset capacity called a Segre cover

Do not promote `sum(k<=t,C(S,k)) >= |Segre|` into an atom construction. Any
actual cover makes the Krawtchouk transform of the complete Hamming ball
approximate the exact Fourier transform

```text
Rhat_r = 2^(a+b-r) - 2^a - 2^b + 2.
```

For `23 x 23`, `S=619`, and `t=6`, this forces every nonzero rank-one dual
codeword into weights `[30,46]` or `[574,590]`, so its character sum has
magnitude at least 527. The atoms must span all 529 dimensions; after all 90
extra copies are concentrated optimally, at most 8,809 ordered pairs are
equal. Unequal atom differences correlate by strictly less than one half.
Consequently the squared-character average is at most
`195,984.95537375606...`, contradicting the required `527^2=277,729`.

This Gate rejects the first nine exact subset-slack-ordered rectangles in a
`1..128` scan, not every shape. The first unclosed case is `13 x 89` with
1,353 atoms and radius 13. Any cover there needs a kernel relation of weight
at most 39; no such aligned layout or decoder is supplied. Adaptive word
decoders and native numerical lifting remain outside F-082. Authority:
`docs/research/E0_SEGRE_SPARSE_COVER_FOURIER_GATE.md`.

## F-083 -- Fourier slack survivor called a feasible sparse cover

Do not retain `13 x 89`, `S=1,353`, `t=13` merely because the pointwise
Fourier interval has slack. Freeze a nonzero left factor. The resulting
89-dimensional Segre ruling has `2^89` points, while its full preimage under
any surjective atom map has dimension only `196+89=285`.

An information-set projection proves that a `k`-dimensional binary subspace
contains at most `sum(i<=t,C(k,i))` ambient words of weight at most `t`.
Here that upper is `10,451,278,437,645,598,672,024`, only
`0.0000168849509765...` of `2^89`. Thus the prior frontier is impossible
without atom search.

Do not overclaim the theorem. The first case surviving both the restricted-
preimage sphere Gate and the Fourier Gate is `18 x 36`, `S=758`, `t=7`.
One large ruling necessarily induces a binary `[146,110]` code of covering
radius at most seven; its sphere density is `3.7123143518692814...`, so the
elementary bound does not reject it. Existence of that ordinary code is not a
simultaneous Segre construction. Adaptive word decoders and native lifting
remain outside F-083. Authority:
`docs/research/E0_SEGRE_RULING_PREIMAGE_SPHERE_GATE.md`.

## F-084 -- Rank-one subset capacity treated as closed under addition

Do not test only whether `Ball(S,t)` can name every rank-one matrix. If every
rank-one matrix has a weight-at-most-`t` representative, every rank-at-most-
`r` matrix has a weight-at-most-`rt` representative. The exact determinantal
count must fit inside `Ball(S,rt)` for every rank.

This immediately rejects the prior `17 x 43`, `S=855`, `t=8` frontier:
`Ball(855,16)` is only `0.01557457908009767...` of the rank-at-most-two
population. It also rejects the earlier `18 x 36` at rank two.

Do not repair a balanced case by adding shorter representatives. Sampling
`s` atom positions contains every weight-at-most-`t` representative with
probability at least `C(s,t)/C(S,t)`. The exact maximum Segre intersection of
an `s`-space therefore bounds all weights together. This rejects `32 x 39`
at ratio `0.9436909272421747...` and three neighboring balanced shapes.

The two Gates reject the first 770 capacity-ordered rectangles in the
side-`1..128` scan. `30 x 40`, `S=1,404`, `t=14` is the first unclosed case,
not a construction. Adaptive word decoders and native numerical lifting
remain outside F-084. Authority:
`docs/research/E0_DETERMINANTAL_RANK_AMPLIFICATION_GATE.md`.

## F-085 -- Rank-amplified representatives treated as pairwise disjoint

Do not retain `30 x 40`, `S=1,404`, `t=14` by charging the full `r*t`
support to every minimal decomposition.  For one rank-`r` matrix, admissible
rank-one terms form the binary point-hyperplane anti-flag graph.  Its exact
least eigenvalue, distinct-representative shell activity, and an
overlap-to-XOR-cancellation knapsack force four cancelled incidences at rank
22.

The resulting radius is 304, not 308.  Its Hamming ball is only
`0.05068730227558649...` of the rank-at-most-22 population.  The same Gate
rejects `31 x 39` at `0.5723204167945867...` and `30 x 44` at
`0.18141146098814276...`.

The first combined survivor moves to `31 x 43`, `S=1,559`, `t=15`, the 847th
capacity-ordered shape.  Its best cancellation ratio is
`3.409163670153519...`; this is not an atom construction.  Adaptive
word-valued probes, nonlinear output decoders, native arithmetic, and joint
causal batches remain outside F-085.  Authority:
`docs/research/E0_BIORTHOGONAL_DECOMPOSITION_CANCELLATION_GATE.md`.

## F-086 -- Treating the anti-flag overlap charge as one-shot

Do not retain `31 x 43`, `S=1,559`, `t=15` by applying the spectral overlap
bound only to a completed rank decomposition. Positivity of
`sum_c e(A_c)` identifies an adjacent pair whose representatives share a
coordinate. Peeling that pair cancels two incidences and leaves a rank-two-
smaller matrix governed by the same dictionary promise, so the charge
recurses.

At ranks 18, 20, and 22 this forces cancellation 2, 4, and 6. The resulting
rank-22 radius 324 has exact capacity ratio `0.4293893830152846...`, rejecting
the prior frontier. The combined scan closes 856 shapes. `31 x 42`,
`S=1,523`, `t=15` is first unclosed at best ratio
`38.10086306517199...`; it is not a construction. Adaptive word-valued
probes, nonlinear output decoders, native arithmetic, and joint causal
batches remain outside F-086. Authority:
`docs/research/E0_RECURSIVE_BIORTHOGONAL_CANCELLATION_GATE.md`.

## F-087 -- Treating adaptive addresses as new information in a linear field layout

Do not reopen extension-field linear summaries merely by choosing their
addresses from earlier returned values. On the zero source, all such values
are zero and select one definite support. Perturbations in the support's
common kernel reproduce the path; exactness therefore puts the query in the
same field span. A support of size `j` contains at most `2^j` binary
directions, yielding the exact necessity

```text
sum_(j=0)^t C(S,j)2^j >= 2^k.
```

For `k=16,384`, `S=19,172`, the minimum is 3,421 probes. The logical target
is 194; even granting every packed-Q4 lane to that one plane permits only 776.
Arbitrary block-local storage allocation does not help: all DFloat11 bits plus
8 GiB still give a relaxed minimum `10.988948%` against `4.740741%`.
Nonlinear stored cells, cross-block mixed cells, and native arithmetic remain
outside F-087. Authority:
`docs/research/E0_EXTENSION_FIELD_ADAPTIVE_SUPPORT_GATE.md`.

## F-088 -- Treating independent linear forms in one word as free locality

Do not reopen the linear-summary route by packing 64 unrelated summaries per
word. The zero-source path fixes its adaptive word support. A `j`-word span
has dimension at most `64j`, and exact Segre intersection bounds the number of
rank-one queries any support can answer.

At `31 x 42`, seven words reach only `0.19975212628...` of the required
population; the first count-feasible point is eight words and already uses
`64/651` of a favorable four-lane Q4 tile. None of 8,256 side-128 rectangles
reaches the registered line; the best is `118 x 128`, 22 words, `11/472`.
Nonlinear cells and global cross-matrix mixing remain outside F-088.
Authority: `docs/research/E0_ADAPTIVE_PACKED_LINEAR_WORD_GATE.md`.

## F-089 -- Assuming nonlinear cells erase determinantal capacity

Do not drop rank amplification when the stored cells or address path are
nonlinear. Every depth-`t` adaptive decoder is a cylinder polynomial of degree
at most `t`. Products of `r` rank-one characters give all rank-at-most-`r`
characters at degree at most `rt`, forcing

```text
RankLeq(a,b,r) <= sum_(j<=rt) C(S,j)(A-1)^j.
```

Arbitrary nonlinear bit cells at `31 x 42` still require 15 probes. Nonlinear
64-bit words become capacity-feasible, not constructed: the first side-128
traffic-line point is `25 x 108`, 50 words, two probes. Exact symbolic search
rejects the systematic `2 x 3` plus one arbitrary advice-bit seed. The fully
non-systematic seven-bit SMT timeout is inconclusive and must not be cited as
a rejection. Authority:
`docs/research/E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE.md`.

## F-090 -- Treating arbitrary nonlinear adaptive words as unrestricted query locality

F-089's smallest side-128 capacity survivor is now closed. For an arbitrary
binary source, choose a largest returned-word fiber at each of the first
`t-1` adaptive depths. A final route retains at least
`2^(D-(t-1)w)` sources, while all query answers on it factor through one final
`w`-bit word. If those query masks span dimension `r`, exact linear projection
fibers force `r<=tw`. Therefore every deterministic exact `S`-cell adaptive
router covers its linear query family by at most `S^t` subspaces of dimension
at most `tw`, even when every cell, address rule and final decoder is nonlinear.

Exact Segre intersection geometry gives coverage only
`7.45058081896844e-05` at the old `25x108`, 50-word, two-probe point. All 8,256
target-feasible side<=128 rectangles are closed. The first local unclosed area
is 5,400 (`24x225` and `25x216`,99 words,4 probes), but fixed four-word support
and a one-value-stage dispatcher are also rejected. Any surviving local route
must use multiple successive value-dependent routing stages.

Under the stronger independently-selectable 32-query service interface, even
that remaining multi-stage local route misses the target. Source-identification
forces some area-5,400 source to activate at least 84 of 99 words across the
complete rank-one family; a greedy 32-query tuple touches at least 32 distinct
words = 2,048 bits = `64/675`, exactly 8x the registered `8/675` line.

Do **not** generalize F-090 to global cross-matrix advice, native Q4/BF16/FP32
semantics, randomized error, or the complete VORTEX state machine. In
particular, the adversarial independently-selected 32-query tuple is not yet
proved reachable along one legal batch-1 Transformer continuation. Authority:
`experiments/nonlinear_router_frontier_20260908/REPORT.md`.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:3b9534800671e17eaef74869cf792186e683d6de -->
## Actual-checkpoint mechanism rejection evidence

- `cold_packed_projection_sequential_materialization`: G2=True, G3=True, G4=False; artifact=4208296 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; baseline p50/p95=105.321 ms/202.285 ms; candidate p50/p95=137.679 ms/234.948 ms.
- `checkpoint_mlp_torchinductor_existing_isa`: G2=False, G3=True, G4=False; artifact=169700 B; reference-layer=7080192 B; compiled-layer-resident=7080192 B; baseline p50/p95=102.057 ms/102.057 ms; candidate p50/p95=2268.660 ms/2268.660 ms.

## Fixed-public dynamic executor hosted result — commit `3b9534800671e17eaef74869cf792186e683d6de` / run `31982198413`

- Verdict: `REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `cold_packed_projection_sequential_materialization`: G2=True, G3=True, G4=False; artifact=4208296 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; baseline p50/p95=105.321 ms/202.285 ms; candidate p50/p95=137.679 ms/234.948 ms.
- `checkpoint_mlp_torchinductor_existing_isa`: G2=False, G3=True, G4=False; artifact=169700 B; reference-layer=7080192 B; compiled-layer-resident=7080192 B; baseline p50/p95=102.057 ms/102.057 ms; candidate p50/p95=2268.660 ms/2268.660 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- EXP086A_RESULT:START -->
## EXP-086A exact state-axis lifting microprogram Gate

- Decision: `REJECT_EXACT_STATE_AXIS_LIFTING_MICROPROGRAM_AS_CORE`
- Official target calls: `768`
- Decisive block size: `128`
- Favorable whole-model operation p50/p95: `20.239988251%` / `20.735470441%`
- Favorable whole-model weight p50/p95: `0.634804985%` / `0.634804985%`
- Joint p50/p95: `20.239988251%` / `20.735470441%`
- Reconstruction mismatches: `0`
- Deterministic core: `98fece89b13977c876bb7285e57529ab1dbb9709d2f786f67baf24336dd9a085`

This is a non-deployable perfect-future, exact-real favorable Gate. Native FP32 reduction order, causal drafting, complete-layer state, CUDA, 405B, 8 GiB, and target latency remain `NOT TESTED`.
<!-- EXP086A_RESULT:END -->

<!-- EXP087A_RESULT:START -->
## EXP-087A lossless entropy-stationary perfect-block Gate

- Decision: `CONDITIONAL_LOSSLESS_ENTROPY_STATIONARY_PATH_REQUIRES_TARGET_BASELINE_AND_TARGET_ENTROPY`
- Exact weight roundtrip mismatches: `0`
- Native block/sequential mismatches: `0`
- DEV-W zlib ratio: `1.261972x`
- DEV-W empirical Shannon ratio: `1.521866x`
- K=128 zlib traffic fraction: `0.619070913%`
- K=128 full-MLP exact-vector p50: `100.000000000%`
- Deterministic core: `26064d1b8648ea8048256fc629d5178f3669425bf21120513613b07411707b2f`

This Gate preserves exact weights on a small public checkpoint and uses perfect future activations. Target 405B entropy, target native 4B baseline, target effective compute, CUDA, 8 GiB, and latency remain `NOT TESTED`.
<!-- EXP087A_RESULT:END -->

<!-- EXP-088B:START -->
## EXP-088B — cross-layer page-mask program sharing

EXP-088B rejects the exhaustive eight-family cross-layer page-mask microprogram as a 405B core under its frozen scope. Reopening requires a richer exact program instruction that changes the information source, not a finer mask sweep.

Frozen fingerprint: two adjacent complete DEV-W layers; all linear weights tiled `64x64`; eight deterministic page families spanning both layers; all 256 masks; final pair hidden plus both layers' complete K/V; three build and three untouched holdout states; no selector and no runtime fallback credited.
<!-- EXP-088B:END -->

<!-- EXP-090A:START -->
## EXP-090A — causal suffix-action carry/secant/nearest draft

EXP-090A rejected the frozen one-layer/Q4-logit-lens suffix-action family as a 128-token core. The selected mode was `raw_q4`; untouched accepted lengths were `[0, 0, 0]`, and true-path rank p95/max was `15186.249999999996/41239`. This result already grants exact prefill history and free packed-kernel construction.

Do not reopen with history length, shallow depth, polynomial order, Q4 range/scales, prompt selection, block length, nearest metric, mode order, or threshold sweeps. Reopening requires a materially new causal suffix representation, sound exact token certificate, or independently generated bounded tree that changes the information source. This does not reject all speculative verification or every online residual state.
<!-- EXP-090A:END -->

<!-- EXP-091A:START -->
## EXP-091A — native exact Jacobi fixed-point decoding

EXP-091A rejects the frozen unchanged-target Jacobi family as a 405B core. The selected seed was `prompt_suffix_cycle_16`; untouched first-sweep exact releases were `[0, 0, 0]` against the no-compression/favorable requirements `85/67`. Best releases after up to eight full checkpoint sweeps were `[7, 7, 9]`.

Do not reopen with seed, n-gram order, cycle width, block length, iteration count, prompt, tie-rule, shallow-draft initialization, or threshold sweeps. Training a consistency model changes the checkpoint and is outside the mission. Reopening requires a materially new exact token information source that changes first-sweep certified release.
<!-- EXP-091A:END -->

<!-- EXP-092A:START -->
## EXP-092A — one-sweep linear block-span correction

EXP-092A rejects one-sweep linear block-span correction under the frozen scope. Even an oracle true block and free coefficients require more than eight new directions on untouched operator inputs.

Frozen fingerprint: K=128 EXP-091A selected seed; official guessed block before target; exact incremental-reference oracle; layers 0/15/29; qkv/o/gate-up/down input roles; exact dyadic integer encoding; deterministic 192-coordinate restriction; three modular rank lower bounds; eight extra static directions granted.
<!-- EXP-092A:END -->

<!-- EXP-093A:START -->
## EXP-093A — one-sweep static top-k branch source

EXP-093A rejects the frozen one-sweep static top-k logits as a bounded exact branch source. At least one required true token lies outside the registered static candidate width, so layout, verification, and kernel work cannot repair the source.

Frozen fingerprint: exact prompt prefix and boundary; `prompt_suffix_cycle_16`; one official BF16 K=128 guessed-context sweep; complete FP32 vocabulary logits; delayed official incremental target; stable descending-logit/ascending-token-ID ranks; build plus untouched holdout; p95 top-4 and worst-case top-16 thresholds.
<!-- EXP-093A:END -->

<!-- EXP-094A:START -->
## EXP-094A — one-layer Parareal residual transport

EXP-094A rejects the frozen one-sweep/one-layer Parareal residual transport as a 405B block source. The online position-aligned deep residual plus corrected-prefix coarse delta did not meet either the exact-chain or bounded-rank Gate.

Frozen fingerprint: one K=128 official fine guessed sweep; sequential official first layer plus rowwise-Q4 head; exact float64 `F(g)-G(g)` residual; sequential corrected-prefix `G(c)`; candidate before delayed target; build plus untouched holdout; exact-chain and p95-top4/max-top16 Gates.
<!-- EXP-094A:END -->

<!-- EXP-095A:START -->
## EXP-095A — online deep-residual affine hull

EXP-095A rejects the frozen online deep-residual affine hull as a 405B token source. Neither the causal eight-neighbor affine transport nor the full-bank target-residual L2 projection met the bounded-rank Gate under the frozen scope.

Frozen fingerprint: one K=128 fine guessed sweep; official first-layer/Q4 shallow states; float64 128-row residual bank; exact hidden-row reuse plus eight-neighbor affine ridge; target-seeing best-single residual; full residual-row-span L2 projection at `2^-40`; delayed incremental target; build plus untouched holdout.
<!-- EXP-095A:END -->

<!-- EXP-096A:START -->
## EXP-096A — complete online residual direct-margin LP

EXP-096A did not reject the residual-margin capacity source. No failed-family closure is registered until the causal coefficient Gate runs.

Frozen fingerprint: one K=128 fine guessed sweep; official one-layer/Q4 coarse path; complete float64 residual bank; target-seeing minimum-L1 FP64 coefficients; unique `2^-20` margin; HiGHS dual-simplex active set plus full fallback; complete-vocabulary `gamma_(2K+4)` roundoff certificate; cheapest-kill-first build and untouched holdout.
<!-- EXP-096A:END -->

<!-- EXP-100A:START -->
## EXP-100A — catalogued explicit rectangular FMM

No scientific failure is registered because the run was invalid. Repair only the recorded integrity fault and rerun the unchanged contract.

Frozen fingerprint: pinned AlphaTensor real-arithmetic catalog; integral
coefficients with absolute value at most two; six factor-space orientations;
depth at most 12; registered 405B projection population; explicit transform,
write, cold-byte, workspace, and role-p95 repair ledger; perfect `N/A=1` future
block grant.
<!-- EXP-100A:END -->

<!-- EXP-102A:START -->
## EXP-102A — real causal greedy draft block

The frozen real causal draft/verify source failed the raw A, N/A, latency, or exact-state Gate even before 405B scaling. Reopening requires a materially different causal source.

Frozen scope: pinned SmolLM2-360M target; same-family SmolLM2-135M and cross-family TinyStories-33M drafts; K `64,96`; build-only K selection; raw no-compression traffic threshold; exact terminal KV; all online work charged.
<!-- EXP-102A:END -->
