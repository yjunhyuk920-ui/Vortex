# Failed and Demoted Approaches

Permanent anti-repetition register. Revisit only with a mechanism that directly addresses the recorded failure and a stronger falsification.

## F-001 — Static low-rank/generic factorization

Failure: storage occasionally fit projections, but real decisions failed or reads remained close to full stream. Do not repeat by changing only rank/block/basis or hiding residual traffic.

## F-002 — Progressive low precision as primary path

Failure: Q2/Q3 quality failure; Q4 autonomous prefixes negligible; target verification amortization exceeded one thousand accepted tokens.

## F-003 — Independent exact-neuron selection

Failure: at most two exact tokens while traffic exceeded target.

## F-004 — Deterministic signed residual refinement

PR #31–#34 observed cancellation but required roughly 90–98% refinement and hundreds of GiB/token.

## F-005 — Prompt-derived recurrent programs

Failure: reuse near one token; exact autonomous prefixes one or two.

## F-006 — Sparse repair with impossible oracles

Failure: most tokens still repaired; projected roughly 128–169 GiB/token and 552–726 GFLOP/token.

## F-007 — Prompt suffix/nonlocal replay

Failure: future-aware reuse far below required amortization.

## F-008 — Raw exact-prefix graph

Failure: 64 records ->64 unique nodes; held-out first miss at step zero.

Classification: auxiliary exact memoization only.

## F-009 — Future-aware suffix DAG as complete runtime

Positive: 64->38 exact nodes.

Failure: future continuation required; causal held-out start coverage 0%.

Classification: auxiliary body compression.

## F-010 — Metadata size relabeled as traffic

False. Separate total representation, logical bytes, physical transactions, and latency.

## F-011 — Probe count relabeled as latency

False. Small serial probes can be cheap; hardware evidence or a valid lower bound is required.

## F-012 — Small-model evidence promoted to 405B success

Forbidden. Synthetic/small-checkpoint work does not measure target VRAM, 405B, PCIe, SSD, CUDA, TTFT, or tokens/second.

## F-013 — Global-range Serfling CPTC-v1 as primary executor

Authority: `results/exp_047/summary.json`.

Correctness passed at E1. Performance failed: 4/525 certificates, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python path about 8.6–9.1x full summation.

Decision: certificate/fallback auxiliary only.

## F-014 — Range-based CPTC oracle/stratified rescue

Authority: `results/exp_047r/summary.json`.

```text
C1 exact per-state range median 100%
C1 p90 100%
C2 median/p90 100%
C2 best 99.21875%
```

Decision: reject range-based CPTC; do not continue variance-only tuning.

## F-015 — Hard Jacobi target-only block decoding

Authority: `results/exp_048/summary.json`.

```text
p50 target passes / 32 exact tokens 58
p50 fraction 181.25%
p90 193.75%
maximum matching prefix 3
```

Do not repeat by changing only fill token, block length, or iteration cap while hiding every failed target pass.

## F-016 — Sequential partial-layer self-draft with target LM head

EXP-048 B3:

```text
18 cases, 54 variants
future information 0
maximum matching prefix 1
p50 committed tokens 1
minimum fraction 1333.463%
p90 2893.843%
```

Do not continue layer/temperature/tree tuning from this failed recursive draft source.

## F-017 — Target-only continuous Picard/Anderson fixed-point generation

Authority: `results/exp_049/summary.json`.

```text
18 cases, 1,458 trajectories
reference-selected p50 prefix 4.5
maximum prefix 6
p90 fraction 168.778596%
hard Jacobi p50 after four passes 4
Anderson p50 after four passes 1
```

Adversarial hidden chains:

```text
Picard prefixes 1,2,3,4
Anderson prefixes 1,2,3,3
hidden suffix transcript indistinguishability true
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Forbidden continuation: solver-hyperparameter-only tuning, soft residual convergence relabeled as exact token progress, or a target-only fixed-point GPU backend.

## F-018 — Fixed target-independent external drafting as universal or practical core

Authority:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
artifact 8852817664
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

Universal counterexample:

```text
fixed draft first token 7
arbitrary target first token 8
matching proposal prefix 0
exact correction committed
exact target output preserved
```

Therefore a target-independent fixed draft cannot guarantee a nonzero exact prefix for every arbitrary target.

Practical fixed TinyStories pool:

```text
3 targets
36 target/draft/prompt pairs
108 K rows
exact mismatches 0
target-future uses 0
matching prefix 0 in 72/108 rows
matching prefix 1 in 24/108
matching prefix 2 in 6/108
matching prefix 3 in 6/108
reference-selected p50 prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
Korean useful acceptance false
structured JSON useful acceptance false
target medians 1.0 / 0.0 / 0.5
```

PROJECTED actual 4B draft requirement:

```text
4/405 + 1/K <=0.01185185185
K >=507 exact proposal tokens before additional overhead
```

Permanent decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested pool is also rejected as a restricted practical core.

Forbidden continuation:

- expanding the same failed draft pool with a proposal tree;
- selecting drafts using target reference/future tokens while calling the selector deployable;
- ignoring one full draft forward per proposed token;
- reporting exact correction tokens as accepted draft tokens;
- claiming narrative/code successes cover Korean/JSON failures;
- replacing the fixed pool with another arbitrary small model without a pre-registered stronger rationale and universal claim restriction.

Allowed reuse:

- exact block verifier;
- external-draft accounting/reference tests;
- universal first-token counterexample;
- restricted-domain draft research only with an explicit non-universal claim and a materially different evidence base.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## F-019/F-020 — Tail exit and enumerative exact advice

F-019 rejects fixed/oracle layer-finalization tail skipping as core. F-020 rejects enumerative exact prefix/KV advice as core. Forbidden rescues include larger copies of the same table, hash-width changes presented as compression, replay presented as held-out generalization, and uncharged build/fallback amortization.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## F-021 — Structurally hashed bit-exact AIG as core

Bit-exact AIG compilation preserved all registered finite-domain decisions, but p50/p90 query work remained 84.17%/94.11% of the same unreduced exact bit-blast, dense-random p50 was 92.45%, and projected storage reached 255.60 TiB. Forbidden rescues are reporting only late-bit controls, relabeling raw bit blasting as compression, hiding circuit bytes, or extrapolating E1 synthetic exactness into a real Transformer claim.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## F-022 — Exact reduced ordered decision diagrams as core

Reduced diagrams were exact and avoided compile ceilings, but global p50/p90 paths were 35%/95%, dense-random node growth was 1.6873x per input bit, storage projection reached 202.25 TiB, and order-search amortization exceeded one million queries. Do not continue by trying only more variable orders, reporting late-bit controls alone, or hiding both-order compile cost.

<!-- EXP-055-AUTHORITATIVE-FINAL -->
## F-023 — Exact identical/sign-related column aggregation as universal core

Exact grouping preserved every registered decision and ideal repeated/sign-related controls fell below 10% logical work at n=64. General dense and forced-unique columns did not share that structure: global p50/p90 logical work was 62.5%/250%, query bytes were 63.64%/200%, and dense/unique p50 was 250%. Do not continue by reporting only repeated synthetic columns, hiding membership/popcount/vector-add work, or assuming real Transformer columns repeat without extraction evidence. Classification: auxiliary exact optimization conditioned on measured repetition.

<!-- EXP-056-AUTHORITATIVE-FINAL -->
## F-024 — Exact prototype plus sparse-residual dictionaries as universal core

The compiler exactly reconstructed all registered columns, and favorable repeated/sparsely perturbed controls improved with width. General dense and unique columns retained too many residuals: p50/p90 logical work was 62.5%/131.25%, query bytes 62.115%/169.643%, dense/unique p50 123.4375%, and 24 cases never beat baseline. Do not continue by adding synthetic prototype counts, hiding residual activation/index costs, or presenting favorable repeated matrices as arbitrary-model evidence. Further use requires measured real-checkpoint structure.

<!-- EXP-057-AUTHORITATIVE-FINAL -->
## F-025 — Exact column grouping/dictionaries on measured real checkpoint weights

No analyzed dense projection in three pinned TinyStories checkpoints contained even one exactly repeated or sign-related column under FP32, Q8, or Q4. Prototype residuals remained dense: Q4 median/p90 residual scalar density was 81.41%/84.28%; p50/p90 operations were 82.89%/85.84%; query bytes were 3.29x/4.91x baseline. Do not continue by increasing prototype search, quoting only the 70.29% best matrix, or treating Q4 structural results as model-output preservation. Retain the analyzers only for conditional measurement on future models.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## F-026 — Conventional exact low-rank factorization of measured real Q4 projections

All 144 pinned real-Q4 dense projections had certified rank `min(rows, columns)`. Conventional exact `W=A@B` therefore has favorable operation and factor-storage lower bounds of 200% before factor bitwidth, metadata, and kernel overhead. Do not revive this using approximate SVD energy, selected matrices, or a new factor optimizer while claiming exact output preservation. Retain modular rank certificates only as falsification infrastructure.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## F-027 — Exact shift-displacement structure on measured real Q4 projections

All 144 pinned real-Q4 dense projections retained full displacement rank under the most favorable of zero-fill/cyclic diagonal/anti-diagonal operators. Favorable query lower bounds were 100% at p50 and p90; generator storage was 200%. Do not continue by adding more visually chosen shifts, selecting a tensor subset, or ignoring transform/boundary costs. Retain displacement certificates only as structural falsification tools.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## F-028 — Static exact-zero sparse streaming on measured real Q4 projections

Real Q4 weights contained only 17.76% median exact zeros. Skipping them left 82.22% median work and required 150.93% median query bytes after exact run metadata. Even the best matrix remained at 69.90% work and 190.12% bytes. Do not revisit using more CSR/BSR block sizes, zero clustering, or index compression: scalar nonzero density itself is already above the 25% Gate. Retain exact sparse formats only as conditional auxiliaries.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## F-029 — Causal exact-zero activation-column skipping

No exact projection-input zero occurred in 56,448 calls or 17,529,344 observed input scalars across prefill and decode. Warm-decode work and query bytes slightly exceeded dense execution after mandatory zero discovery and metadata. Do not revisit with more zero scanners, module selectors, or near-zero thresholds: near-zero is approximate and exact zero population was empty. Retain the hook/accounting machinery only for architectures with explicit exact-zero activations.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## F-030 — Exact non-mask attention-probability zero skipping

After excluding causal and local-window structural masks, warm-decode exact-zero probability density was only 0.0305% in aggregate. Whole-model work and bytes exceeded dense execution after QK, softmax, discovery, metadata, and unchanged Linear costs. Do not revive by counting structural mask zeros, reporting the 7.16% maximum row alone, or using near-zero thresholds while claiming exactness. Retain the probability validator/accounting as an auxiliary.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## F-031 — Exact cached Key/Key-Value equivalence reuse

No exact K or KV duplicate occurred in 147,456 measured layer/head rows. Hashing and metadata increased bytes. Do not revive by using approximate similarity while claiming exactness, by counting repeated token IDs instead of vector bit patterns, or by omitting local-window eligibility and metadata. Retain the validator as auxiliary.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## F-032 — Exact output-row identity, sign reuse, and sparse-delta prototypes

No identical or sign-related dense output rows occurred. Only four projections admitted a dual-cost-beneficial sparse-delta plan, while population p50/p90 remained dense. Do not revive by ignoring per-row scales/biases, activation reads, residual indexes, or by selecting a plan that saves operations while increasing bytes. Retain the row compiler as an auxiliary exact dictionary tool.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## F-033 — Exact low Kronecker-rearrangement rank

Every selected real-Q4 dense rearrangement was full rank at its four-row cut. Even favorable 4-bit-factor accounting required at least 200.877% of dense operations and slightly more static storage. Do not revive by reporting query bytes alone, using one prime without witness verification, or treating a low field rank as an exact integer factor reconstruction. Retain the certifier as auxiliary.

## Continuation pointer after EXP-065

`FAILED_APPROACHES_RECENT.md` is the authoritative continuation for F-034
onward. As of EXP-081A it also closes the registered sparse-channel oracle,
frozen exact-anchor macro, fixed DCT/block-zonotope proof state, standard
recursive Strassen Hyperblock core, and nonlinear lookup/tiny residual-syndrome
path, in addition to the EXP-066--076 families. F-048 additionally closes an
output-head-only certificate and proof-carrying execution with no charged local
trace source; it does not close verification as an auxiliary.
F-049 closes exact terminal-only row/column Hamming spanning trees by a
certified favorable lower bound; it does not by itself close arbitrary
synthetic-intermediate linear circuits. F-050 then closes a static synthetic
tree/DAG as a *new* core because it is contained by archived EXP-072B and lacks
a fully charged target route; it explicitly leaves a genuinely query-adaptive
cold information source open. F-051 closes raw Q4 page omission with no coded
source for unread contributions; it explicitly leaves a fully charged Coded
Causal Cold Source open as an unsupported search class.
F-052 closes exhaustive finite-semiring tables and Boolean absorbing
cell-probe shortcuts as direct numerical Transformer sources. It does not
close the newly specified Causal Residual Atlas; that class must now pass a
real causal-residual certificate Gate.
F-053 records that Gate's first-row rejection and closes the legal one-page
common-spectral/global-L2-ball Causal Residual Atlas primary path. It does not
prove every possible correlated causal code impossible, but reopening requires
a materially new paid information source and E0 equation rather than an Atlas
parameter or bound sweep.
F-054 rejects forward-only continuation as Atlas normal form plus the two
currently constructible decision-dual sources: a raw-checkpoint dynamic dual
build at the registered 64-token lifetime and a static full-vocabulary dual
scan. It leaves only a concrete, lossless, sub-dense Bilinear Cross Residual
source open and does not claim a universal bilinear-query lower bound.
F-055 closes matrix-local separable linear covering codes for that residual:
even an all-binary, all-8-GiB, free-decoder coefficient-probe grant has a
`1.52104775688%` cross-work lower bound. It does not close nonlinear or
cross-matrix shared advice, word-packed probes, or causally restricted query
populations.
F-056 closes only *free* cross-matrix projection reuse: useful zero-outside
linear advice is the shortened block space, and every fixed outside probe buys
at most one local dimension. Its valid global lower bound is far below the
target, so it explicitly forbids a general impossibility claim and leaves a
fully specified global code or certified causal query restriction open.
F-057 closes the frozen calibration-built full-factor Causal Bilinear Span
Ledger: all 24 build rows were independent, and the first five held-out rows
were exact misses against the dimension-23 ledger. It forbids post-result
rank/build/prompt/prime sweeps, but does not claim held-out rank 28 or reject
nonlinear, implicit, or non-factor-scanned exact query codes.
F-058 closes query-adaptive unions of materialized trace-built exact linear
leaves as a primary core. A perfect router cannot exceed the independent
answer directions actually built, and any partition of the full EXP-084A
build span still misses all five frozen evaluation rows. It leaves repeated-
query caching and a genuinely implicit nonlinear checkpoint-derived source
logically open.
F-059 closes exact-field nonlinear branching as a distinct source: a fixed
open-cell path computes the same rational bilinear function, and
Baur--Strassen reduces it to static MatVec. It does not close finite-word
discontinuities, native rounding, bit operations, or general cell probes.
F-060 closes local finite-word rank-one truth tables and the audited Mailman,
broadword, Boolean-nonemptiness, and free-native-state shortcuts. It does not
close globally nonlocal nonlinear bounded-word data structures under the
global 8 GiB advice grant.
F-061 closes the direct exact-numerical lift of Larsen--Williams all-zero
rectangles, KPI25's limited-independence application to the full rank-one query
code, and whole-MatVec lower bounds used without the mandatory `n`-fold
scalarization. It explicitly leaves the general nonlinear numerical rank-one
constructor/lower-bound gap open.
F-062 closes Williams' finite-semiring preprocessing graph as a native
reference-exact 2.5% core. Its favorable lookup payload falls as `1/b`, but
its checkpoint-dependent catalog grows as `K^b/b`; native BF16/FP32 rounding
also violates the semiring premise. It does not close a different
scalar-specific adaptive numerical structure or a future covering lower bound.
F-063 closes naive division of global nonlinear advice and illegal summation
of single-matrix CKL worst cases. Exact XOR synergy shows why conditional
information cannot be allocated per matrix. It does not close a new theorem
that jointly charges the cross-matrix probes used to unlock that synergy.
F-064 retains a valid Fourier-fiber direct sum for all linear queries but
closes its promotion to rank-one by query cardinality alone. The restricted
rank-one character matrix lacks the full spanning premise; its structure or a
new constructor remains open.
F-067 closes the direct sum-of-spiky-matrices evaluator by a finite
resource-sensitive sign-pattern count that includes sparse factor supports.
It does not close a further nonlinear adaptive index over those factors.
F-068 closes integer entrywise powers of low-rank roots as a distinct runtime
source because exact multinomial expansion is the existing static low-rank
normal form. It does not close every entrywise nonlinear representation.
F-069 closes a literal cached-answer table with nonlinear nearest-query
routing: a finite fixed-side packing forces a 13,742-bit address before native
or system costs. It does not close a succinct nonlinear answer decoder.
F-070 closes trapdoored additive masking as a source: the mask product is fast
but the arbitrary shifted-checkpoint product remains dense. It does not close
checkpoints originally sampled together with a valid trapdoor.
F-085 closes the former `30 x 40` fixed-linear sparse-cover frontier.  The
anti-flag spectrum forces support collisions inside a minimal biorthogonal
decomposition; distinct-representative activity strengthens this to four XOR
cancellations and an exact rank-22 capacity contradiction.  It also rejects
`31 x 39` and `30 x 44`, but it does not close the new unconstructed
`31 x 43`, arbitrary adaptive word decoders, or native numerical lifting.
F-086 closes that `31 x 43` successor. A positive anti-flag internal-edge
bound yields a concrete overlapping adjacent pair, and peeling it makes the
cancellation theorem recursive on rank `r-2`. This closes the first 856
fixed-linear capacity-ordered shapes and leaves the unconstructed `31 x 42`,
but still does not cover adaptive word addresses, nonlinear output decoding,
or native numerical lifting.
F-087 closes the natural adaptive extension-field successor. On the zero
source, value-dependent probes of linear field cells expose one fixed support;
exactness forces the query into that support's field span. Exact support
counting requires 3,421 probes at `k=16,384`, and an aggregate storage
relaxation still fails after granting all DFloat11 bits plus 8 GiB to the
one-bit surrogate. This does not cover nonlinear stored cells or cross-block
mixed cells.
F-088 closes block-local adaptive words made from arbitrary independent
linear summaries. Zero-transcript supports plus exact Segre subspace
intersections require eight words at `31 x 42`; every side-128 rectangle
misses the registered traffic line even with full padding and four favorable
Q4 lanes. It does not cover nonlinear word contents or global mixing.
F-089 closes the claim that arbitrary nonlinear stored bits automatically
escape rank amplification. Adaptive depth becomes cylinder degree, and
products of rank-one characters force exact determinantal capacity at degree
`rt`. It also closes the smallest systematic `2 x 3` plus one arbitrary
advice-bit/two-probe seed. It does not construct or reject the fully
non-systematic seven-bit seed, the `25 x 108` nonlinear-word capacity point,
global mixed cells, or native arithmetic.
F-090 closes that `25 x 108` nonlinear-word capacity point without restoring
linearity. Largest-fiber recursion plus exact Segre intersections applies to
arbitrary nonlinear stored words, arbitrary deterministic value-adaptive
addresses at every depth, and arbitrary final Boolean logic. It closes all
target-feasible side-128 local 64-bit word cases. The new first local unclosed
points are `24 x 225` and `25 x 216`, 99 words/four probes; their nonadaptive
and one-value-stage adaptive submodels are also closed. A stronger independent
32-query union Gate also forces `64/675`, exactly 8x target, on some abstract
query tuple. The legal batch-1 Transformer reachability of that tuple plus
global nonlinear/native lifts remain open.
Read that continuation and each scope boundary before reopening a family.

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

The old hosted run remains INVALID. The repaired actual full run at
experiments/codex_fmm_integrity_20260909/hardened_run now rejects only programs
retained/evaluated by the frozen catalog/orientation/beam/cut/family/IO-bin search.
Direct arithmetic38.25165%, free-transform oracle13.01026% both miss10%.
The only empty direct cell is fully accounted:2,304 workspace rejections,
minimum12.24255GiB. Completed no-plan had been mislabeled as corruption.

Pinned coefficient-cap2 catalog, depth<=12, bounded search and perfect N/A=1
future/repair/offline grants remain. No general FMM/native impossibility.
Revisit only with a materially different paid construction.
<!-- EXP-100A:END -->

<!-- EXP-102A:START -->
## EXP-102A — real causal greedy draft block

The frozen real causal draft/verify source failed the raw A, N/A, latency, or exact-state Gate even before 405B scaling. Reopening requires a materially different causal source.

Frozen scope: pinned SmolLM2-360M target; same-family SmolLM2-135M and cross-family TinyStories-33M drafts; K `64,96`; build-only K selection; raw no-compression traffic threshold; exact terminal KV; all online work charged.
<!-- EXP-102A:END -->

## F-CAUSAL-GLOBAL-20260908 — literal exact response-column delta maintenance

The standard-causal sign/delta compiler in
`experiments/causal_global_bridge_20260908/` admits successive legal desired
right factors with Hamming distance equal to the full right width. An exact
state that maintains `y=W s` by explicitly applying the changed source columns
therefore touches/aggregates all columns on that legal transition and restores
the complete dense effect.

Do not reopen the old response-column/time-axis transport as a "dynamic exact
summary" unless a new finite subdense aggregate for arbitrary dense column
changes is constructed and all representation/address/native-state costs are
paid. This entry does not reject all dynamic summaries or all globally nonlinear
matrix-vector data structures.
