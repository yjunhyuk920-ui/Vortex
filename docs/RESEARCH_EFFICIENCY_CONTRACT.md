# VORTEX Research Efficiency and Candidate-Selection Contract

## Purpose

Maximize progress toward the fixed 405B / 8 GiB / 4B-class objective per unit of research effort.

Scientific rigor remains mandatory, but completeness of a mathematical taxonomy is not a project goal. A candidate is not entitled to implementation merely because it is testable, elegant, adjacent to the previous experiment, or not yet falsified.

## Objective-gap rule

The target gap controls candidate selection.

For the declared 405B-versus-4B Q4 comparison, the target-equivalent fraction before selector, metadata, fallback, and runtime overhead is approximately:

```text
4 / 405 = 0.0098765432
current declared total allowance = 0.01185185185
```

Therefore an isolated optimization whose optimistic ceiling is only a few tens of percent is not a plausible universal core. It may be retained as an auxiliary component, but it may not consume the primary research track unless it composes with an independently justified mechanism that closes the remaining orders-of-magnitude gap.

## Mandatory E0 candidate triage

Before opening an experiment branch, every proposed core mechanism must answer and record:

1. **Target-scale upside** — Under assumptions deliberately favorable to the candidate, can it plausibly reach at least one order-of-magnitude reduction in fully charged work or traffic, and is there a credible path toward the final approximately 1.185% target-equivalent fraction?
2. **Mechanism novelty** — Which previously rejected assumption or mechanism class does it replace rather than rename?
3. **Evidence basis** — What measured structure, theorem, architectural fact, or scaling argument makes the mechanism plausible on real dense Transformers?
4. **Scaling reason** — Why should the useful effect remain stable or strengthen from small checkpoints toward 405B rather than disappear?
5. **Universality** — Can it be derived automatically for arbitrary public unmodified dense checkpoints without a user-authored target-specific adapter?
6. **Correctness closure** — What exact, declared probabilistic, or fail-closed contract prevents silent output corruption?
7. **Resource closure** — Are selector, metadata, compile, intermediate, verification, correction, fallback, RAM, SSD, PCIe, and VRAM costs included?
8. **Cheap falsifiability** — What is the smallest theorem, certificate, oracle upper bound, or real-checkpoint measurement that can kill the idea before backend or kernel work?

A proposal that cannot answer these questions is not a core experiment.

## Immediate rejection rules

Reject or classify as auxiliary before implementation when any of the following holds:

- even the optimistic upper bound cannot produce a material order-of-magnitude improvement;
- the mechanism depends on a family already closed by committed evidence and introduces no new information source or structural reason;
- the maximum benefit is confined to isolated matrices, prompts, rows, heads, or synthetic controls with no population-level scaling argument;
- indexes, metadata, selector scans, intermediates, verification, or fallback erase the apparent gain under favorable accounting;
- success requires training, checkpoint modification, future target tokens, an unbounded state table, or undeclared approximation contrary to the fixed mission;
- there is no plausible reason the effect should survive larger hidden sizes, more layers, broader prompts, or arbitrary dense checkpoints;
- the proposed next step is only another parameterization, mode ordering, rank choice, threshold, or nearby decomposition of a repeatedly rejected mechanism family.

Do not perform experiments merely to complete a list of known decompositions or optimization techniques.

## Cheap-kill-first rule

Every core candidate must begin with the cheapest decisive Gate available.

Preferred order:

```text
information/resource upper bound
-> exact algebraic or causal certificate
-> favorable oracle upper bound
-> pinned small-real-checkpoint measurement
-> minimal operation replacement
-> backend/kernel implementation
-> target hardware
```

No physical kernel, model-wide backend, large parameter sweep, or long-running workflow is allowed before the candidate survives the earlier cheaper Gate.

A negative lower bound or unfavorable oracle ceiling closes the candidate immediately. Do not build a constructive implementation to reconfirm an already decisive negative bound.

## Family-closure rule

Repeated negative evidence must narrow the search space.

After a mechanism family is rejected on real checkpoints or by a valid universal bound:

- retain reusable certifiers, validators, formats, and fail-closed machinery as auxiliary;
- prohibit nearby variants unless they introduce a materially different information source, execution dependency, or asymptotic mechanism;
- do not increase experiment numbers for cosmetic variations;
- record the exact new assumption that justifies reopening the family.

The next core experiment must either change mechanism class or present new measured evidence that invalidates the prior rejection premise.

## Research portfolio default

Unless current evidence justifies a different allocation, direct research effort approximately as follows:

```text
70% high-upside new execution paradigms capable in principle of 10x-100x change
20% cheap falsification, bounds, certificates, and real-checkpoint screening
10% auxiliary engineering, cleanup, and incremental optimization
```

This is a prioritization rule, not fabricated time accounting. The primary track must not be consumed by optimizations whose best plausible outcome remains far from the objective.

## Promotion requirements

A candidate may advance from cheap screening to substantial implementation only when all are true:

- the favorable ceiling survives the declared p50/p90 thresholds;
- population-level evidence exists, not only a best-case fragment;
- costs are fully charged at the current evidence level;
- the candidate is not a renamed rejected approach;
- a route from the screening metric to actual Transformer operation replacement is explicit;
- the expected information gain from the next implementation stage justifies its cost.

Passing a local mathematical property is not promotion. Passing a storage Gate alone is not promotion. Reducing query bytes while doubling operations is not promotion.

## Current directive after EXP-065

EXP-066 Tensor-Train/MPO is authorized only as a bounded cheap-kill bond-rank certificate Gate because it is already preregistered and strictly generalizes the one-cut Kronecker test.

Before EXP-066 survives its lower-bound Gate, the following are prohibited:

- exact MPO core reconstruction;
- physical MPO kernels;
- model-wide MPO runtime integration;
- broad mode-order or factorization rescue searches beyond the preregistered schedules;
- a sequence of adjacent classical single-matrix tensor decompositions.

If EXP-066 shows high bond ranks or fails the operation/storage thresholds, close exact classical single-matrix tensor factorization as a primary core direction for the measured population.

The following primary candidates must then come from a materially higher-upside execution class, with their own E0 triage:

- joint multi-projection or multi-layer common-arithmetic compilation, such as Q/K/V and Gate/Up computation sharing;
- certificate-guided demand-driven or lazy execution that computes only values required to settle the final decision and fails closed;
- proposal plus exact verification only if verification is proven substantially cheaper than a target forward;
- another genuinely new information source or execution representation with a credible route to the final target fraction.

These are priority classes, not presumed solutions. They must still pass the same cheap-kill and proof-first rules.

## Current directive after EXP-072A

EXP-066 failed static storage; EXP-067 through EXP-070 failed joint reuse, demand, temporal replay, and local-pattern Gates; EXP-071 did not prove impossibility. EXP-072A then rejected a self-contained exact arithmetic DAG as a universal hot core by an information-capacity Gate, so the deferred synthesis/kernel plan is prohibited on the primary track.

Before another cold-backed core proposal, EXP-073 must replace proxy hardware assumptions with a sanitized read-only inventory and separately authorized same-machine baselines. This calibration is not part of the 70% core-candidate allocation and may not expand into general hardware engineering. Its purpose is to fix the physical byte, latency, capacity, and compatibility constraints that every future E0 candidate must close.

EXP-073 Stage 1 is now complete. Future E0 accounting must use the measured 8,192 MiB GPU, maximum PCIe Gen2 x16 link, 23.4983 GiB host RAM, and 97.6183 GiB root free capacity rather than broader proxy hardware. Actual storage/H2D bandwidth and native 4B Q4 latency remain unavailable until separately authorized Stage 2, so no candidate may substitute PCIe signaling rate or storage type for measured performance.

After calibration, the next core candidate must introduce a new query-time information source or execution dependency beyond exact static compression, local/row/column reuse, absolute-unread norm bounds, temporal exact replay, and enumerative advice. It must preregister a credible path to `1.185185%`, not merely to the older 10% screening threshold.

## Communication rule

Report research efficiency honestly:

- distinguish a useful falsification from progress toward runtime success;
- state when a result only closes a family;
- do not describe a long sequence of low-upside negative tests as increasing feasibility;
- report why the selected next experiment has higher expected value than rejected alternatives.

## Current directive after EXP-074

The Qwen3.5-122B-A10B surrogate was screened by a no-download Gate. MTP-1 plus
expert paging fails at `10.0x` the 1B baseline-equivalent target traffic even
with free drafting and fixed expert reuse. Do not build a page scheduler,
download the 35B/122B checkpoints, or treat residency as speed for this path.

The only retained branch is a long accepted block. Its optimistic p50 minimum
is nine tokens, but proposal cost and route diversity raise the reference
requirements to 25-612 tokens. The next cheapest information gain is therefore:

```text
metadata-only native-MTP surface audit
-> smallest unchanged checkpoint accepted-prefix Gate
-> middle-rung MoE route-union trace only if proposal survives
-> physical scheduler only after EXP-073 Stage 2 and logical closure
```

This is restricted surrogate screening, not a replacement for the primary
arbitrary dense 405B research track. Any continuation must explain how its
information source transfers beyond Qwen-specific MTP/MoE or remain auxiliary.

## Current directive after EXP-075

Static MTP availability is no longer the blocking assumption for the selected
official 0.8B checkpoint: 15 MTP tensor keys and a pinned vLLM loader surface
were confirmed. Do not spend further primary-track effort on metadata variants,
other small Qwen sizes, quantization conversions, or runtime backends before the
accepted-prefix Gate.

The highest-value next action is the bounded EXP-076 causal measurement because
it can cheaply kill the only retained EXP-074 branch. It must measure population
p05/p50 accepted length and exact rollback, not configured draft length or an
isolated prompt. Proposal, tied LM-head, target verification, rejected suffix,
and fallback costs must be derived from actual tensor shapes and calls.

If the fully charged favorable ceiling fails, close the long-block native-MTP
surrogate and return the primary portfolio to a materially different execution
information source. If it passes, authorize only a middle-rung MoE route-union
trace. Backend, scheduler, 35B/122B download, and target-server work remain
prohibited until their prior Gates survive.

## Current directive after EXP-076

The fully charged native-MTP branch failed. Build-selected `K=4` produced
held-out accepted-prefix p05/p50/p95 `0/4/4`, below shape-required `9/11`
tail/median minima, and two zero-accept cases made the traffic Gate fail closed.
Do not allocate work to K/prompt/quantization variants, another small Qwen,
35B/122B weights, router traces, or a page scheduler.

There is no promoted core candidate. The next core proposal must begin at E0
with a materially different query-time information source and a fully charged
credible path to `1.185185%`. EXP-073 Stage 2 may be separately authorized to
measure the real 4B/storage/H2D envelope, but calibration belongs outside the
core-candidate allocation and cannot be reported as feasibility progress.

## Current directive after EXP-077A

The new activation-conditioned information source passed admission to a cheap
favorable oracle but failed decisively. At the 10% MLP budget it preserved only
`71.5278%` held-out top-1 and exceeded the mean/p95 KL ceilings by
`44.21x/61.62x`; all family Gates failed. This is useful family closure, not
increased target feasibility.

Do not allocate primary-track work to fraction/layer/prompt sweeps, blockifying
the same score, training a target-specific router, sparse-kernel optimization,
or larger-model confirmation. The next core candidate must recover or amortize
the omitted nonzero contribution through a materially different execution
dependency and pass a fully charged E0 route to the final fraction before code.
EXP-073 Stage 2 remains separate physical calibration.

## Current directive for EXP-078A

EXP-078A is admitted only because it changes the reused object from selected
channels or prior outputs to the complete transient computation law. The direct
constructor reveals an unfavorable but not logically impossible lifetime
requirement. The cheapest remaining falsification is therefore the already
available seven-position unchanged-checkpoint trace.

Stop immediately on early population, family, top-1, or KL failure. Do not run a
longer trace, construct dense macro matrices, sweep ranks, or implement kernels
after such a failure. If all observations are right-censored, preregister a
longer trace and a constructor route before promotion.

## Current directive after EXP-078A

The cheapest Gate rejected frozen tangent reuse at the first later token for
every held-out case. Do not allocate work to longer traces, ranks, selected
layers/prompts, dense construction, or sentinel engineering around this path.

A new candidate must change the execution dependency again. A delta-updated
operator is admissible only if its E0 proposal explains how the delta is obtained
without first doing the skipped full gate/operator work and charges update,
verification, exact cache repair, fallback, RAM/SSD/PCIe/VRAM, and cold traffic.
No core candidate currently survives.

## Current directive for EXP-079A

EXP-079A is admitted because it changes the information interface: current
causal activations may query exact original row blocks, while unresolved data is
carried as a correlated fail-closed proof state. It is not admitted as another
exact low-rank representation; the DCT image is only a hot center and all
residual traffic remains explicit.

The shape-only dense-405B plan reaches the final p50 arithmetic ceiling only by
granting every other operator, dual oracle selectors, nonlinear propagation,
and failed fallback for free. Therefore execute only the pinned small-checkpoint
favorable Gate. Stop on budget, local-radius, population, family, top-1, or KL
failure. Do not sweep rank/basis/block/layer/prompt choices or build a nonlinear
interpreter, fallback engine, kernel, larger-model run, or target-hardware path
after failure. A pass authorizes only nonlinear proof propagation with newly
charged selector and fallback equations.

## Current directive after EXP-079A

The p50 arithmetic route passed but both favorable information Gates failed by
orders of magnitude: held-out top-1 was `4.1667%`, and the independent minimum
sound radius was `48.663918x/57.778748x` at p50/p95. Increasing to the p95
allowance did not create a near-lossless ceiling.

Stop all fixed pilot basis, rank, row-block, layer, prompt, fraction, selector,
nonlinear-propagator, and kernel work around this path. A new core proposal must
obtain information about the dense residual through a different causal or
interactive dependency and pass the final fraction before code. There is no
promoted core candidate.

## Current directive for EXP-080A

EXP-080A is admitted only because it adds a new asymptotic mechanism to the
closed multi-token family: exact cross-column bilinear arithmetic, not another
proposal source or acceptance sweep. It must grant perfect future activations
and execute only the arithmetic upper-bound prototype.

Stop if standard recursive Strassen cannot meet both final fractions after all
scalar additions, padding, and tile accumulations are charged. A favorable
modern exponent with unit constants is not constructive evidence. Do not build
a kernel or rerun Jacobi/draft families after failure. A pass authorizes only an
exact packed-constructor Gate; causality remains independently closed or
unsupported.

## Current directive after EXP-080A

Standard recursive Strassen failed the constructive arithmetic Gate by
`30.469145x` at its best registered block, despite free perfect future
activations and negligible amortized weight traffic. Stop kernel, tile, leaf,
block-length, draft, and Jacobi work around this engine. The unit-constant
exponent pass is target-setting evidence only.

A new proposal is admitted only if it changes one of the two missing interfaces:
an explicit exact low-constant rectangular construction that closes the full
arithmetic equation, or a new causal information source that creates long valid
blocks without equivalent target execution. Hold the other interface as a
clearly labeled oracle and apply the cheapest falsification Gate first. No core
candidate is promoted.

## Current directive for EXP-081A

EXP-081A is admitted because nonlinear lookup cells plus query-time error
syndromes introduce information absent from prior fixed subspaces, tangents, and
future-block attempts. Execute only exact synthetic controls and the held-out
small-checkpoint residual-code Gate.

Stop if exact weighted coverage is below `99.75%`, any family is below `99%`,
corrected p50/p95 error exceeds `0.01/0.05`, or fully charged target equations
fail. Do not tune tree/rank/field/layer/prompt choices after observing failure.
A pass authorizes only minimal integer projection replacement, not a kernel or
large-model run.

## Current directive after EXP-081A

The cheapest Gate rejected the population premise by more than ninety
percentage points. Stop all tree, code-rank, field, layer, projection, prompt,
tolerance, integer-kernel, and larger-checkpoint work around SRLM. Its exact
syndrome/fingerprint reference may remain auxiliary, but optimizing it cannot
remove `91.318328%` observed fallback.

No core candidate is promoted. Before opening EXP-082, test only a genuinely
different proof target at E0. The preferred high-upside direction is an
end-to-end exact greedy/fixed-RNG decision certificate with a free exact-trace
oracle choosing the minimum cold page set. If that favorable population/family
page fraction cannot approach `1.185185%` after interval state, selector,
verification, and fallback, reject the family without building interval
propagation, a runtime, or a kernel.

## Current directive after EXP-082A

EXP-082A closes exact terminal-only row/column Hamming spanning trees. Their
certified favorable coefficient lower bound is `1.562367394%`, above the final
`1.185185185%` budget before every positive systems cost. Do not build the MST,
sweep its hyperparameters, optimize a delta runtime, download a larger model,
or run hardware for this family.

No core candidate is promoted. The only adjacent question not logically
closed by this certificate is an exact circuit with useful synthetic
intermediate coefficient vectors. Keep it at E0 until (1) a novelty audit
separates it from EXP-053/054/072, (2) a causal compiler is specified without
uncharged dense discovery, and (3) edge work, intermediate state, traffic,
storage, build, verification, and fallback all fit a favorable target equation.
Do not assign a new experiment number merely to rename a closed circuit family.

## Current directive after the synthetic-intermediate E0 audit

Static synthetic trees and static shared linear DAGs are closed as a new core:
the tree expands into the exact synthetic arithmetic-DAG class already archived
under EXP-072B, and cold placement is only a residency change. Do not open
EXP-083, synthesize a Steiner network, rerun CSE/beam/addition-chain search,
build a cold circuit interpreter, or benchmark hardware for this class.

The only adjacent admissible class is a Query-Adaptive Cold Source. Before any
implementation, derive a fully favorable registered-shape equation charging
selection, index/state, probes, cold bytes, execution, intermediate state,
verification, misses, fallback, and compile amortization. Then identify a
causal information source from the committed prefix and unchanged checkpoint
that instantiates the equation without dense discovery or a free oracle. If
either is absent, retain `NO_SURVIVING_CANDIDATE` and do not assign an
experiment number.

## Current directive after the query-adaptive equation

The equation ticket is resolved. Even with zero fast-path and build cost, the
mean conservation Gate needs `98.814814815%` exact coverage and `84.375x`
useful information amplification. The retained verifier makes those
`99.081653739%` and `108.891389x` before selector, decode, miss, or state costs.

Do not optimize page size or implement a selector over raw Q4 pages: raw page
omission carries no information about the omitted dense contribution. Search
only for a Coded Causal Cold Source derived from the current committed
activation and unchanged checkpoint. Reject on paper any proposal that reduces
to static linear circuits, exact replay/advice, training, future target traces,
an external prover, or dense-equivalent discovery. No experiment number is
assigned until one source closes the equation favorably enough to preregister
the cheapest real-weight falsification Gate.

## Current directive after the post-Atlas causal-source audit

Forward causal pairs alone are the rejected Atlas normal form. Do not spend an
experiment on a renamed forward cache, free backward/transpose trace, static
full-vocabulary composite, approximate MIPS without an exact charged fallback,
or proof system without a local result source. The dynamic dual constructor
already costs `1.5625%` at the 64-token service life, and the favorable static
last-down table is `12.720703125 GiB`; both fail before implementation.

The only adjacent high-upside question is the Bilinear Cross Residual
`r^T W u`. Apply the cheapest Gate first: either derive a scoped lower bound
for the admissible preprocessing/query model, or specify a concrete lossless
checkpoint-derived structure with finite 405B build, state, operation,
traffic, verification, miss, and fallback equations. Do not assign an
experiment number, run a model, build a kernel, or use hardware until that E0
equation fits. Retain `NO_SURVIVING_CANDIDATE` otherwise.

## Current directive after the causal bilinear span threshold

The factor-scanned query-span interface passes E0 only through dimension 23
and only with pair extraction, the local result source, native numerical
repair, and complete runtime state granted free. Its 36-row last-down screen
has a decisive rank-28 stop derived from the weighted fallback equation.

Execute only that frozen E1 structural Gate. Do not sweep span dimension,
side rank, primes, prompts, positions, or decision directions. Stop as soon as
rank 28 is certified. A failure closes the factor-scan ledger before a general
VJP extractor, exact numerical decoder, backend, larger model, or hardware
work. A pass authorizes only a paid pair-extractor and native-semantics Gate;
it does not promote a Core Candidate because the result/proposal source is
still absent.

## Current directive after the query-adaptive code-union bound

Do not spend another forward pass on partitioning or extending a trace-built
linear ledger. Perfect routing cannot exceed the independent answer directions
that were actually constructed; the strongest registered union covers only
`0.43979%` of a 20M independent population, and every partition of the full
EXP-084A build span misses all five frozen evaluation rows.

The only open high-upside interface is now an implicit nonlinear
checkpoint-derived exact source for `r^T W u`. Demand a concrete constructor
and query equation before implementation. Kill it at E0 if it hides dense
discovery, an exponential table, a future trace, a free membership/proof
oracle, an approximate-only answer, or any already rejected static linear,
separable, or global-advice form. Do not assign EXP-085, run a model, build a
backend/kernel, download a larger checkpoint, or touch target hardware until
the complete target-scale equation survives.

## Current directive after the finite-word source audit

Do not spend implementation or hardware time on local two-sided truth tables,
Mailman/Four-Russians, full-scan broadword packing, Boolean rectangle
nonemptiness, or exhaustive native-state tables. Their cheapest decisive E0
charges already fail.

Work only on the General Finite-Word Rank-One Probe Gap. Before any execution,
require either a concrete globally nonlocal nonlinear exact constructor with
every target-scale charge, or a theorem that covers that precise model under
the global 8 GiB advice grant. A literature family name, asymptotic hidden
constant, Boolean output, or current weak direct-sum bound is not enough. Keep
NO_SURVIVING_CANDIDATE and do not assign EXP-085 until that E0 boundary
changes.

The finite theorem substitution in
`docs/research/E0_GENERAL_FINITE_WORD_RANK_ONE_PROBE_BARRIER.md` does not change
that boundary. Nisan--Rudich--Saks reaches only five strict GF(2) bits and an
over-favorable 64-bit physical-file ceiling. Ko
2025 misses the structured rank-one target by its explicit collection and
query-count premises and does not grant advice probes free. CKL's
single-matrix redundancy range cannot absorb
the global advice. Do not promote this barrier audit into a universal
impossibility theorem or turn 2.5% into a scheduler setting.

## Current directive after the global nonlinear frontier audit

Do not spend another constructor cycle on all-zero rectangles, Boolean
nonemptiness, direct exact-subrectangle summaries, or bit-plane naming. The
singleton query argument already shows why the direct numerical summary must
retain the raw covered information. Do not spend another theorem cycle on
KPI25 limited independence for the full rank-one code; its three-query XOR
dependency falsifies the premise. Do not use whole-MatVec or dynamic
Multiphase bounds without the explicit `n`-fold scalarization or model-change
charge.

Continue only on a materially different E0 route: either a globally coupled
adaptive exact numerical data structure whose information source is not a
hereditary Boolean rectangle and whose complete 405B equation fits, or a
direct scalar rank-one lower bound covering arbitrary nonlinear 8 GiB advice.
Cross-request full-sweep batching may be analyzed only as a separately
declared throughput contract. Forty outputs give `1/40` per output, which is
32x the registered `1/1280` per-token allowance and leaves block `rho=1`;
never cite it as evidence for the registered single-stream goal.
Keep NO_SURVIVING_CANDIDATE and do not assign EXP-085 or touch model/hardware
work until the active ticket's deliverable changes.

## Current directive after the finite-semiring preprocessing audit

Do not spend another cycle translating Williams' full MatVec graph to Q4,
BF16, bit planes, or a 32-vector wave. Its cheapest physical expansion already
exposes the invariant tradeoff `query/raw=1/b`, `sidecar/raw=K^b/b`; hiding
neighbor values, addresses, the pattern catalog, or native regrouping changes
the problem rather than solving it. Do not benchmark a truncated graph or use
one-bit query success while omitting its wrong arithmetic and `55,292.57 GiB`
model-wide sidecar.

Continue the claimed ticket only through a materially different scalar route:
a globally coupled bounded-word exact numerical representation with a complete
native-order 405B and 32-token equation, or a direct scalar rank-one lower
bound that grants arbitrary nonlinear 8 GiB advice and adaptive probes. Apply
the cheapest finite E0 falsification first. Keep NO_SURVIVING_CANDIDATE and do
not assign EXP-085, run a model, build a backend/kernel, or touch hardware
until one required ticket deliverable exists.

## Current directive after the global-advice synergy audit

Do not divide global nonlinear advice by matrices, layers, or tiles; do not
sum conditional information; and do not compose independently chosen CKL
worst cases without a direct-sum proof. XOR synergy is the finite
counterexample. Also do not quote the displayed `n^2/64` finite coefficient
when the substituted advice lies outside that proof regime or turn a hidden-
constant-one diagnostic into evidence.

Continue only through a theorem that retains one arbitrary global 8 GiB
advice function and jointly charges every adaptive cross-matrix probe used to
unlock it for a composable scalar rank-one transcript, or through a concrete
globally coupled constructor with a full native-order 405B/32-token equation.
Keep NO_SURVIVING_CANDIDATE and do not assign EXP-085, run a model, build a
backend/kernel, download a larger checkpoint, or touch hardware until one
required ticket deliverable exists.

## Current directive after the Fourier-fiber direct-sum audit

Retain the all-linear fiber theorem; do not repeat XOR/advice-allocation
arguments. Also do not transfer full Walsh spanning to `r tensor u`, multiply
the 26.2% witness into a Transformer claim, or call the 0.028137293% method
ceiling an algorithm. Query cardinality alone is now a closed theorem route.

Continue only by proving target-scale rank or adaptive-tree complexity for
the restricted rank-one character matrix on every large nonlinear advice
fiber, or by specifying a concrete scalar-specific bounded-word constructor
with complete native-order 405B/32-token accounting. Keep
NO_SURVIVING_CANDIDATE and do not assign EXP-085, run a model, build a
backend/kernel, download a checkpoint, or touch hardware until one required
ticket deliverable exists.

## Current directive after the spiky / entrywise-power audit

Do not implement or search parameters for sums of spiky, permutation-block,
or block-diagonal rank-one components as a direct evaluator.  The finite
description Gate already includes sparse supports and arbitrary block masks;
even a globally pooled compute allowance with cross-matrix components misses
the registered sign-checkpoint space by `218,789,422,765.93` log2 bits. Do not
reopen entrywise integer powers of
low-rank roots: exact multinomial expansion is the closed static low-rank
normal form.

Continue only with a materially different query algorithm: either a globally
adaptive bounded-word exact numerical data structure with complete
native-order 405B/32-token accounting, or a covering lower bound that grants
one global 8 GiB nonlinear advice function and jointly charges every adaptive
cross-matrix probe.  Keep `NO_SURVIVING_CANDIDATE`; do not assign EXP-085,
run a model, build a backend/kernel, download a checkpoint, or touch hardware
until one active-ticket deliverable exists.

## Current directive after the adaptive codebook / trapdoor audit

Do not materialize one answer per nearest codeword pair or joint representative
query. The exact-difference Gate already grants the optimal router and
arbitrary centers; the fixed-side packing still demands a 13,742-bit address.
Do not hide the table behind an unnamed compression function. Such a function
is the open mechanism and must be specified and charged directly.

Do not mask an arbitrary checkpoint with a sampled trapdoored matrix unless a
separate exact solver for the shifted arbitrary matrix is supplied and fully
charged. `Wx=(W+R)x-Rx` does not eliminate `(W+R)x`.

Continue only with a concrete nonliteral compressed decoder or a joint lower
bound that covers it under global 8 GiB advice and adaptive cross-matrix
probes. Keep `NO_SURVIVING_CANDIDATE`; do not assign EXP-085, run a model,
build a backend/kernel, download a checkpoint, or touch hardware.

## Current directive after the extension-field rank-saturation screen

Do not spend another cycle treating rank weight, projective points, or a
rank-covering radius as physical locality. The direct materialization and the
query-specialized identity layout already fail favorable finite resource
Gates. Also do not enlarge the tiny dictionary hill climb: its purpose was to
show that nonlinear leaders are a real distinct mechanism, not to infer scale.

The only admissible continuation of this code route is a batch-aware nonlinear
coset constructor: for all queries in one causal batch, it must find one small
shared set of physical atoms whose span contains them, without scanning the
dictionary. Persistent state, native summaries, address bytes, decoding, KV,
verification, and fallback remain charged. Without an explicit construction
and a favorable 405B equation, change mechanism class. Keep
`NO_SURVIVING_CANDIDATE`; do not assign EXP-085 or run model/hardware work.

## Current directive after the joint-batch Segre geometry screen

Do not enumerate more tiny dictionaries, cache factor envelopes, or derive
another query-cardinality bound. The exact product-code weight hierarchy and
complete controls have already fixed the geometry: K rank-one queries lie in
a `K x K` factor envelope, literal registered catalogs exceed
`2^1,046,528` names, and proof-safe support counting forces only 20,218 bit
atoms. The first is impossible to store literally and the second is too weak
to decide the target.

Continue this route only with an implicit near-source-size code that generates
the requested `R tensor U` restriction and physical addresses on demand,
including native order and complete 405B costs. Otherwise change mechanism
class after its cheapest novelty Gate. Keep `NO_SURVIVING_CANDIDATE`; do not
assign EXP-085 or run model/hardware work.

## Current directive after the sparse functional dictionary audit

Do not revisit literal bilinear answer books, direct local rank-one tables, or
coordinate-LDC terminology. Their missing functional recovery structure is now
explicit. Also do not cite the 494,026-word counting boundary as an algorithm:
it says only that query-name cardinality loses resolving power.

The one adjacent open linear route is an **aligned** near-linear cold
dictionary with a succinct atom generator and a sub-dense exact rank-one
decomposer. Continue it only by giving such a construction or by proving a
restricted-rank-one covering lower bound for the non-systematic linear model.
Native numerical width/order, atom metadata, physical random probes, and all
405B state/scheduler/fallback costs remain mandatory. Otherwise change to a
materially different globally coupled nonlinear source. Keep
`NO_SURVIVING_CANDIDATE`; do not assign EXP-085 or run model/hardware work.

### Directive after the fixed-linear decoder Gate

Do not try another invertible transform, FFT basis, tensor basis, or linear
syndrome map. Their 32-query union activity is already closed with perfect
packing and free hot cells. Continue the dictionary route only through an
explicit nonlinear sparse coset representative whose search is itself
sub-dense and whose atom layout is succinct. Apply a finite toy/exact Gate to
that nonlinear decoder before any model work. A causal restriction is a
separate route and must prove reachability exclusion rather than assume it.

## Current directive after lossless/cut/gauge screening

Do not spend another cycle optimizing entropy codecs, attention gauges, or
cut-query notation as standalone cores. Their favorable ceilings already miss
the final contract before omitted costs. Fused lossless execution remains a
legal auxiliary only for a new mechanism that removes or amortizes the full
sweep by at least the registered 841-token favorable requirement.

Continue with a globally coupled exact source that includes MLP, embeddings,
and persistent state, or with a lower bound that actually covers nonlinear
global advice. Keep `NO_SURVIVING_CANDIDATE`; do not assign EXP-085, run a
model, build a backend/kernel, download a checkpoint, or touch hardware.

## Current directive after the corrected average-oracle audit

Do not spend an E1 run on OMEGA-XORLIFT or another average-case amplifier
unless a new concrete oracle has already passed E0. The common-row Fourier
Gate is not the paper's average-distance premise. OMEGA-ROWLOTTERY supplies a
valid accuracy counterexample but fails immediately because its direct
rank-covering row forms read at least one full coefficient source.

A cold-backed variant must begin with a materially different compressed
nonlinear oracle and close all repeated probe and call costs, random-instance
construction, decoding, verification, native numerical lifting, and fallback.
Otherwise continue with a different non-oracle source or a covering
global-advice lower bound. Keep `NO_SURVIVING_CANDIDATE`; do not assign
EXP-085, run a model, build a backend/kernel, download a checkpoint, or touch
hardware.

## Current directive after the nonlinear-fiber decision-depth screen

The complete `2 x 2` nonlinear-fiber search is a cheapest-possible novelty
screen, not a scale result. Its optima are affine parity-covering witnesses;
all `2 x 3` affine fibers show the same behavior. Do not enlarge this toy by
parameter sweep or random sampling without a theorem that predicts a
high-dimensional separation.

Continue only with a concrete scalable nonlinear decoder or a large-fiber
rank-one theorem that jointly charges global advice and adaptive cross-matrix
probes. Keep `NO_SURVIVING_CANDIDATE`; do not assign EXP-085, run a model,
build a backend/kernel, download a checkpoint, or touch hardware.
