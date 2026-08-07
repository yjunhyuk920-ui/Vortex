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
