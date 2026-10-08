# Vortex native producer continuation

Prepared 2026-10-08 UTC from public primary manuscripts and pinned public code. Base: `ddb626aadc9fde8a9b1971967747c62196c90a96`, branch `research/cloud-continuation-20260930`. This is source-level constructive discovery and a bounded manual audit, not a numerical experiment, formal-checker run or performance result.

## Result and unchanged theorem target

The next missing object is still a finite, causal producer of the original native effect before the dominant work it replaces. The reviewed sources provide operational verification, real-system structural certification, or bounded-error surrogates. None supplies that producer with a sufficient target-scale upper cost bound. No new core is admitted.

Intended final theorem, **not established**: for every publicly released, unmodified dense HF 405B-class checkpoint and every original legal input/context/RNG/continuation under the declared reference interface, a finite automatic constructor produces an executor whose output words and RNG agree, whose required successor state has an inductive continuation correspondence, and whose full GPU peak is at most 8 GiB. Under the frozen same-machine workload population its warm p50/p95 latency is at most 1.2/1.5 times native 4B Q4, with the original TTFT requirement. Every constructor, source, host/SSD/GPU allocation, transfer, arithmetic, state, check, guard, repair and fallback is paid. No missing ABI, population, baseline or TTFT detail is silently assigned a convenient value.

This continues [the prior producer-cost screen](../source_frontier_20261008/REPORT.md). Three different information dependencies are compared below. They are discovery principles, **not three qualified new cores**; no CSE, codec, TOP, scalar chart, replay or low-rank family is reopened.

## 1. Prove a native reachable-state property, then structurally omit its effect

**Outline.** Construct an invariant covering legal native states, certify that an original region has a cheaper exact observable/state effect on that invariant, and evaluate a paid membership guard before the omitted region. A rejection executes the original region. The proof must cover the original rounded program, not just its real formula.

**Exactness obligation.** If the guarded source returns `(output, encoded_next_state)`, it must agree with the reference output/RNG and preserve the state relation for every admitted continuation. The guard cannot depend on an output obtained by executing the region it intends to skip.

### Operational software verification and activation specificity

[NeuroCodeBench 2.0](https://arxiv.org/html/2510.23389v1) studies plain-C neural software, explicit mathematical-library operational models, and networks up to 170K parameters. Its results expose soundness/scalability difficulties; it supplies no cheap Transformer saturation-source constructor.

At pinned commit `92901329b63d2efcb3b06305c7c3cf0d4192eae3`, the [GLU harness](https://github.com/emanino/plain_c_nn_benchmark/blob/92901329b63d2efcb3b06305c7c3cf0d4192eae3/benchmarks/activation_functions/safety_properties/glu/glu_0_safe.c) computes logistic through `0.5*tanh(0.5*y)+0.5`. Its nondeterministic scalar assumptions are not an HF reachable-state invariant. The [dense harness](https://github.com/emanino/plain_c_nn_benchmark/blob/92901329b63d2efcb3b06305c7c3cf0d4192eae3/benchmarks/neural_layers/safety_properties/linear/linear_0_safe.c) retains the coefficient-product accumulation loop.

The separately inspected [PyTorch CUDA SiLU](https://github.com/pytorch/pytorch/blob/302f29e957bd44cd3ac34cbbddf4de25381d99ac/aten/src/ATen/native/cuda/ActivationSiluKernel.cu) uses `x_acc/(1+exp(-x_acc))`; [opmath](https://github.com/pytorch/pytorch/blob/302f29e957bd44cd3ac34cbbddf4de25381d99ac/aten/src/ATen/OpMathType.h) promotes Half/BF16 internally to float. These upstream pins are implementation examples, **not** a validation of Vortex's older frozen torch2.8.0 CPU ABI or an unspecified target GPU. A saturation property cannot be transferred between these paths without proof. Native zeros are not declared impossible.

The inspected [HF LlamaMLP](https://github.com/huggingface/transformers/blob/cb33194ad6152bd9fad6305378d92db385dd7b32/src/transformers/models/llama/modeling_llama.py#L152-L165) evaluates gate, activation, up branch, product and down projection; it supplies no pre-projection guard.

### A concrete certificate implementation retains dense work

At TorchLean `f0bd983dfc107e1f776d56f3492932ef570cce16`, [IBP.linear](https://github.com/lean-dojo/TorchLean/blob/f0bd983dfc107e1f776d56f3492932ef570cce16/NN/MLTheory/CROWN/Core.lean#L204-L231) traverses all M×D coefficients in each endpoint fold. The displayed source makes 4MD directed multiplication calls, 2MD directed accumulation calls and 2MD endpoint selections, plus bias work. These are derived source-operation counts; they are not compiled-instruction or latency measurements.

For the displayed fresh-gate-enclosure procedure, all MD gate coefficients remain requested. Against three equal-sized SwiGLU projection streams, even granting free omission of both other projections leaves at least one-third of the original logical coefficient inventory. It therefore supplies no tenfold FFN coefficient-traffic route. Actual cache transactions and whole-model timing are unmeasured.

The [IEEE32 executable nonlinear backend](https://github.com/lean-dojo/TorchLean/blob/f0bd983dfc107e1f776d56f3492932ef570cce16/NN/MLTheory/CROWN/Extras/BoundOpsIEEE32Exec.lean#L93-L109) returns sigmoid bounds [0,1] for every accepted interval. Exceptional/unsupported paths may return none. This fallback gives no input-specific native saturation word.

[Full directed-IBP soundness](https://github.com/lean-dojo/TorchLean/blob/f0bd983dfc107e1f776d56f3492932ef570cce16/NN/MLTheory/CROWN/Proofs/DirectedIBPFullSoundness.lean#L127-L145) assumes [real node equations](https://github.com/lean-dojo/TorchLean/blob/f0bd983dfc107e1f776d56f3492932ef570cce16/NN/MLTheory/CROWN/Proofs/DirectedIBPFullSoundness.lean#L51-L68) and concludes enclosure of real values. Rounded verifier endpoints do not establish equality with the original CUDA/HF reduction, activation, cast or state. No Lean proof was run.

**Paid route and decisive gap.** A construction-time invariant could remove dominant work if cheap membership and maintenance cover the required population. Its complete cost is constructor + stored original/source/invariant + guard + exact producer + state update + rejected-path fallback. The strongest counterexample is a reachable state outside the claimed invariant or an activation/reduction path with different native words. The cheapest next deliverable is a finite native invariant and exact replacement algorithm with its input dependencies and sufficient upper costs. Larger verification suites or finer scalar thresholds do not supply it.

## 2. Evolve a compact exact observer state instead of recomputing the body

**Outline.** Derive an encoded state from the original checkpoint and initial state, update it causally from current legal input, and produce the original observers and successor state from that update. It must not obtain its innovation by first executing the original dense body.

**Exactness obligation.** Initial correspondence and every encoded transition must commute with the original native transition and all required observations. Costs include encoding, stored maps, nonlinear innovation, every observer/readback, maintenance, guards and fallback.

### Controlled Koopman certification is not the missing native closure

[On the Existence of Koopman Linear Embeddings for Controlled Nonlinear Systems](https://arxiv.org/html/2602.14537v1#S6.SS1) gives a symbolic CAP-structure certifier. Its exact finite-dimensional embedding also requires an autonomous invariant-subspace closure. The certifier alone does not construct that closure, and real control-affine equations do not establish native HF state correspondence.

The manuscript's literal repository link returns 404. An independently verified [matching public README](https://github.com/soc-ucsd/Existence-of-accurate-Koopman-linear-embedding/blob/0dc3f85ee6652f1e2e3984fb46100ce563fd9422/README.md) identifies the available implementation. No access restriction was bypassed.

### Static implementation discrepancy, narrowly scoped

At commit `0dc3f85ee6652f1e2e3984fb46100ce563fd9422`, [utility.py](https://github.com/soc-ucsd/Existence-of-accurate-Koopman-linear-embedding/blob/0dc3f85ee6652f1e2e3984fb46100ce563fd9422/utility.py#L53-L96) has `SCD(Phi,B)` without the residual Γ required by paper Algorithm 2. Its full-rank branch checks only Phi. The [driver](https://github.com/soc-ucsd/Existence-of-accurate-Koopman-linear-embedding/blob/0dc3f85ee6652f1e2e3984fb46100ce563fd9422/main.py#L13-L31) interprets flag=1 as success.

Hand-derived witness for this reusable code path: let x0⁺=x1²+u and x1⁺=x0. With Phi=(x1²,x0) and B=(1,0)ᵀ, the first step has identity transformation and returns lower-subsystem Phi*=0, B*=1. Its next full-rank step accepts. The discarded upper residual is x1²; carrying it into the next paper check rejects it.

Direct substitution from zero initial state gives x0(3)=u0²+u2, which cannot equal an initial-state term plus a constant linear map of all three controls on an open interval of u0. Thus the shown code path does not certify its advertised multi-step property. This is a **manual source-level counterexample**, not an executed test, formal proof, rejection of the paper theorem, or HF reachability result. No upstream fix or issue was sent.

### Actual measurement and recurrent-attention dependencies

The [September Koopman–Luenberger manuscript](https://arxiv.org/html/2609.36334v1) uses finite-time/data-based operator approximations. Its [pinned observer code](https://github.com/WentaoTang-Pack/KoopmanLuenbergerObserver/blob/a0cd9ceb26b80dd513b2c98d280c251e761cb0cb/luenberger_observer.py#L397-L421) integrates original plant and observer together: it obtains y=h(x), calls the original vector field and then applies observer correction. That schedule pays the plant plus observer; it is not an output producer that replaces the plant. No continuous-system code was run or promoted to native HF evidence.

The [softmax-as-RNN manuscript](https://arxiv.org/html/2507.23632v1#S3.SS1) represents the exact real attention numerator through infinitely many Taylor moments, holding the query-dependent denominator separately. It supplies no finite original-native closure. Its [pinned decoder](https://github.com/gmongaras/On-the-Expressiveness-of-Softmax-Attention-A-Recurrent-Neural-Network-Perspective/blob/295c4da34b7e07a2f2ae838bb3dc5590ee593143/GPT_Trainer/LlamaDecoderLayer.py#L552-L639) retains Q/K/V projections. The 80-term branch forms the full QK matrix and a truncated FP64 polynomial. Although it computes a normalized-weight variable, its return uses the unnormalized polynomial weights. This source path neither implements a compact exact recurrent executor nor establishes original softmax-word equality. The observation is scoped to that branch; no attention experiment was run.

**Paid route and decisive gap.** A genuinely closed native observer could amortize checkpoint work over a finite session. An embedding-existence or real-system CAP check is insufficient: original-native innovation and required observations must be generated before the removed work, with constructor dimension/storage and per-step upper costs. The strongest failure is a nonlinear feedback contribution omitted from the closure. The cheapest decisive check is the actual recursive source/state algorithm and all of its residual/observer dependencies. The existing synthetic Krylov and exposed-product results are not enlarged.

## 3. Use a cheaper surrogate only when an original-native effect certificate accepts

**Outline.** Produce a compressed/surrogate result, derive a causal certificate against the original native result, and accept only an identical original output/state effect. Failed certificates pay the original path.

**Exactness obligation.** A small norm error, equal top choice, or equality with a different unquantized kernel is insufficient. The accepted certificate must establish original output words, RNG consumption and state correspondence.

[Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/html/2605.20868v1) expressly distinguishes custom-kernel O_ref, quantized O_quant and production-SDPA O_dense. Section 4 bounds O_quant−O_ref; section 9.10 retains an arithmetic-path discrepancy from O_dense. Exact dense output is returned by dense fallback. Its producer scores all compressed keys, selects promotions, then performs attention; originals, scratch, page-in and checks remain necessary. Section 5.1's 56% compressed KV inventory is not a tenfold whole-system route. The study uses INT8 model weights, a 96 GB GPU and greedy generation, so it does not establish Vortex's unmodified-checkpoint, 8 GiB or RNG contract.

The paper's code/tag paths returned 404, and its DOI could not be retrieved through the available public read tool. Implementation details beyond the manuscript remain unverified; no verification/sign-in/agreement gate was acted on.

**Paid route and decisive gap.** The general route remains logically open only with a cheaper original-native certificate and sufficient accepted coverage. Charge surrogate generation + certificate + native path-gap enclosure + state update + all retries/fallback. Zero tolerance against O_ref does not prove equality with O_dense. The strongest counterexample is a native rounding/state disagreement inside an accepted error bound. The cheapest next check is a certificate for the actual original interface, not another approximate-quality or scalar tolerance experiment.

## Comparison and next construction

These principles respectively change what must be executed, what state must be evolved, and what evidence can authorize accepting a cheaper result. Their missing sources differ:

- Reachability slicing needs a paid native invariant and replacement effect before its projection.
- Observer execution needs a closed native innovation/observer recurrence without calling the removed body.
- Surrogate acceptance needs a cheap original-native output/state certificate, including arithmetic-path differences.

The first is the most concrete verification-interface lead from this review, but its inspected implementation fails the cheap-producer screen. None has a credible complete tenfold route; no candidate is selected for a numerical run. This is not a universal impossibility theorem.

For a declared work or traffic partition, keep the existing necessary screen f+p+(1−f)r≤0.1, with unchanged share f, paid extra source/check/maintenance share p and retained replaced share r. Work and traffic are assessed separately. Passing that screen would still require a finite schedule and sufficient upper bounds for the actual target and same-machine baseline.

The next useful artifact must name the omitted original region, legal native state relation, finite constructor, causal source/guard and exact native/state proof; then give paid construction/storage/per-step/short-session/fallback upper bounds. It may preserve a compressed state relation rather than reconstruct every hidden scalar, but every required continuation observer stays covered. No unspecified perfect selector or future target effect is accepted as a primitive.

## O1–O6 ledger

| Obligation | Status | Decisive missing item |
|---|---|---|
| O1 uniform automatic construction | OPEN | Target-wide frontend plus finite source/invariant/observer constructor |
| O2 complete causal program | OPEN | Executable replacement producer before removed work, through prefill and continuation |
| O3 native output/RNG/state induction | OPEN | Original ABI and all-legal relation; real enclosure or surrogate equality is insufficient |
| O4 total resource upper bounds | OPEN | Charged constructor, metadata, source, movement, native arithmetic, maintenance and fallback |
| O5 target memory/latency closure | OPEN | Sufficient 8 GiB schedule and frozen same-machine baseline/workload/TTFT comparison |
| O6 independent complete evidence | OPEN | Checkable complete theory/algorithm artifacts; source audit is auxiliary |

All dependencies remain the unchanged contract, original interface and actual source algorithm. No essential full-mission obligation closes here.

## Evidence and boundaries

Two independent source readings manually checked the CAP residual witness and pinned implementation. TorchLean's operation counts, fallback bound and real-equation theorem were independently read back. These are bounded human-readable source/manual checks; no verifier, reviewed source code, model, solver, native numerical harness or GPU was run. No checkpoint, customer material, user-computer command or hidden remote computation was used.

Documentation/JSON/link/hash/append checks are recorded separately in `LOCAL_VALIDATION.json`. Repository preservation will be verified by actual commit/readback; this authoring snapshot contains no self-referential final commit claim.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
FULL_MISSION_O1_O6=OPEN
NEW_NUMERICAL_EXPERIMENTS=0
MODEL_RUNS=0
SOLVER_RUNS=0
GPU_RUNS=0
README_CURRENT=true
HANDOFF_STATUS=IN_PROGRESS
