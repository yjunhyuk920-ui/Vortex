# VORTEX Constructive Theory Contract

Version: CTC-2026-09-05. User-approved governance, not a theorem or a performance result.

## 1. Primary deliverable and precedence

The primary research deliverable is a **constructive execution theory that closes the fixed mission**, not a growing collection of small lemmas, rejected candidates, tests, or commits.

A complete constructive theory gives a finite automatic constructor, an executable algorithm, original-output and successor-state preservation proofs, and complete resource/time upper bounds at the target scale. An existence claim or a conditional fast path is not enough.

This contract governs theory acceptance, research priorities, architecture neutrality, and scientific completion. It supersedes conflicting older clauses in policy files, research handoffs, and historical next-experiment directives. It does not override the fixed mission, user instructions, safety restrictions, or recorded evidence. Earlier failures retain their exact proved/measured scope. Historical experiment directives are not perpetual authorization.

## 2. Fixed mission and final theorem first

The target is unchanged: arbitrary public unmodified Hugging Face dense 405B-class checkpoints; executor replacement only; batch size one; one GPU with total peak allocation <=8 GiB; original output/RNG and required successor-state contract; same-machine native 4B Q4 warm time/token p50 <=1.2x and p95 <=1.5x. Preserve the existing TTFT/user-experience requirement. Do not invent a relaxed threshold where the protocol still needs to be fixed.

No retraining, fine-tuning, distillation, LoRA, semantic weight modification, user-authored model adapter, hidden remote compute, unbounded precomputation, or free fallback may replace that mission. Automatic lossless runtime representations must preserve the original checkpoint and charge their complete costs.

Start each core round by writing the intended final theorem, with explicit quantifiers over checkpoints, legal inputs, context lengths, RNG states, and the execution interface. Freeze the ABI, reference runtime, baseline, workload/latency population and statistical protocol before using results. Restrictions discovered during research must be stated, not silently removed from the quantifiers.

Maintain a proof-obligation ledger:

| ID | Obligation |
|---|---|
| O1 | Uniform finite automatic construction for every checkpoint in the unchanged mission scope |
| O2 | Complete causal program from loading/prefill through output and successor-state transition |
| O3 | Exact original-output/RNG contract and inductive state correspondence for all legal continuations |
| O4 | Explicit upper bounds on construction, storage, movement, arithmetic, state and all auxiliary work |
| O5 | Target-scale memory and latency closure against the same-machine baseline and declared population |
| O6 | Independent checkable proof/algorithm artifacts, dependencies, assumptions and reproducibility |

For every obligation record its statement, dependencies, open assumptions, proof/artifact location, status and next decisive action. Every primary task must name an obligation it resolves. `OPEN` is legitimate during discovery; an open essential obligation prevents full theory acceptance.

## 3. No hidden hard subroutines

Every critical encoder, selector, address generator, decoder, certificate builder/checker, numerical kernel, repair procedure and state updater needs inputs, outputs, a finite algorithm, a termination argument and a cost bound.

The following are research questions, not solved primitives: 'find a small exact circuit', 'select only necessary weights', 'certify all unread weights', 'recover the exact next state', or 'assume a long accepted block'. Perfect selectors, target future tokens/states, free metadata, unproved rank/compressibility/reuse, and unbounded search cannot discharge an obligation.

Separate explicit machine/ABI assumptions from algorithmic properties that must be proved. A hardware model may specify service bounds; it may not provide the missing decoder or guarantee the desired speed by definition. Unproved target-wide algorithmic assumptions mean `THEORY_STATUS=CONDITIONAL`, not theory achievement.

## 4. Sufficient upper bounds, not optimistic survival

A lower bound L <= T_goal only fails to reject a candidate. It does not establish T_algorithm <= T_goal. Peak-throughput/roofline bounds and favorable oracles are screening diagnostics, never sufficient achievement evidence.

Derive a finite execution schedule and a defensible upper bound U for that algorithm, under stated machine guarantees, with every dependency and service cost charged. Serial addition is a conservative default. Replace sums with overlap bounds only after proving the schedule, resource compatibility and bounded stalls. Maximum advertised bandwidth is not a guaranteed service rate.

Charge preprocessing/compilation, original and transformed weight storage, host RAM, SSD, GPU allocations including allocator overhead and fragmentation, transfers/pages/addresses, arithmetic and reductions, prefill, KV/cache, metadata, workspaces, verification, repair, rollback, synchronization and fallback. Give finite initialization and short-session costs as well as the explicitly justified amortization horizon. No unlimited history, future tokens or free offline work.

For each legal workload event z, a proved T_V(z) <= U(z) can support quantile bounds under the frozen common population. The p50/p95 comparison needs bounds for the baseline too: an upper bound for VORTEX and an upper bound for the baseline alone do not prove a ratio. Use a certified baseline lower bound or a directly proved coupled ratio in the same model. Keep real measured baseline quantiles and model claims separate, including uncertainty and warm/cold conditions.

The 4/405 parameter ratio and legacy approximately 1.185% allowance are heuristics for some accounting models, not universal latency proofs. Likewise, a >=10x work/traffic reduction is a core-entry criterion, not final success. All remaining costs must close the actual objective.

## 5. Architecture neutrality and constructive selection

Before selecting a core direction, invent three materially different execution principles. Each must reverse a hidden premise, computation order, information flow or verification unit, and provide a concrete algorithmic outline, exactness equation, paid resource equation, explicit >=10x elimination/amortization route, strongest counterexample and cheapest decisive test.

A new name or a combination/minor variant of existing techniques is not enough. Existing sound mathematical tools may be used inside a genuinely different execution principle; unsupported novelty claims are forbidden.

Compare the three and concentrate on the strongest survivor's unresolved construction. Each continuation round revisits the comparison using existing evidence; it need not abandon a surviving construction or rename three ideas. A new direction still requires three materially different proposals first.

Freeze the objective, not one architecture. Legacy block-verification variables A, N/A and r apply only where defined. Requirements such as A >> 1, N/A -> 1 and r << 1 cannot be imposed on unrelated architectures. Provide an equally complete model for the actual proposed algorithm. Method-specific gates remain binding only on the method they evaluate.

## 6. Small experiments must connect to the theorem

Before primary work on a toy/synthetic instance, name the exact Transformer operation or proof obligation it addresses, the lifting argument to native finite-word arithmetic and required state, and the scale argument. It may probe an unresolved lifting lemma, but must not pretend the lemma is already proved.

An isolated parity/storage/rank result, larger SMT timeout or proof packaging without that connection remains bounded auxiliary work. Do not automatically grow the toy problem after a pass. Work on the outstanding native numerical, causal information, state-transition or full-cost obligation.

Use cheapest-decisive-test-first to stop a rejected candidate promptly, not as a reason to fill the primary track with negative tests. Reopening a closed family requires a new information source, asymptotic mechanism, dependency or measured fact that defeats the recorded rejection premise. Preserve all failed evidence.

Default effort priorities remain approximately 70% high-upside constructive work, 20% decisive falsification and 10% auxiliary engineering. These are priorities, not fabricated time accounting. Progress is closure of essential obligations, not test/file/commit counts.

## 7. Three independent completion axes

Use these fields independently:

```text
THEORY_STATUS=NOT_ESTABLISHED|CONDITIONAL|VERIFIED_IN_MODEL
HARDWARE_STATUS=NOT_TESTED|PARTIAL|E7_VERIFIED
HANDOFF_STATUS=IN_PROGRESS|REMOTE_COMMIT_VERIFIED|BLOCKED_REMOTE_WRITE
```

`VERIFIED_IN_MODEL` requires O1-O6 closed, an explicit finite program and independently checkable proof artifacts, and no unresolved essential algorithmic assumptions. A finite test corpus or a checklist of PASS strings is not a universal proof. State exactly which machine model was used and which hardware assumptions remain unmeasured. A complete proof need not wait for GPU measurements, but must not be called an actual hardware achievement.

`E7_VERIFIED` retains the existing real-405B, complete <=8-GiB allocation, original-contract and same-machine latency/reproduction gates. Keep empirical E0-E7 and Phase A-D unchanged; theory is an additional axis, not an E7 substitute.

`REMOTE_COMMIT_VERIFIED` means that artifacts are durably recorded and read back. It says nothing by itself about theory or runtime success. Documentation-only governance completion is not scientific progress or a newly achieved theorem.

## 8. Stop a candidate, not the mission

One rejection, auxiliary lemma, passed suite, document update or remote commit is not an automatic stopping criterion for goal-directed research. Within the available session, continue with another credible principle or the surviving construction's next essential obligation.

At a session/resource boundary, preserve partial evidence and report the remaining obligation honestly. Do not hide failure, fabricate PASS, relax the target, demand infinite silent work, or promise background completion. A safety abort or exact fallback prevents silent corruption but cannot be counted as successful fast execution; every admitted case and all fallback tail costs must satisfy the claimed contract.

Report the algorithm/proof, essential obligations actually closed, complete budget and scope first. Put test counts and repository logistics afterward. For incomplete work, identify the decisive missing construction without replacing it with a long inventory of auxiliary achievements.
