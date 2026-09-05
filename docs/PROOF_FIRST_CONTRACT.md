# VORTEX Proof-First Contract

Version: CTC-2026-09-05.

Read [Constructive Theory Contract](CONSTRUCTIVE_THEORY_CONTRACT.md) first. Proof-first now means constructing and closing the final execution theorem, not merely finding necessary conditions that can be checked. Historical versions and their scientific classifications are preserved under `governance/history/pre_ctc_20260905/`; current scientific state comes from pinned evidence and root ledgers.

## Two acceptance tracks

Theoretical acceptance is `THEORY_STATUS=VERIFIED_IN_MODEL` only after O1-O6 are closed: arbitrary-checkpoint construction, complete causal execution, exact native output/RNG/state induction, full resource upper bounds, target-scale baseline/latency closure and independent checkable evidence. A program may be an executable specification rather than an optimized GPU backend, but it may not contain a missing oracle or unbounded synthesis.

Actual mission acceptance remains Phase D / E7: a real unmodified dense 405B, total peak <=8 GiB on one GPU, original contract, same-machine native 4B Q4 p50 <=1.2x and p95 <=1.5x, the existing TTFT protocol and independent reproduction. Theory acceptance neither grants E6/E7 nor requires pretending hardware was measured.

## Required proof package

Freeze the final theorem, legal domains, reference numerical ABI, RNG consumption, baseline implementation, workload population and model/hardware assumptions. Each lemma has hypotheses, dependencies, proof and scope. Link every implementation primitive and resource term to its obligation.

Prove correctness for all admitted inputs, not only the recorded corpus. Changed state representations need an initial-state relation and an inductive transition/observation correspondence for every legal continuation. Token-only or distribution-only agreement does not establish the required same-RNG contract. Prove rounding/reduction semantics; exact real arithmetic or parity is not a native floating-point proof.

Expose constructor, causal selector/address work, native decoder, state transition, certificate, repair and termination costs. Fail-closed fallback/abort is safety, not a valid way to exclude expensive cases from acceptance. Probabilistic errors must be disclosed and union-accounted and cannot replace the fixed exact mission.

## Resource closure

List complete initialization, preprocessing, host/device storage, weights, KV/cache, workspace/fragmentation, metadata, paging, transfers, arithmetic, verification, repair, rollback, fallback and synchronization bounds. Preserve short-session and cold/TTFT costs alongside justified finite amortization.

A lower bound can reject, never certify success. Give a constructive schedule with an upper bound and prove any overlap. Peak bandwidth, free selector work, empirical best cases, the parameter-count ratio or a <=10% traffic fragment cannot close the budget. Quantile ratios require a valid baseline comparison under the same frozen population; see section 4 of the constructive contract.

## Local validation and empirical evidence

Phase A: scoped proofs, executable specifications, bounds, assumptions and counterexamples.
Phase B: independent references, randomized/property/adversarial/boundary tests, fault injection, deterministic replay and checksums. Synthetic success is not LLM success.
Phase C: available unmodified pinned small checkpoints; actual operation replacement in generation for E2, held-out prompts, causal/forward/selector/fallback accounting, token/logit/state agreement, CPU/RAM and scale evidence. Offline observations remain E1. Empirical scaling claims need the same protocol at multiple sizes, not separate truth tables; retain the existing three-size gate where applicable.
Phase D: actual target machine, model, complete memory, raw latency/baseline and relevant hardware profiling. Unavailable fields remain NOT TESTED.

E0-E7 remain empirical levels: idea, reference, small operation replacement, held-out causality, representative hardware, scaling, target memory, final target. Keep MEASURED, DERIVED, PROJECTED and UNVERIFIED separate. Independent proof checking must be possible; mandatory duplicate hosted execution is not imposed.

A favorable oracle is diagnostic only. `REAL_EXECUTOR_ONLY` remains the authoritative empirical arm. A conditional theorem is CONDITIONAL; a scoped toy theorem is not full-theory acceptance. Register thresholds locally before results, persist preregistration and evidence after local validation, and apply the [session protocol](WORK_SESSION_PROTOCOL.md).
