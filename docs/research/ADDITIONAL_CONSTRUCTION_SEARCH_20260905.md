# Additional construction search — no qualifying new core obtained

Date: 2026-09-05. Parent: `e913c9b980a71b63468c7ba9fa9ac2ba65555e6c` (PR #126).

## Outcome

The user asked for actual achievement of the fixed VORTEX mission, not another renamed failed approach. This search did **not** produce an all-conditions algorithm or a completed constructive theory. It also did not produce three qualifying materially new execution principles. No candidate is admitted, no old experiment is repeated, and no new runtime or synthetic experiment is claimed. This is a limited record of an unsuccessful construction search, not scientific promotion.

The exact missing construction is still a finite checkpoint-derived causal program that generates the original output and required successor state while avoiding nearly all original dense coefficient work, with sufficient end-to-end resource bounds. A new name, a small certificate, a literature result, or this commit does not supply that program.

## Fixed contract and acceptance

The mission remains arbitrary public unmodified HF dense 405B-class checkpoints, executor replacement only, batch one, total GPU allocation <=8 GiB, original output/RNG and required successor state, and same-machine native 4B Q4 warm latency p50 <=1.2x and p95 <=1.5x with the existing TTFT requirement. No model-domain narrowing, retraining, weight modification, hidden compute, uncharged preparation or fallback is introduced.

`AGENTS.md`, `MISSION_AND_WORKING_PRINCIPLES.md`, and `docs/CONSTRUCTIVE_THEORY_CONTRACT.md` remain authoritative and unchanged. Full-mission O1-O6 remain OPEN. No theorem is promoted merely because a necessary condition is not disproved.

## Additional primary-source leads and the unresolved construction

### Automatic structural multiplication

Kozma and Opler, *Fast and simple multiplication of bounded twin-width matrices*, arXiv:2602.20023v1, 2026-02-23, Theorems 1.4–1.7:

- A binary matrix with bounded twin-width admits near-linear vector queries after preprocessing, without supplying a contraction sequence.
- The ordered result has O(n^2+dn) preparation and O(dn) query cost.
- The general Hamming-coherence bound has a double-exponential dependence on d in its stated upper bound.

This provides a genuine construction for its stated structural domain. It does not establish small d for every VORTEX checkpoint, or preservation of a deployed floating-point reduction by algebraic regrouping. No such domain-wide bound or native lift was constructed here. Using this as an existing structural tool is not a newly invented VORTEX execution principle; no structural benchmark was rerun.

Source: https://arxiv.org/html/2602.20023v1

### Compositional precision selection

Budzinskiy et al., *LAMP: Look-Ahead Mixed-Precision Inference of Large Language Models*, arXiv:2601.21623v2, 2026-05-07, Sections 1.2 and 2:

The method selects computations to recompute more accurately using downstream numerical sensitivity. Its paper explicitly distinguishes this from weight quantization and states that it does not reduce Transformer memory footprint. The experiments simulate low-precision accumulation; the authors do not report practical kernel runtime comparisons there.

This does not prove bit-identical original logits, RNG consumption and all required successor states. No deterministic exact-native certificate with sufficiently cheap generation was obtained by extending this analysis here. Precision/error-budgeting is not reintroduced as a new core or as a memory solution.

Source: https://arxiv.org/html/2601.21623v2

### Native arithmetic and lower-bound scope checks

Cankaya, *Bit-Exact AI Inference Verification Without Performance Tradeoffs*, arXiv:2606.00279v2, 2026-06-05, describes reconstruction of native numerical behavior under specified hardware/software and reduction conditions. It is a reference-semantics source, not a cheap VORTEX output generator.

Clifford, Gronlund and Larsen, *New Unconditional Hardness Results for Dynamic and Online Problems*, arXiv:1504.01836v1, Section 1.1, gives a finite-field cell-probe tradeoff with explicit space, word-size and field-size dependence. It is not a theorem ruling out all finite-word Transformer executors on the VORTEX hardware contract. No such impossibility theorem is claimed here.

Sources:
- https://arxiv.org/html/2606.00279v2
- https://arxiv.org/html/1504.01836v1

These sources were inspected to distinguish applicable tools from missing constructions, not to relabel previously rejected methods or count source review as algorithmic progress. The search was not an exhaustive review of all literature.

## Hardware and evidence boundary

The existing hardware plan contains historical inventory entries and explicit NOT TESTED performance gates. Reading that document is not a new inventory, remote-machine access, a GPU experiment or a same-machine baseline measurement. No current hardware state or target timing is inferred from it.

No model download, model inference, native kernel replacement, checkpoint parameter survey, GPU profiling, 405B execution, latency experiment or GitHub Actions dispatch was performed in this search. No core execution code was added. Local validation concerns only this documentation addition and exact preservation of the previous README prefix.

## Handoff

Scientific status remains:

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
```

The current `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, validation matrix, old evidence and policies are unchanged because no scientific status or next construction obligation was resolved. README gains an additive link to this record, not a promotion or a replacement of the active frontier. Remote persistence requires separate SHA/tree/blob read-back; it must not be inferred from creating this file.
