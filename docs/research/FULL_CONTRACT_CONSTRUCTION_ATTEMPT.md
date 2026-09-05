# Full-contract construction attempt — 2026-09-05

## Outcome and provenance

The request was to construct an algorithm satisfying **all** fixed VORTEX conditions. No such algorithm was obtained in this attempt. This document is an incomplete research record, not an executable solution, a new impossibility theorem, or a claim of scientific progress.

Remote baseline: `8799b14ff59cc4618314c05ffd40a345a5129caa`, PR #125, branch `research/theory-closure-native-coding-20260905`. The current AGENTS, canonical mission, constructive contract, README and next-obligation file were read from the connected repository. The supplied local-only joint-transition bundle was inspected as previous work; its experiments are not claimed as this attempt's experiments and are not silently merged.

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
NEW_EXECUTOR_IMPLEMENTED=false
NEW_RUNTIME_EXPERIMENTS_RUN=false
```

## Required result, without strengthening or weakening it

For each checkpoint W in the unchanged target class, a uniform finite constructor must produce an executable representation C_W and a causal executor E_W. Under the frozen native execution ABI, every legal input, RNG state and continuation must preserve the exposed outputs and the required successor-state relation.

The memory bound is total peak GPU allocation <=8 GiB, not just resident weights. On the declared same-machine workload population, warm time/token quantiles must satisfy Q50(T_E) <=1.2 Q50(T_4B_Q4) and Q95(T_E) <=1.5 Q95(T_4B_Q4). The existing TTFT condition remains unchanged. Preparation, all storage tiers, instructions, addresses, movement, verification, repair and state costs count.

Correctness is universal over legal continuations. The specified latency tests are quantile tests; one slow input alone does not disprove a frozen population's p95. Conversely, an average kernel-work reduction or a favorable control input does not prove those quantiles. No workload, machine or reference implementation was silently invented to declare success.

## Construction routes examined and why none is a completed source

These are examined routes, not three newly invented accepted principles. No novelty or >=10x target-wide route is claimed.

### 1. Compile the entire native transition into a direct query program

A fully specified but rejected construction clarifies the cost issue. Fix a bounded finite-word native ABI. Let k be the number of bits in its current input, required state and RNG, and let v be the number of output and successor-state bits. Enumerate the 2^k bit patterns and evaluate a totalized bounded native transition, with ABI-defined fault results for undefined operations. This over-approximates reachable states and does not assume a free reachability oracle. Store the result table, or construct a binary decision DAG by merging identical subtrees bottom-up. Query by following the current bits and reading the stored output.

This is finite and does not require a perfect selector. It is not a VORTEX solution. The unreduced table has 2^k entries of v output bits; construction evaluates that total transition on every pattern and incurs a worst-case bound proportional to 2^k times totalized reference evaluation plus output work. Merging identical subtrees supplies no target-wide small-size upper bound. Querying a leaf still has to materialize the required output/state, or pay for a proved virtual representation. The actual bounded native k includes the context state, not merely the next token ID.

No table was built. This route violates the intended bounded preparation/storage budget absent a new, proved compact constructor. Replacing enumeration by 'find the small equivalent program' leaves exactly the unsolved subroutine that the contract forbids. This reasoning does not prove that every possible compiler must have exponential cost.

### 2. Execute a compressed operation grammar rather than individual coefficients

A native ordered operation sequence can be represented compositionally: a terminal is an explicitly specified native operation and a concatenation node denotes function composition in the original order. This preserves semantics if each component is evaluated exactly. However, storing a repeated sequence once does not imply that evaluating that sequence on different input/accumulator states costs one operation. A general interpreter still visits the expanded operations; a compact exact summary needs its own constructive size and query bound.

Research on computation-friendly compressed matrices provides useful direct-evaluation constructions for its stated matrix classes. It does not supply the missing compact native transition summary for every unmodified dense Transformer. Ordinary algebraic reassociation cannot silently replace the reference reduction and rounding. No new small-summary generator was derived here, so this was not implemented as another conditional fast path.

### 3. Recover the whole output/state from a small exact correction code

A decoder can recover an output when supplied with a suitable predictor, bounded residual and exact syndrome. That does not construct a cheap syndrome or prove the residual bound for the unchanged checkpoint/input scope. For a rounded native operator, the code must also represent the actual rounding-dependent result; the algebraic code of an exact real/integer sum is not automatically that code.

Neither a target-wide cheap predictor/certificate nor a native-code generator was obtained. Computing the original output and then its syndrome remains full original work. This route therefore remains unresolved, not an algorithm satisfying the request.

## Whole-budget check

For an actual serial schedule, a conservative time upper bound would have to charge preparation at the declared initialization/amortization horizon, input routing, every cold-memory access, arithmetic, native numerical handling, metadata, output/state materialization, verification/repair and synchronization. A proven overlap schedule could tighten that bound. No numerical upper bound was established for a new complete executable schedule in this attempt.

In particular, none of the routes above simultaneously provides a small representation, a cheap current-input query, exact required state, and target-scale quantile/TTFT closure. This is a statement about the constructions examined, not a lower bound against all algorithms.

## Primary sources actually checked

- Tosoni et al., *Toward Greener Matrix Operations by Lossless Compressed Formats*, arXiv:2409.18620v1, 27 September 2024: https://arxiv.org/html/2409.18620v1 . Its stated experimental scope is sparse Boolean matrices with real-valued vectors; no BF16 dense-Transformer or native-state guarantee is imported.
- Anand, van den Brand and McCarty, *The Structural Complexity of Matrix-Vector Multiplication*, arXiv:2502.21240v3: https://arxiv.org/html/2502.21240v3 . Structural-parameter-dependent algebraic guarantees do not establish a small parameter or native finite-word equivalence for the unchanged VORTEX scope.
- Cankaya, *Bit-Exact AI Inference Verification Without Performance Tradeoffs*, arXiv:2606.00279v2, 5 June 2026: https://arxiv.org/html/2606.00279v2 . The reference execution's reduction, fusion and hardware/software numerical details matter. An exact emulator is not a cheap universal result generator.

Access date: 5 September 2026. These are selected relevant primary sources, not a claim to have exhaustively surveyed all publications or to have reproduced their results. HTML versions were consulted; no PDF, figure or table analysis is claimed.

## Obligations and handoff

O1-O6 remain OPEN for the full mission. No isolated lemma, source review or persistence operation is promoted to closure. The decisive missing artifact is still a concrete constructor/query program that preserves the complete native output/state contract and has a sufficient whole-resource upper bound closing the target budget.

Do not enlarge the earlier synthetic transition corpus or implement the rejected exhaustive table as the next core task. The existing NEXT_EXPERIMENT remains the applicable construction requirement; no new promising survivor was established here. This note does not authorize another audit to substitute for that work.

No model inference, new numerical kernel tests, GPU measurements, target-405B execution, TTFT or baseline measurements, full repository test suite, or GitHub Actions were run. Only this additive documentation change is locally validated. README is updated additively, retaining the previous verified README bytes unchanged as its prefix. Policies, runtime, existing results, research-state statuses and validation claims are unchanged.
