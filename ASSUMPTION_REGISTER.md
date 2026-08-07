# Assumption Register

No unverified assumption may be used as a success condition.

## A-001 — Range-certified signed cancellation is broadly exploitable

EXP-047/047R evaluated about 98–100% of contributions even under exact realized ranges.

Status: CONTRADICTED FOR RANGE-BASED CPTC AS CORE.

## A-002 — Alpha-spending Serfling implementation is valid in declared scope

Reference/property/adversarial checks, zero committed-corpus wrong accepts, zero bound violations, deterministic replay, exact fallback.

Status: SUPPORTED PHASE A/B E1; DOES NOT ESTABLISH SAVINGS.

## A-003 — Certificate overhead is smaller than skipped work

EXP-047 Python path about 8.6–9.1x full sum; EXP-047R C2 reference much slower while reading nearly all contributions.

Status: CONTRADICTED FOR TESTED IMPLEMENTATIONS.

## A-004 — Low-dimensional decision projection is sufficient model-wide

Pairwise LM-head reconstruction is exact; nonlinear model-wide propagation remains unresolved.

Status: PARTIAL / UNVERIFIED MODEL-WIDE.

## A-005 — Probabilistic certification is product-acceptable

No final model-wide delta or product requirement exists.

Status: UNVERIFIED.

## A-006 — Tiny-model trends predict 70B/405B

Three tiny checkpoints only.

Status: UNVERIFIED.

## A-007 — Target RAM/SSD capacity and bandwidth are sufficient

Status: UNVERIFIED; Phase D NOT TESTED.

## A-008 — Target/draft/KV/work state fits 8 GiB

Status: UNVERIFIED; E0; Phase D NOT TESTED.

## A-009 — 4B-class speed coexists with exact fallback

All deployable proposal mechanisms through EXP-050 remain far above 1.185185% target-equivalent traffic.

Status: HIGH-RISK AND UNSUPPORTED.

## A-010 — Auxiliary VM/DAG/certificate/verifier components aid final runtime

Status: OPTIONAL. Reuse only after a new core mechanism survives its own Gate.

## A-011 — Loose range metadata caused CPTC failure

EXP-047R exact realized range median/p90 100%.

Status: CONTRADICTED.

## A-012 — Sound static tile metadata is useful

Soundness passed; usefulness failed with median/p90 100%.

Status: SOUND E1, NOT USEFUL FOR TESTED CORE.

## A-013 — One target stream can verify many exact tokens

EXP-048 future oracle verified 96 exact tokens/one pass =1.0416667%.

Status: VERIFIER ARITHMETIC SUPPORTED E1; CAUSAL PROPOSAL SOURCE UNSOLVED.

## A-014 — Early target layers provide useful recursive draft tokens

EXP-048 max matching prefix 1, p50 committed 1, p90 2893.843%.

Status: CONTRADICTED FOR TESTED PARTIAL-LAYER DRAFT.

## A-015 — Hard Jacobi provides cheap long exact blocks

EXP-048 p50 58 target passes/32 tokens.

Status: CONTRADICTED.

## A-016 — Continuous Picard/Anderson propagates exact causal information faster

EXP-049 favorable p50 4.5, maximum 6, p90 168.778596%; Anderson/Jacobi 0.25x.

Status: CONTRADICTED FOR TESTED TARGET-ONLY FAMILY.

## A-017 — Arbitrary causal target permits universal >1 exact position/round target-only solving

Hidden triangular transcripts remained indistinguishable before predecessor resolution.

Status: CONTRADICTED WITHIN DECLARED BLACK-BOX ROUND INTERFACE.

## A-018 — Fixed target-independent external draft provides long exact prefixes across arbitrary targets

EXP-050 universal first-token counterexample produced matching prefix zero.

Status: CONTRADICTED FOR UNIVERSAL ARBITRARY-TARGET GUARANTEE.

## A-019 — Tested fixed external draft pool is practically useful

EXP-050 favorable exact-reference selection:

```text
p50 prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
Korean and structured JSON coverage false
```

Status: CONTRADICTED FOR TINYSTORIES 1M/3M/8M FIXED POOL.

## A-020 — A 4B external draft can satisfy final budget if exact prefixes are long

PROJECTED:

```text
4/405 + 1/K <=0.01185185185
K >=507
```

Status: ARITHMETICALLY DERIVED; 507-TOKEN CROSS-MODEL EXACT PREFIX CONTRADICTED BY CURRENT POOL AND UNVERIFIED GENERALLY.

## A-021 — A causal target-independent selector can choose a useful external draft

The EXP-050 selector used exact reference and still failed. No deployable selector exists.

Status: UNVERIFIED GENERALLY; IRRELEVANT FOR TESTED POOL AFTER FAVORABLE ORACLE FAILURE.

## A-022 — Final next-token decision becomes suffix-stable after very few target layers

Assumption:

With the exact target greedy prefix fixed, the intermediate hidden state after a shallow block prefix, passed through the original final norm and LM head, already yields the final target token and no later block changes it.

This is different from recursive partial-layer drafting because the input prefix remains exact for every token state.

Status: ACTIVE FOR EXP-051.

Contradiction tests:

- non-deployable suffix-stable oracle median logical fraction >10%;
- p90 >25%;
- median block-depth fraction >10%;
- any required family median stable depth >50%;
- worsening depth with model size;
- output head alone consumes excessive fraction;
- late-decision adversarial residual chain finalizes only at last block.

## A-023 — A sound causal selector can know suffix stability without executing omitted layers

Current evidence: none. Exact-reference suffix-stable depth uses later target outputs and is non-deployable.

Status: UNVERIFIED.

A shallow oracle depth is necessary but not sufficient. A deployable tail certificate must bound all omitted nonlinear attention/MLP residual effects without executing them.

## A-024 — One LM-head probe plus shallow block prefix can approach 4B-class traffic

Assumption:

The target LM head and final norm, combined with a small prefix of block weights, fit below 1.185185% of the full 405B target stream.

Current evidence: no 405B architecture-specific measurement. EXP-051 will measure actual logical head/layer shares only on pinned tiny checkpoints.

Status: UNVERIFIED / PROJECTED ONLY.

A large tied output embedding may set a nonzero traffic floor even at depth zero.

## A-025 — A fixed early-exit depth works for every arbitrary target

Universal risk:

An arbitrary residual network can keep token `a` dominant through every early layer and flip to `b` only in the final layer.

Status: ACTIVE ADVERSARIAL CONTRADICTION TEST FOR EXP-051; EXPECTED UNSUPPORTED UNIVERSALLY.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## A-026/A-027/A-028 — Advice coverage, reuse, circuit compilation

A-026 enumerative exact advice generalizes across unseen families: CONTRADICTED (0% held-out hits). A-027 natural exact states repeat at least 85 times: CONTRADICTED on the corpus (median/max 1/1). A-028 a non-enumerative bit-exact weight-derived circuit remains compact: ACTIVE UNVERIFIED for EXP-053.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## A-029/A-030 — AIG reduction and input-adaptive exact diagrams

A-029 structural hashing alone makes exact dense arithmetic cheap: CONTRADICTED for the registered bounded operators. A-030 a weight-derived reduced ordered decision diagram can trade compile/storage for a short exact input-adaptive query path without exponential growth: ACTIVE UNVERIFIED for EXP-054.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## A-031/A-032 — Decision paths and word-level column aggregation

A-031 an exact reduced ordered diagram yields a universally short path: CONTRADICTED for the registered operator family. A-032 quantized target weight columns contain enough exact repeated/sign-related structure for word-level grouped popcount aggregation to close query and storage costs: ACTIVE UNVERIFIED for EXP-055.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## A-034 — Full-rank real Q4 matrices may still have low exact shift-displacement rank

Status: ACTIVE FOR EXP-059 ONLY. EXP-058 ruled out ordinary exact low rank, not Toeplitz-, Hankel-, or circulant-like full-rank structure. EXP-059 must use exact registered displacement operators and modular certificates; visual banding and approximate spectral decay are not evidence.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## A-035 — Real Q4 matrices may contain enough exact scalar/block zeros for sparse streaming

Status: ACTIVE FOR EXP-060 ONLY. Ordinary rank and displacement rank do not measure zero sparsity. EXP-060 must account for every stored value, index, row pointer, padded scalar in nonzero blocks, and format search. Q4 model-output preservation remains a separate unverified assumption.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## A-036 — Causal dense-projection inputs may contain useful exact zeros

Status: ACTIVE FOR EXP-061 ONLY. Static weight zeros failed, but runtime activations could skip complete weight columns. EXP-061 must measure exact IEEE zero at every registered dense-projection input, separate prefill from warm decode, exclude causal-mask zeros already handled by standard attention, preserve held-out prompt families, and charge activation-index metadata. Near-zero thresholds are approximation and are forbidden.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## A-037 — Non-mask causal attention probabilities may underflow to exact zero

Status: ACTIVE FOR EXP-062 ONLY. EXP-061 found no exact zeros at dense inputs, but softmax can theoretically underflow for sufficiently negative unmasked scores. EXP-062 must exclude causal-mask and padding entries, use exact returned probabilities, compare hooked/attention-output-enabled tokens with reference generation, and charge QK, softmax, Value accumulation, probability scanning, indexes, and the unchanged non-attention model work.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## A-038 — Cached Keys or Key-Value pairs may repeat exactly across positions

Status: ACTIVE FOR EXP-063 ONLY. Identical Key vectors permit one QK score to be copied for every duplicate position. Identical Key-Value pairs additionally permit one probability-times-Value product to be reused when the copied scores produce identical probabilities. EXP-063 must compare exact tensor bit patterns, exclude structurally ineligible local positions, charge cache scanning/hashing/group metadata/copies/additions, and preserve all reference tokens. Approximate vector similarity is forbidden.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## A-039 — Real Q4 output rows may share exact prototypes

Status: ACTIVE FOR EXP-064 ONLY. Identical or sign-related weight rows can share a dot product; rows near an exact prototype may reuse the prototype dot plus an exact sparse residual. Bias additions, row mappings, prototype storage, residual indexes, activation reads and output copies/sign operations must all be charged. Approximate residuals are forbidden.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## A-040 — Real Q4 matrices may have low exact Kronecker rank

Status: ACTIVE FOR EXP-065 ONLY. For every nontrivial shape factorization, rearrange the Q4 matrix so the rank equals the minimum number of Kronecker-product terms over the certified field. Rank certificates may reject a candidate but cannot promote it without an exact integer reconstruction and full operation/storage accounting.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## A-041 — Multi-cut TT/MPO ranks may remain low despite failed one-cut Kronecker rank

Status: ACTIVE FOR EXP-066 ONLY. Pair row and column radix modes, interleave them into MPO physical dimensions, and certify every prefix/suffix unfolding rank. Exact TT/MPO storage is lower-bounded by the certified bond ranks. Mode order search, rank metadata, scales, biases, contractions and intermediates must be charged. Approximate tensor decomposition is forbidden.

<!-- EXP-072A-AUTHORITATIVE-FINAL -->
## A-042 — A self-contained exact Q4 DAG can universally fit the 8 GiB hot envelope

Exact standard-basis outputs recover all coefficients, so artifacts must distinguish every Q4 map. The registered worst-case information is `188.98828125 GiB`, `23.62353515625x` the hot allowance before overhead.

Status: CONTRADICTED FOR THE UNIVERSAL SELF-CONTAINED HOT-ARTIFACT INTERFACE BY EXP-072A.

## A-043 — A cold-backed exact online runtime can meet the final physical query budget

EXP-072A does not cover a runtime that queries original or other lossless cold data. EXP-071 also did not establish the required model-wide finite lower bound. No constructive mechanism has survived the `1.185185%` target-equivalent Gate, and host/storage/transfer budgets are not yet measured on the target.

Status: UNVERIFIED; PRIMARY OPEN EXECUTION CLASS AFTER EXP-072A.

## A-044 — The privately identified Ubuntu target provides a valid same-machine 4B baseline and reproducible storage/transfer envelope

Stage 1 measured the sanitized inventory: Quadro M5000 8,192 MiB, compute capability 5.2, 8,058 MiB free at snapshot, maximum PCIe Gen2 x16, 23.4983 GiB host RAM, one 238.4749 GiB non-rotational ATA block device, and 97.6183 GiB root free capacity. Python 3.12.3 and active Ollama 0.30.6 are present; fio and nvcc are not available. Private identifiers were not retained.

The inventory portion is confirmed. Valid same-machine 4B Q4 latency, storage bandwidth, H2D bandwidth, loaded PCIe state, usable process VRAM, thermal stability, and production-interference controls remain unverified pending separately authorized Stage 2.

Status: PARTIALLY CONFIRMED BY EXP-073 STAGE 1 INVENTORY; BASELINE/PERFORMANCE PORTION ACTIVE AND UNVERIFIED; PHASE-D RUNTIME VALIDATION NOT TESTED.

## A-045 — MTP-1 plus expert paging makes 10B active work behave like 1B

EXP-074 granted free proposal generation and perfect fixed expert reuse. One
accepted token still required `10.0x` the 1B baseline-equivalent target weight
traffic against a `1.2x` p50 allowance.

Status: CONTRADICTED FOR THE REGISTERED MTP-1 PLUS PAGING INTERFACE.

## A-046 — Causal long blocks and router locality can close the revised Gate

An ideal fixed-route, zero-cost proposal reaches the p50 Gate at nine perfectly
accepted tokens. The independent-uniform expected route control requires 98;
0.8B-equivalent proposal work with fixed routes requires 25. No accepted-prefix
or real route distribution has been measured.

Status: UNVERIFIED; REQUIRES METADATA SURFACE AUDIT, SMALL-CHECKPOINT CAUSAL
PROPOSAL EVIDENCE, THEN MIDDLE-RUNG MOE ROUTER TRACES.

## A-047 — The 122B artifact can be downloaded with safe experiment headroom

The 81 GB decimal artifact nominally fits 97.6183 GiB root free space, but the
derived remainder is only 22.1812 GiB and fails the registered 30 GiB safe
workspace Gate.

Status: NOMINAL FIT DERIVED; SAFE-WORKSPACE ASSUMPTION CONTRADICTED FOR CURRENT
CAPACITY; NO DOWNLOAD PERFORMED.

## A-048 — The smallest official Qwen3.5 checkpoint exposes native MTP state

EXP-075 pinned `Qwen/Qwen3.5-0.8B` and found one declared MTP hidden layer plus
the exact registered set of 15 indexed `mtp.*` tensors. Pinned vLLM source also
contains the matching configuration rewrite, registry, loader remap, and
recursive-step surface.

Status: CONFIRMED AT E1 STATIC METADATA/SOURCE INTERFACE ONLY.

## A-049 — The exposed native MTP state produces sufficiently long causal blocks

No MTP weight was loaded and no proposal was executed. Configuration values of
two or four speculative tokens are not accepted-prefix evidence. Repeated use of
one MTP layer, Gated DeltaNet rollback, LM-head cost, rejected suffixes, and
fallback remain unmeasured.

Status: ACTIVE AND UNVERIFIED FOR EXP-076; PRIMARY CHEAP FALSIFICATION.

## A-050 — A supported unchanged small-model runtime can be reproduced locally

The existing local dependency set reports Transformers 4.50.3, whereas the
pinned checkpoint config declares a 4.57.0 development build and its model card
requires a latest implementation. vLLM source support was audited statically
but vLLM was not installed or executed. A clean pinned dependency environment
and exact state semantics are still required.

Status: INFRASTRUCTURE COMPATIBILITY UNVERIFIED; NOT A SCIENTIFIC FAILURE.

<!-- EXP-076-AUTHORITATIVE-FINAL -->
## A-049 closure -- The exposed native MTP state produces sufficiently long causal blocks

EXP-076 executed the pinned native MTP equations causally. With build-selected
`K=4`, held-out accepted-prefix p05/p50/p95 was `0/4/4`, including two
zero-accept cases. The necessary p05/p50 minima were `9/11` after charging the
20,452,864-parameter MTP layer and tied 254,279,680-parameter LM head.

Status: CONTRADICTED FOR THE REGISTERED QWEN3.5-0.8B NATIVE-MTP LONG-BLOCK
SURROGATE.

## A-050 closure -- A supported unchanged small-model runtime can be reproduced locally

An isolated Python 3.12.13 environment with Torch 2.6.0+cpu, Transformers
5.12.0, Tokenizers 0.22.2, Hugging Face Hub 1.26.1, and Safetensors 0.8.0
loaded the pinned unchanged BF16 checkpoint and completed all 24 cases. The
checkpoint manifest and twelve result files passed SHA-256 verification.

Status: CONFIRMED FOR THE EXP-076 WINDOWS CPU REFERENCE ONLY; VLLM, CUDA,
QUANTIZED EXECUTION, AND TARGET-SERVER COMPATIBILITY REMAIN UNVERIFIED.

## A-051 -- A materially different cold-backed information source can meet the final Gate

No currently implemented candidate survives the fully charged route toward
`1.185185%`. EXP-076 closes native MTP under its scope but does not prove all
online cold-backed executors impossible. A future candidate must identify its
new query-time information source and pass E0 before implementation.

Status: OPEN BUT UNSUPPORTED; NO EXP-077 CORE IMPLEMENTATION AUTHORIZED.

## A-052 -- Nonzero activation-conditioned MLP contributions concentrate within 10%

For each causal token, the contribution score
`abs(SiLU(gate) * up) * L2(down_column)` may concentrate strongly enough that
the top 10% of SwiGLU intermediate channels preserves the unchanged target
logit distribution. This is materially different from EXP-061 exact-zero
skipping: no selected or omitted activation is required to equal zero.

EXP-077A tests the favorable ceiling on the pinned unchanged Qwen3.5-0.8B
checkpoint. Full gate/up computation, scoring, and selector cost are free, so a
pass cannot establish a deployable speedup. Failure at this ceiling rejects the
registered activation-norm fracturing path before any larger download.

Status: ACTIVE AND UNVERIFIED; PREREGISTERED FOR EXP-077A E1 ORACLE GATE.

### A-052 closure

The authoritative proposal-conditioned causal-cache replay matched all 192
unmodified target decisions. At `9.988839%` selected MLP parameters, held-out
top-1 agreement was `71.5278%`, mean KL `0.884161`, and p95 KL `3.080752`.
Every family failed its registered top-1 minimum; 20% selected channels still
reached only `83.3333%` population top-1 agreement.

Status: CONTRADICTED FOR THE REGISTERED ACTIVATION-NORM INDIVIDUAL-CHANNEL
FRACTURING ORACLE. OTHER INFORMATION SOURCES ARE NOT DECIDED.
