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

## A-053 -- An exact anchor-conditioned MLP operator remains valid long enough to amortize construction

For a causal anchor `a`, the complete bias-free SwiGLU mapping
`W_down diag(SiLU(W_gate a)) W_up` is exact at `a`. EXP-078A asks whether that
same map remains inside the registered target-logit contract on later causal
activations without reading target future tokens.

Direct materialization costs `H/3` exact active-MLP token-equivalents, while the
combined hidden-by-hidden hot application costs `H/(3 E I)` of exact active MLP
work for `E` active paths. The registered small-checkpoint p50/p05 minimum reuse
spans are `13,821/6,249`; the nine-path 122B screen requires
`115,299/26,354`. These are derived implementation costs, not a universal lower
bound against every possible constructor.

Status: ACTIVE AND UNVERIFIED FOR EXP-078A E1 FAVORABLE LIFETIME GATE. PHYSICAL
CONSTRUCTION, SENTINEL, FALLBACK, LONG-HORIZON SURVIVAL, AND LARGE-MODEL SCALING
ARE NOT TESTED.

### A-053 closure

The unchanged target passed 192/192 baseline decisions. The complete frozen
anchor operator had held-out top-1 agreement `0/126`, valid-prefix p05/p50/p95
`0/0/0`, and mean/p95 KL `14.898423/25.335417`. Every family failed at the first
reuse token.

Status: CONTRADICTED FOR THE REGISTERED UNCHANGED PRIOR-TOKEN OPERATOR. A NEW
CAUSAL DELTA-CONSTRUCTION INFORMATION SOURCE IS NOT DECIDED.

## A-054 -- A correlated cold-backed proof state closes after few row-block reads

For every MLP down projection, a procedural DCT pilot image supplies a small hot
center. The exact non-pilot residual is retained as disjoint row-block L2 balls;
current causal activations trigger complete original row-block reads. The
registered hypothesis is that dual favorable oracles can preserve target logits
and reduce the minimum local sound radius while total logical work/traffic stays
inside `1.185185%` of the complete target.

This directly tests whether correlation-aware deferred residuals change the
near-full refinement premise recorded by F-001/F-004. It does not assume a
deployable selector or nonlinear certificate: those are free favorable grants,
and a pass promotes only the next proof-propagation Gate.

Status: ACTIVE AND UNVERIFIED FOR EXP-079A E1. NO MODEL RESULT, PHYSICAL SPEED,
8 GIB PEAK, 122B, OR DENSE-405B EXECUTION EVIDENCE EXISTS.

### A-054 closure

The unchanged control passed, but the p50 dual-oracle candidate preserved only
`4.1667%` held-out top-1. The separately optimized minimum sound-radius p50/p95
was `48.663918x/57.778748x` the exact MLP-output signal; p95 budget did not
approach closure.

Status: CONTRADICTED FOR THE REGISTERED FIXED DCT PILOT AND ROW-BLOCK L2
PROOF-STATE INTERFACE. OTHER COLD-BACKED QUERY-TIME INFORMATION SOURCES ARE NOT
DECIDED.

## A-055 -- Exact cross-token rectangular arithmetic closes the dense MAC Gate

If `K` exact future activations are available, a subcubic exact multiplication
of `W[m,n] @ X[n,K]` may reduce scalar arithmetic per committed token rather
than merely amortizing weight traffic. EXP-080A grants the future block and one
target sweep for free and charges constructive Strassen multiplications,
additions, padding, and tile accumulation on the frozen 405B shapes.

Status: ACTIVE FOR EXP-080A ONLY. CAUSAL PRODUCTION OF THE BLOCK IS NOT AN
ASSUMPTION OF THIS GATE AND REMAINS CONTRADICTED/UNSUPPORTED BY EXP-048-050 AND
EXP-076 UNDER THEIR SCOPES.

### A-055 closure

The perfect-future grant made traffic negligible, but standard recursive
Strassen reduced the registered dense arithmetic only to `36.111580%` at
`K=16,384`, not the required `1.185185%`. Every smaller block was worse. The
constructive miss is `30.469145x`; the unit-constant `omega=2.371552` pass is a
non-constructive diagnostic and cannot contradict this measurement.

Status: CONTRADICTED FOR STANDARD RECURSIVE STRASSEN WITH THE REGISTERED
RECTANGULAR TILING/CHARGING INTERFACE. OTHER EXPLICIT EXACT LOW-CONSTANT
ALGORITHMS AND A NEW CAUSAL FUTURE-BLOCK SOURCE ARE NOT DECIDED.

## A-056 -- Nonlinear lookup errors occupy a tiny recoverable residual code

After an eight-stage depth-four lookup generator, held-out causal projection
errors are assumed to lie in a rank-eight field code on at least `99.75%` of
weighted calls. A syndrome then reconstructs the complete error and an
independent fingerprint verifies it before commit. The assumption concerns
output-error structure after nonlinear coding, not exact input-span reuse or
approximate low numerical rank.

Status: REJECTED UNDER THE EXP-081A FROZEN GATE. Weighted held-out exact
coverage was `8.681672%` against `99.75%`, with every family below `10.72%`.
The same `54/622` coverage occurred in all six projections and is consistent
with repeated template-prefix state rather than a general residual code.
Observed fallback leaves derived traffic at `92.244986%` of dense. Reopening
requires a new source of residual information, not rank/tree/field/prompt
tuning. Worst-case queries always retained dense fallback; no universal
subquadratic claim was established. Large-model scaling, BF16/Q4 fidelity, and
physical speed remain unverified.

## A-057 -- Real Q4 projections have a very low-weight differential tree

After adding a zero vertex, either the rows or columns of each registered Q4
projection are assumed to admit an exact Hamming spanning tree whose weighted
population cost can meet `1.185185185%` after metadata and traffic. This is
strictly more permissive than assigning rows to at most 32 prototypes: every
vertex may be a reusable parent and chains may be arbitrarily long.

Low VC/Pollard dimension supplies the asymptotic structural hypothesis. A
random dense Q4 matrix is the strongest counterexample. EXP-082A first uses a
certified exact-block lower bound and will not build the tree if that favorable
bound already fails.

Status: ACTIVE AND UNVERIFIED FOR EXP-082A E1. NO SMALL-CHECKPOINT RESULT,
PHYSICAL EXECUTION, LARGE-MODEL SCALING, OR E2-E7 EVIDENCE.

Closure: REJECTED AT E1. The exact 32-coefficient block certificate gave a
weighted best-orientation coefficient lower bound of `1.562367394%` on 21
pinned Q4 projections, already above the `1.185185185%` final allowance while
granting free metadata, values, activations, construction, and traffic. The
assumption that a terminal-only row/column differential tree can meet the
target is contradicted for the measured checkpoint Gate. This does not assert
a universal lower bound for arbitrary synthetic-intermediate circuits or for
all possible models; those require a distinct assumption and Gate.

## A-058 -- Static Synthetic Intermediates create a new target-feasible class

The hypothesis was that allowing compiler-created coefficient vectors, then
placing the resulting exact tree or circuit in cold storage, created a new
execution mechanism beyond EXP-053/054/072/082A with a plausible fully charged
route below `1.185185185%`.

Status: CONTRADICTED AT E0 FOR THE STATIC CLASS. A synthetic Hamming tree is a
restricted static exact linear DAG, and archived EXP-072B already allowed
general shared synthetic forms. Cold placement changes residency only. The
generic Steiner consequence of EXP-082A is merely `0.781183697%`, so no finite
real-weight Steiner impossibility is claimed; instead the proposal fails
novelty and lacks a favorable operation/traffic equation. Almost-all random
hypercube and binary linear-circuit theory is adverse supporting evidence.

A materially different assumption remains unverified: the current causal
activation exposes a Query-Adaptive Cold Source that can select a tiny exact
subset without dense discovery, and all selection, probes, misses, state,
verification, and fallback fit the target. No algorithm or evidence currently
supports that assumption.

## A-059 -- A coded causal cold source crosses the conservation frontier

The remaining hypothesis is that the current committed activation and
unchanged checkpoint expose a lossless query-dependent code or certificate
that determines omitted dense contributions without an equivalent discovery
scan. On held-out causal queries it must close operations and traffic under
`R = g + kappa/N + rho*h + (1-rho)*(m+1)`, peak branch state under 8 GiB, and
the final measured latency contract.

Status: OPEN AND UNSUPPORTED AT E0. The derived best-case floor is
`98.814814815%` exact coverage and `84.375x` useful information amplification;
using the retained six-check verifier makes those `99.081653739%` and
`108.891389x` before any other cost. Raw Q4 page selection with no information
about unread coefficients is contradicted by indistinguishability, but coded
pages or causal certificates are not universally ruled out. No concrete source,
E1 population evidence, operation replacement, or physical result exists.

## A-060 -- Committed-prefix images make residual certification target-feasible

For each unchanged dense projection, exact committed-prefix input/image pairs
are assumed to provide a low-rank causal center whose remaining residual can be
certified after reading only a very small set of cold column pages. The
registered rank-16 screen requires at least `99.899840530%` token-level
certificate coverage after favorable 64-token build amortization, with actual
page-rounded cold work `0.349720584%` and `1,009` page requests/token.

Status: ACTIVE BUT UNVERIFIED AT E0. The algebra and single-projection residual
bound pass synthetic/reference controls. EXP-069 contradicts exact span hits
but does not measure this fully retained residual certificate. EXP-079A's fixed
four-direction radius failure is adverse, while a committed-prefix basis is a
new causal source. An activation orthogonal to the prefix span with aligned
weight-block contributions is the strongest counterexample and forces dense
fallback. Real small-checkpoint residual concentration, end-to-end output
certificate coverage, numerical enclosure, short-prefix behavior, physical
I/O, and E2-E7 are all untested.

## A-061 -- One target-feasible page preserves the first post-prefill decision

After a prompt-only top-16 basis is built, at least one contiguous 64-column
page of each required layer-11 `q_proj` and `down_proj` is assumed to preserve
the unchanged greedy token at the first genuine post-prefill decode call. The
frozen Gate grants the exact current dense image as an illegal center and
enumerates every page, so selector discovery and pair-construction error are
free.

Status: SUPPORTED AT E1 UNDER THE FROZEN FAVORABLE ORACLE ON THE FINITE
PINNED SMALL-MODEL POPULATION.
The finite 18-prompt population permits zero token failure at the derived
`99.899840530%` frontier, requires 3/3 success in each family and 36/36 branch
top-1 agreement, and retains mean/p95 KL `<=0.02/0.05`. Failure rejects the
registered rank-16/page-64 per-projection path before a bound propagator.
Passing does not establish a causal selector, legal `Z` construction, sound
end-to-end certificate, simultaneous composition, scale, or physical speed.

### A-061 closure

EXP-083A passed 18/18 token states, all six families at 3/3, and 36/36 required
projection branches. Mean/p95 KL was
`0.007225545020063708/0.037660752986209814`, controls were clean, and a separate
model replay reproduced the same deterministic core. The assumption as written
-- existence of at least one favorable page per registered branch -- is
supported for this population.

The support does not transfer to an executable selector because the Gate used
native-anchored centers and exact-reference enumeration. A post-hoc
minimum-radius selector failed two tokens. It also does not validate legal
pair-built centers, native outward bounds, multiple layers/positions, or scale.

## A-062 -- A legal causal-pair center and target-free bound retain Gate coverage

Exact committed-prefix pairs are assumed to construct a numerically valid
`Z=WQ` without scanning `W`, and current-input plus static checkpoint metadata
are assumed to choose and certify the required cold page without native current
outputs or candidate logits. The fully charged hit/fallback equation must still
meet the registered coverage frontier.

Status: REJECTED AT E1 UNDER THE FROZEN LEGAL GATE. EXP-083B reached rank 16
and selected its page legally on the first untouched row, but the strict final
certificate was unresolved and exact fallback fired. The required finite
coverage permits no fallback. Controls and independent replay were clean, so
this contradicts the assumption rather than indicating infrastructure failure.

## A-063 -- A common spectral bound can certify the legal last-down slice

For stored pair-only `Q_hat/Z_hat`, the page with maximum current residual
energy is assumed to leave a small enough unread vector that a verified
`beta_W >= ||W||_2`, pair-image defect, native arithmetic enclosure, final
RMSNorm ball, and LM-head row-norm margin certify the unchanged greedy token.

The component equation assumes a verified static spectral compiler amortized
over 20,000,000 checkpoint-service tokens. It requires 24/24 untouched rows,
4/4 per family, zero false accept, zero fallback, and mean/p95 KL
`<=0.02/0.05`.

Status: REJECTED AT E1. On the first frozen row, the verified unread radius was
`16.3298572850`, pair-image radius `6.2376633562`, and total down radius
`23.4205200666`; the candidate pre-RMSNorm norm was only `10.7091120605`.
The strict margin was unresolved and the registered stop rule rejected the
legal Atlas primary path.

### A-063 closure

The failure cannot be attributed solely to the deliberately conservative
RMSNorm implementation term. Post-hoc necessary-condition analysis finds an
actual candidate/native final-hidden separation of `38.9079080403`, while the
candidate top-two logit gap and verified LM row norms admit a hidden-ball
radius below `4.1978252811` even with all rounding deleted. A ball containing
the native state therefore cannot satisfy the same row-norm certificate.

This closure is scoped to the common-spectral, one-page, global-L2-ball Atlas
mechanism. A genuinely new directional/correlated causal information source
is not universally ruled out, but it requires a new assumption and fully
charged E0 Gate rather than a parameter or bound sweep.

## A-064 -- Paid primal/dual images cheaply determine causal decision queries

Assumption: a Causal Decision-Dual Code can use checkpoint-derived `WQ` and
`W^T P` images to determine prompt-dependent signed decision functionals below
the registered target, without an equivalent dense constructor, an enumerated
direction table, or hidden fallback.

Status: REJECTED AT E0 FOR BOTH CURRENTLY CONSTRUCTIBLE VARIANTS. The exact
decomposition retains `r^T W u` outside both paid spans, and two operators can
share all cached images while differing on that term. A dynamic exact dual
direction costs `1.5625%` at 64 tokens before all other work. A favorable
two-byte final-slice vocabulary table is `12.720703125 GiB` and its full scan
costs `6.765979992%` of registered Q4 bytes.

### A-064 closure

The closure applies to forward-pair-only state, raw-checkpoint dynamic dual
construction at the registered service life, and static full-vocabulary dual
scans. It does not establish a universal lower bound for every preprocessed
exact bilinear data structure. The remaining unsupported assumption is that a
finite checkpoint-derived source can answer the Bilinear Cross Residual
`r^T W u` losslessly and sub-densely for arbitrary prompt-dependent residual
pairs. That assumption may not advance to E1 until a complete E0 equation and
an actual construction exist.

## A-065 -- Separable linear covers make every cross residual target-cheap

Assumption: allocating the fixed hot state across arbitrary matrix-local left
and right linear covering codes can make all exact cross residuals fit the
registered `1.185185185%` coefficient budget.

Status: REJECTED AT E0 UNDER THE DECLARED CARTESIAN COEFFICIENT-PROBE MODEL.
The image pair consumes `a*n + m*b - a*b` independent bits. A finite binary
sphere-covering and weighted-state argument gives a favorable minimum raw
cross fraction of `1.52104775688%` even when the entire `8 GiB` state and every
non-cross operation are free.

### A-065 closure

The closure covers arbitrary linear code choices, optimal row/column center
selection, and lossless matrix-local image encoding. It assumes independently
selectable query pairs across matrices and counts coordinate probes; it does
not establish that a real checkpoint reaches those tuples or rule out
nonlinear/global shared advice and word-packed algorithms. The surviving
unsupported assumption must therefore change one of those interfaces and
close a new E0 equation before E1.

## A-066 -- Cross-matrix mixing makes one linear advice state target-cheap

Assumption: one `8 GiB` linear advice state can expose useful high-dimensional
images in many matrix blocks without paying equivalent state or cancellation,
and thereby answer arbitrary independent rank-one residual queries below the
final fraction.

Closure: free projection reuse is contradicted in the declared systematic
linear model. Only shortened block spaces direct-sum; a fixed outside support
adds at most its cardinality in local dimensions. The remaining global
span-to-cover theorem forces `6,407,133` coefficient uses, but that is only
`0.1338958916%` of the p50 allowance and cannot reject target feasibility.

Status: PARTIALLY CONSTRAINED BUT UNSUPPORTED. NO GLOBAL CONSTRUCTION, TARGET
DIRECT-SUM LOWER BOUND, CAUSAL REACHABILITY RESULT, OR PHYSICAL MAPPING EXISTS.
NONLINEAR AND DATA-DEPENDENT STRUCTURES REMAIN UNDECIDED.

## A-067 -- Causal Bilinear Cross Residual queries occupy a tiny exact span

Assumption: after prompt-only primal/dual residualization, held-out causal
query tensors `r tensor u` lie almost entirely in one automatically compiled
factorized span small enough to scan and verify below the final 405B fraction.

The E0 shape threshold is now fixed. With favorable two-byte factors and all
registered metadata/verifier/build costs, the maximum every-token scan is
dimension 23. It leaves room for at most four weighted fallbacks from 36
last-down rows. The finite theorem `misses >= max(0, rank-23)` makes certified
rank 28 the cheapest decisive rejection.

Status: REJECTED AT E1 FOR THE FROZEN AUTOMATIC FACTOR-SPAN LEDGER. EXP-084A
passed 114 controls, observed build rank 24 under all three registered primes,
filled the first-23 exact ledger, and then observed zero exact hits and five
exact misses on the first five held-out rows. The fifth miss exceeded the
four-fallback allowance and triggered the registered stop.

### A-067 closure

Independent zero-forward verification rebuilt every exact witness/residual
coordinate and deterministic core. The run stopped with held-out rank five,
not 28, so it does not reject the best post-hoc 23-dimensional subspace or all
nonlinear/implicit data structures. It does reject the preregistered
calibration-built every-token factor scan and forbids a parameter/population
sweep. Pair extraction, native BF16/Q4 coefficient semantics, earlier layers,
scale, physical latency, and E2-E7 remain unsolved rather than promoted.

## A-068 -- A perfect router turns many small trace ledgers into target coverage

Assumption: nonlinear causal routing over many separately built exact linear
leaves can retain enough distinct Bilinear Cross Residual answers while each
query scans only one target-cheap leaf.

Status: REJECTED AS A UNIVERSAL PRIMARY CORE UNDER THE REGISTERED MATERIALIZED
TRACE-DIRECTION MODEL. A globally independent population contributes at most
one hit per independent stored direction. Build amortization caps the strongest
favorable union at 87,958 directions, versus 20M service queries, and the
full-fallback equation requires `99.999992826%` coverage. The frozen EXP-084A
diagnostic independently shows `0/5` membership even against the complete
24-row calibration span.

### A-068 closure

The router and membership lookup were free, only one leaf was hot, and native
repair and fallback overlap were omitted. The rejection therefore does not
depend on a poor selector. It also does not establish that all actual causal
queries are independent or close implicit nonlinear/cell-probe structures.
The surviving unsupported assumption is now narrower: checkpoint preprocessing
might contain an exact implicit nonlinear answer source whose information is
not a materialized trace basis. No constructor or target equation exists yet.

## A-069 -- Exact algebraic nonlinearity creates adaptive answer information

Assumption: rational arithmetic, exact multiplication/division, comparisons,
and query-dependent checkpoint probes can answer `r^T W u` in a materially
different class from static arithmetic circuits even when the program is
exact on an open query domain.

Status: REJECTED AS A DISTINCT MECHANISM CLASS. A full-dimensional fixed path
computes the same rational bilinear function by identity, and
Baur--Strassen's unit-cost bound maps it to a static `W u` circuit with at most
four times the arithmetic. At the p50 scalar-query allowance this gives only
the containment ceiling `32/675 = 4.740740740...%`, not a proof that any
static circuit meets the final target.

### A-069 closure

The closure covers exact field operations and locally fixed algebraic
branch/probe paths. It does not model finite BF16/Q4 rounding, bitwise or
floor/modular word operations, discontinuous addresses, or unrestricted
cell-probe schemes. The surviving unsupported assumption is now explicitly
finite-word: such discontinuity might encode a non-exhaustive exact source
under full state, table, probe, traffic, verification, and fallback charges.

## A-070 -- Finite-word discontinuity guarantees an arbitrary-checkpoint 2.5% source

Assumption: bounded words, bitwise operations, native rounding, or
data-dependent addresses automatically let preprocessing replace every exact
rank-one numerical query while reading no more than 2.5% of an arbitrary
unchanged dense checkpoint.

Status: REJECTED FOR THE DECLARED EXPLICIT CONSTRUCTORS, NOT AS A UNIVERSAL
IMPOSSIBILITY. The strongest local GF(2) truth table meets 2.5% query traffic
only by storing 8.660768514174031e11 Q4-checkpoint equivalents, using 78-bit
addresses, and amortizing 173,215 dense-equivalent builds per token. Mailman
remains above budget, broadword still scans the payload, Boolean
nonemptiness is not a numerical answer, and finite rounding state is not free
storage.

### A-070 closure

The audit proves that discontinuity by itself is not an information source and
that query-read fraction cannot omit the representation that supplies the
answer. It does not cover all globally nonlocal nonlinear cell-probe schemes.
The surviving unsupported assumption is exactly the Finite-Word Rank-One Probe
Gap: either a fully charged constructor or a correctly scoped lower bound is
still required. Universal 2.5% remains NOT ESTABLISHED.

## A-071 -- A known global nonlinear or modern cell-probe theorem closes 2.5%

Assumption: an existing globally nonlocal nonlinear Boolean data structure or
the newest general cell-probe lower bound can be transferred directly to
reference-exact native numerical `r^T W u` and thereby prove either universal
2.5% feasibility or impossibility.

Status: REJECTED FOR THE AUDITED TRANSFERS. Larsen--Williams relies on
hereditary all-zero rectangles and returns nonemptiness, not a numerical sum.
Any direct exact subrectangle summary is injective by singleton queries and
retains the raw information. Korten--Pitassi--Impagliazzo requires
`k>t*w+1`, while three distinct rank-one queries have an exact XOR dependency
and prevent even 3-wise independence. Whole-MatVec lower bounds lose a factor
`n` when scalarized; the newest dynamic Multiphase theorem has the wrong model
and scale.

### A-071 closure

This closure is deliberately route-specific. The singleton lemma assumes no
raw probes inside the summarized region; the independence argument targets
the full GF(2) rank-one family; scalarization does not rule out a direct scalar
theorem. A different globally coupled adaptive numerical constructor or a new
lower bound covering arbitrary nonlinear 8 GiB advice remains possible in the
formal sense. Universal 2.5% remains NOT ESTABLISHED and universal
impossibility remains NOT PROVED.

## A-072 -- Finite-semiring preprocessing supplies the universal 2.5% source

Assumption: Williams' subquadratic preprocessed MatVec theorem can be mapped
directly to arbitrary unchanged Q4/BF16 Transformer weights, with the lookup
graph fitting the global 8 GiB state and one-vector savings extending to the
registered 32-token block.

Status: REJECTED FOR THE PUBLISHED GRAPH. Charging only the minimum neighbor
pattern payload yields `query/raw=1/b` and `sidecar/raw=K^b/b`. At
theorem-parameter `b=14`, the favorable model-wide one-bit sidecar is
`55,292.57 GiB`; Q4 and BF16 query payloads are already `2.629552%` and
`10.518207%` of DFloat. The publication gives no 32-query shared-probe bound.
Native BF16 and FP32 addition also fail associativity, so their reference
accumulation is not the finite semiring used by the proof.

### A-072 closure

The closure charges only the published direct graph and direct numerical
relabelings. It deliberately omits costs in the graph's favor and therefore
does not reject a different scalar-specific globally adaptive structure. A
new finite transition monoid would need its own complete representation and
native-order equation. Universal 2.5% remains NOT ESTABLISHED and universal
impossibility remains NOT PROVED.

## A-073 -- Global advice can be evenly divided for CKL tile bounds

Assumption: the global 8 GiB nonlinear advice may be divided by the number of
Transformer matrices or full hidden-square tiles, after which CKL's
single-matrix rank-one lower bound may be applied and the independent
worst-case probe counts summed.

Status: CONTRADICTED. For independent `N`-bit matrices, their `N`-bit XOR is
global advice that recovers any selected matrix once all other matrices are
known. Hence each conditional information term is `N` and their sum is `mN`,
although advice entropy is only `N`. No per-matrix division follows. The
published single-matrix theorem also does not compose independently selected
hard queries into one causal model execution.

### A-073 closure

The counterexample does not make advice free: exploiting it requires knowing
or probing the other matrices. A future direct-sum theorem may succeed by
charging those probes jointly. The current finite screens show only that the
naive route is invalid and far too weak, not that universal 2.5% is feasible
or impossible. The general nonlinear numerical rank-one gap remains open.

## A-074 -- A synergy-charging all-linear bound transfers to rank-one

Assumption: once arbitrary nonlinear advice and cross-matrix probes are
jointly charged, the complete Walsh-character fiber argument can be applied
unchanged to `r tensor u` queries and 32-token rank-one tuples.

Status: CONTRADICTED. Full Walsh characters restricted to any advice fiber
span all functions on that fiber. Rank-one characters form a much smaller
column set and need not span it. Registered query descriptions upper-bound
the column count by `2^39,254,528` for one independent tuple and
`2^1,256,144,896` for 32 tuples. The corresponding low-degree dimension
screen is already nonrestrictive at `0.028137293%` of coefficients.

### A-074 closure

The all-linear theorem remains valid and is not demoted. The closure targets
only its unsupported rank-one transfer and query-cardinality route. A future
proof must establish target-scale rank or adaptive-tree complexity for the
restricted character matrix on every large nonlinear advice fiber. Universal
2.5% remains neither established nor disproved.

## A-075 -- A native-exact local shortcut can replace the old 2.5% premise

Assumption: after dropping the fixed evidence-ratio hypothesis, exact
rounding absorption, activation history, product reuse/cancellation, or the
low-VC/Pollard structured-MatVec theorem can independently supply enough work
reduction to meet the unchanged-result, 8 GiB, and 20 ms/token contract.

Status: CONTRADICTED FOR THE SCREENED CONSTRUCTORS. Favorable observed local
elimination uppers are all at most `5.6920%`, versus `98.8173%` required before
metadata, verification, and traffic. The Pollard extension sees at least 653
unique BF16 values per column and 1,276 per row. Even granting `d=1` and free
logarithms/constants leaves a `35.6027%` dense-work ratio.

### A-075 closure

This is not a claim that no exact runtime exists. It closes only the named
local information sources and the published low-VC/Pollard constructor on the
pinned real tensor. A materially new globally coupled exact answer source may
still reopen feasibility. The former `2.5%` number is no longer a premise.

## A-076 -- BF16 layer boundaries make Atlas state sparse to repair

Assumption: an Atlas-plus-one-page layer output lands in the same BF16 word as
the native output for almost every coordinate, so native rounding can lock
most state coordinates exactly and dense repair is needed only for a small
remainder.

Status: CONTRADICTED FOR THIS CONCRETE PREDICTOR. A free oracle comparison on
the frozen EXP-083B row locks `1/1,024` down-projection coordinates,
`7/1,024` post-residual coordinates, and `2/1,024` post-RMSNorm coordinates.
The earliest repair point therefore leaves `99.90234375%` of rows unlocked,
far above the complete compute-only allowance of `1.18270517%` before any
certificate, selection, state, or traffic cost.

### A-076 closure

This closes ΩROUNDLOCK only with the Atlas-plus-one-page predictor and one
dense-row repair per unlocked coordinate. It does not reject a materially
different predictor or a globally coded exact repair mechanism. ΩBACKCUT was
also screened but reduces to the already registered decision-dual
`r^T W u` source, so it is a duplicate rather than a new assumption.

## A-077 -- Direct spiky incidence fits every bounded-word checkpoint

Assumption: every registered `N x N` checkpoint matrix has a sum-of-spiky
representation whose total active row/column factor incidence fits the
complete favorable work allowance, so arbitrary `r^T W u` queries can be
answered by direct factor dot products.

Status: CONTRADICTED FOR THE UNIVERSAL CLAIM. The model-wide allowance is
`L=4,800,000,000`. A finite block-mask count and Warren sign-pattern bound
permits at most `2^184,958,474,578.07` representations, versus
`2^403,747,897,344` bounded-word sign checkpoints. The proof grants optimal
masks, sparse supports, arbitrary cross-matrix components, ignored invalid
cross cells, real factors, associative arithmetic, and all other costs free.

### A-077 closure

This closes the direct masked-factor evaluator only.  A different globally
adaptive nonlinear index over the factors is outside the proof.

## A-078 -- Entrywise-power roots create a distinct exact execution source

Assumption: a low-rank root followed by an integer entrywise power can answer
the scalar without reducing to the rejected low-rank runtime family.

Status: CONTRADICTED ALGEBRAICALLY.  The exact multinomial expansion contains
`binom(r+p-1,p)` separable terms.  At `N=16,384` the complete favorable cap is
96 expanded terms; the execution normal form is static low rank.  The EPMF
magnitude factorization result also does not supply the arbitrary sign matrix
or native accumulation order.

## A-079 -- A literal representative table covers all fast rank-one repairs

Assumption: nonlinear routing to cached representative queries can keep every
raw difference inside the complete target budget while materializing one
exact scalar per representative in the allowed state.

Status: CONTRADICTED FOR THE LITERAL TABLE. A fixed-right-vector query packing
requires more than `2^13741.2548` representatives at `N=16,384`; 8 GiB stores
only `2^36` one-bit entries, and the minimum address is 13,742 bits. The Gate
permits arbitrary joint centers and exact minimal difference repair.

### A-079 closure

A compressed nonlinear function of the representative index is not a literal
table and remains the open general decoder problem.

## A-080 -- A trapdoored mask transfers fast multiplication to any checkpoint

Assumption: sampling a fast trapdoored `R` makes `Wx` fast through
`Wx=(W+R)x-Rx`.

Status: CONTRADICTED AS AN INFORMATION SOURCE. The trapdoor evaluates `Rx`,
not `(W+R)x`; the latter is an arbitrary dense matrix product. The published
family accelerates matrices sampled together with their trapdoor and supplies
no arbitrary-checkpoint compiler or native rounded accumulation proof.

## A-081 -- The average-distance premise requires common advantage on every row

Assumption: Hirahara--Shimizu's OMv premise can be screened by requiring one
positive prediction advantage on every output row.

Status: CONTRADICTED. The paper averages normalized Hamming distance over
output coordinates. Concentrating the complete `8 GiB` one-bit grant into
exact rows and guessing the rest gives `58.510196237313%` average GF(2)
coordinate accuracy, even though most rows have no advantage. The former
Parseval calculation remains correct only for the stronger common-row model.

### A-081 closure

The concentrated-row counterexample satisfies only the accuracy clause; its
direct dense row products are not near-linear. `OMEGA-ROWLOTTERY` is separately
rejected because exact recovery needs a rank-covering set of row forms whose
direct coefficient payload is at least one full binary matrix source. A
different compressed cold-backed oracle remains open and must pay every state,
probe, repeated call, decoder, verifier, fallback, and native numerical lift.

## A-082 -- Rank-amplified short representatives can remain disjoint

Assumption: after choosing one weight-at-most-`t` representative for every
rank-one query, every rank-`r` matrix may require the full `r*t` support
because one minimal decomposition can keep all representative supports
disjoint.

Status: CONTRADICTED for the registered first three fixed-linear survivors.
Admissible term pairs form the rank-`r` anti-flag graph.  Its least
eigenvalue, the exact minimum support activity of distinct representatives,
and XOR multiplicity accounting force four cancellations.  `30 x 40`,
`31 x 39`, and `30 x 44` then fail exact determinantal capacity.

### A-082 closure

This does not contradict all fixed-linear covers.  `31 x 43`, `S=1,559`,
`t=15` still passes the present necessity at ratio
`3.409163670153519...`; no atoms or decoder are known.  Adaptive finite-word
addresses and nonlinear output operations remain outside the assumption.

## A-083 -- A one-shot support-overlap charge cannot be reused

Assumption: after the biorthogonal Gate finds a decomposition with support
overlap, the proof must stop because selecting the overlapping terms changes
the remaining problem.

Status: CONTRADICTED for the fixed binary atom model. A positive summed
internal-edge bound supplies an adjacent pair `q,q'` sharing an atom. The
pair belongs to a minimal biorthogonal decomposition, so subtracting it leaves
rank `r-2`. The residual matrix is covered by the same global dictionary and
the same radius-`t` representative promise. Consequently

```text
Delta_r >= 2 + Delta_(r-2).
```

Together with the one-shot knapsack, this forces cancellation 2, 4, and 6 at
ranks 18, 20, and 22 for `31 x 43`. Radius 324 has exact determinantal ratio
`0.4293893830152846...`, so that former frontier is impossible.

### A-083 closure

The theorem does not cover adaptive word addresses or nonlinear output
decoding. The new first fixed-linear survivor is `31 x 42`, `S=1,523`,
`t=15`, with best necessary-condition ratio `38.10086306517199...`; it has no
atoms or decoder.

## A-084 -- Value-adaptive extension-field addresses evade sparse support

Assumption: after packing an `m x k` binary matrix into `GF(2^m)^k`, a decoder
can avoid a short field-span representation by choosing every next linear
summary from values returned by earlier probes and applying arbitrary exact
post-processing.

Status: CONTRADICTED for deterministic exact block-local field-linear cells.
On the zero source every returned cell is zero, fixing one adaptive path `J`.
Every perturbation in the common kernel of those cells follows the same path,
so exactness forces the queried direction into `span_GF(2^m)(J)`. Field
extension preserves binary independence, and hence

```text
sum_(j=0)^t C(S,j) 2^j >= 2^k
```

is necessary. For `k=16,384`, `S=19,172`, the first count-feasible radius is
3,421, not the logical target 194 or the four-Q4-lane traffic grant 776.

### A-084 closure

Arbitrary allocation among block-local field-linear dictionaries also fails:
`1 <= p log2(2e lambda/p)`. Giving all DFloat11 bits plus 8 GiB to the one-bit
surrogate still requires at least `p=10.988948%`, versus `4.740741%`. The
closure does not cover nonlinear stored cells, cross-matrix mixed cells, or
source-dependent nonlinear hot advice.

## A-085 -- Independent packed linear bits evade field-span locality

Assumption: a physical word containing 64 unrelated binary linear summaries
can evade the extension-field support theorem because one read exposes an
arbitrary 64-dimensional binary space and later addresses may depend on all
returned bits.

Status: CONTRADICTED for block-local linear words. On the zero source, the
adaptive support is fixed and the query lies in the span of the exposed
summaries. A support of `j` words has dimension at most `64j`; the exact
maximum Segre intersection of such a subspace gives

```text
sum_(j=1)^p C(L,j) M(a,b,min(64j,ab))
    >= (2^a-1)(2^b-1).
```

For `31 x 42`, seven words cover at most `0.19975212628...` of the rank-one
population by this favorable union bound, so at least eight words are needed.

### A-085 closure

Across all side-`1..128` rectangles, the best four-Q4-lane traffic point is
`118 x 128`, 22 words, `11/472`, still `7425/3776` times the registered line.
This does not cover nonlinear word contents or global cross-matrix encoding.

## A-086 -- Nonlinear cell contents evade determinantal rank amplification

Assumption: replacing stored linear atoms by arbitrary nonlinear checkpoint
functions and choosing addresses adaptively invalidates every rank-amplified
capacity inequality.

Status: CONTRADICTED for arbitrary deterministic block-local cells. A depth-
`t` decision tree is a cylinder polynomial of cell-support degree at most
`t`. Products of `r` rank-one parity characters give rank-at-most-`r`
characters at degree at most `rt`. Their linear independence forces

```text
RankLeq(a,b,r) <= sum_(j<=rt) C(S,j)(A-1)^j
```

for alphabet size `A`. At `31 x 42`, arbitrary nonlinear bit encodings still
need at least 15 probes.

### A-086 closure

For 64-bit nonlinear words the theorem isolates, but does not construct, a
gap. `25 x 108` with 50 padded words and two probes is the smallest side-128
capacity point at the traffic line. The systematic `2 x 3` plus one arbitrary
advice-bit seed is exactly impossible at two probes; the fully non-systematic
seven-bit case remains unresolved after an explicitly inconclusive SMT
timeout. Global mixed cells, native arithmetic, and joint batches stay open.

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:3b9534800671e17eaef74869cf792186e683d6de -->
## Fixed-public dynamic executor assumption audit

- No synthetic checkpoint weights are accepted as scientific evidence.
- Exactness is STATE-EXACT for the executable cache/RNG state tested; no bisimulation shortcut is claimed.
- Physical saving includes artifact bytes, resident replaced-layer bytes, and peak hot projection bytes, or requires both measured p50 and p95 latency improvement.
- TARGET-W performance is not inferred from DEV-W.

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

<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:7777eda66232c6a37c39f4e55f5b9ce56032fe92 -->
## Fixed-public dynamic executor assumption audit

- No synthetic checkpoint weights are accepted as scientific evidence.
- Exactness is STATE-EXACT for the executable cache/RNG state tested; no bisimulation shortcut is claimed.
- Physical saving includes artifact bytes, resident replaced-layer bytes, and peak hot projection/tile bytes, or requires both measured p50 and p95 latency improvement.
- Output-row streaming does not split the reduction axis; actual DEV-W STATE-EXACT execution remains the fail-closed authority.
- TARGET-W performance is not inferred from DEV-W.

## Fixed-public dynamic executor hosted result — commit `7777eda66232c6a37c39f4e55f5b9ce56032fe92` / run `32219219400`

- Verdict: `REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4`
- DEV-W: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; resolved SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `meta-llama/Meta-Llama-3.1-405B-Instruct@f9801cba95a53242b3cc928a4a418d12571d1c5f`; metadata access=True; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed=True.
- Frozen workload SHA-256: `f52dad873703170179b08bf2c48637da2256c2db53ffd245e724c6fa91bb40e6`; 128 decode steps per workload.
- Actual layer tensor audit: `9` parameter tensors, `7080192` reference bytes.
- `checkpoint_mlp_output_row_streamed_lossless_existing_isa`: G2=True, G3=True, G4=True; output-row-tile=128; artifact=4223092 B; reference-layer=7080192 B; compiled-layer-resident=1771776 B; peak-hot=393216 B; baseline p50/p95=113.832 ms/203.051 ms; candidate p50/p95=151.764 ms/241.453 ms.
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.

<!-- EXP085A_PREREGISTERED -->
## EXP-085A frozen assumptions

- `A-085A-1`: additive intermediate macro-pages can expose enough exact BF16 rounding closure before nearly all pages are read. **UNVERIFIED**.
- `A-085A-2`: an output-row/page bound table is small enough for the complete 405B 8-GiB ledger. **DERIVED BY THE GATE, NOT YET RUN**.
- `A-085A-3`: output-row splitting preserves gate/up BF16 rows and the down FP32 accumulation envelope contains the official result. **TO BE CONTROL-TESTED**.
- `A-085A-4`: oracle failure is sufficient to reject this page-separable fingerprint, but not every globally coupled nonlinear exact compiler. **FIXED SCOPE**.


<!-- EXP086A_GLOBAL_BF16_SUCCESSIVE_REFINEMENT -->
## EXP-086A frozen assumptions

- `A-086A-1`: low BF16 mantissa streams can often remain unread while the
  complete unchanged SwiGLU output stays bitwise exact. **UNVERIFIED**.
- `A-086A-2`: one global gate/up precision plus row-adaptive down precision is a
  favorable ceiling for the fixed MSB-prefix source. **FROZEN**.
- `A-086A-3`: zero-order prefix entropy grants ideal coding with coder, framing,
  random access, decode, metadata and kernel costs free. **NOT A RUNTIME**.
- `A-086A-4`: rejection closes only the fixed significance-prefix source; a
  cross-weight context model or nonlinear exact generator is different.

<!-- EXP087A_PREREGISTERED -->
## EXP-087A frozen assumptions

- `A-087A-1`: a complete SwiGLU BF16 word residual is a reusable degree-two Boolean function of twelve checkpoint-static input predicates on unseen causal states. **UNVERIFIED**.
- `A-087A-2`: four non-state-keyed programs can cover every held-out state without dense fallback. **UNVERIFIED**.
- `A-087A-3`: the target-shape packed residual sidecar fits below 4 GiB and its favorable packed-word query work is below the MLP target fraction. **DERIVED, NOT PHYSICAL**.
- `A-087A-4`: failure of the target-seeing oracle rejects this exact finite-context quadratic fingerprint but not every non-polynomial or recurrent transition compiler. **FIXED SCOPE**.
