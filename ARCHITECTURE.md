# VORTEX Architecture

## Native maps with collisions

[Native fiber theorem](experiments/codex_native_fiber_20260909/REPORT.md): exact input/output relabeling preserves
the fibers of the native map. Full-rank algebraic rank-normal form is therefore
not enough to implement an injective native coordinate copy. A constructive
encoded architecture needs a noninjective operator or explicit side state plus
a continuation proof, all paid. This adds an O3 constraint; it does not reject
general joint graph encodings or establish any target resource closure.

## Implicit checkpoint-program carrier frontier — 2026-09-09

See `experiments/implicit_program_carrier_20260909/REPORT.md`.

The architecture search has now made three previously hidden program carriers
explicit:

```text
P1 ADDRESS CARRIER
  query -> checkpoint-independent local parity tables
  fixed logical (row,block) reads
  checkpoint -> alias translation relation
  result: arithmetic cheap; explicit translation representation too large

P2 QUERY-TIME PROGRAM SYNTHESIZER
  checkpoint row-block patterns -> Patricia trie
  query -> traverse shared checkpoint pattern structure
  result: arithmetic/event count cheap; edge-label program traffic source-scale

P3 FULLY PAID ENCODED GRAPH NODE
  checkpoint W -> A W B = J_r
  linear node -> rank-normal cheap map
  nonlinear node -> explicit E_out N(E_in^-1 z)
  result: literal transformed AND restores removed dense maps
```

These failures tighten the open object. It is no longer enough to say “the
selector is implicit,” “the program is structurally shared,” or “stay encoded.”
The representation must reduce the checkpoint-dependent information **actually
consumed per token**, including the route/program/nonlinear operator that tells
the machine what to do.

The next architecture axis is dynamic multi-query information sharing:

```text
G0 = Compile(checkpoint)
(answer_t, G_{t+1}) = ExactUpdateQuery(G_t, causal_query_t)
```

where `G_t` is finite paid state and both the query and state update may change
dense information. This is admissible only if arbitrary legal dense right-factor
changes can update `G_t` subdensely. Literal `y'=y+W(s'-s)`, sparse-delta luck,
response caches and repeated-query tables remain closed.

### Concurrent native-gauge refinement

See [experiments/codex_gauge_transport_20260909/REPORT.md](experiments/codex_gauge_transport_20260909/REPORT.md).
The concurrent construction adds two finite mechanisms that narrow P3's scope:

```text
permuted dense layout + inverse-column leaf schedule -> original reduction tree
opaque-word butterfly E -> E(native_mul(E^-1 z, E^-1 w))
```

The first shows that native finite-word coordinate permutation must transport the
original logical leaf/reduction schedule; naive physical reordering is not exact.
It retains every original coefficient/product/addition. The second proves that a
non-coordinatewise opaque-bit encoding can have an explicit exact transformed
native Hadamard in `O(n log n)` XOR work plus `n` original products. Therefore
the P3 failure above is specifically the **per-operator rank-normal + literal
factorized transformed Hadamard** realization, not a quadratic lower bound for
all transformed gates.

The combined missing object is stricter: find a checkpoint-dependent encoding
`E` with both a cheap exact transformed native graph and, crucially, a cheap
arbitrary dense projection `E F_W E^-1` without executing `F_W` in full. Neither
concurrent mechanism supplies that projection. Per-output bilinear ranks may not
be summed as a lower bound when intermediate products are shared.

## Implicit nonlinear direct-query frontier — 2026-09-09

See `experiments/implicit_nonlinear_direct_query_20260909/REPORT.md`.

The current architecture search now distinguishes the following executed paths:

```text
P1 query-addressed image source
  checkpoint W -> all local linear images Y[B,p]
  current right-factor block bits -> direct image addresses
  selected images -> XOR -> exact GF2 Wv
  status: exact arbitrary-GF2 construction, target storage/query traffic rejected

P2 inverted exact-transition source
  checkpoint column -> exact value classes
  current x_j -> one leaf product/value
  row members -> original ordered accumulators
  strengthened: group by (accumulator,weight) online
  status: exact declared ABI, universal >=90% route rejected

P3 safe encoded coordinates
  z=P h; W'=P_out W P_in^-1
  keep elementwise/Hadamard primitive coordinatewise
  status: only aligned coordinate permutations survive; dense support unchanged
```

P1 is important because it is the first current-frontier producer whose runtime
does not fetch a checkpoint-dependent row/program stream: the **query is the
address source**. Its failure moves the architecture boundary from “invent an
implicit address rule” to a stricter object: the representation must avoid both
a complete image dictionary and a quadratic source-dependent instruction stream.

The open architecture is therefore:

```text
Compile(all arbitrary native checkpoint bytes)
  -> compact/nonlocal finite representation G

Address(G, current causal state/right factors, prior returned words)
  -> subdense paid probes

Decode(probes, current state)
  -> exact native ordered dense effects
  -> exact logits/KV/cache/RNG successor
```

It must not rely on duplicate value/state transitions or a gauge that leaves
nonlinear operators free. If coordinates are transformed nontrivially, the
conjugated nonlinear/residual/cache operators themselves are part of the core
algorithm and their constructor/storage/arithmetic must be charged.

## Direct global producer frontier — 2026-09-09

See `experiments/direct_global_producer_20260909/REPORT.md`.

The current architecture search now distinguishes three source paths:

```text
REJECTED reconstructive path
  global code -> reconstruct most/all original weights -> original dense kernel

REJECTED independent-witness path
  exact mathematical sum -> supposedly cheap native rounding witness
  (rounding gadget shows witness may contain arbitrary MatVec information)

OPEN direct-query path
  arbitrary checkpoint -> implicit globally nonlinear representation G
  current causal factors -> adaptive bounded word probes
  -> exact native ordered dense effects directly
```

The direct-query path has a new global guardrail. Arbitrary nonlinear cells may
mix all 883 matrices; on a final route for simultaneous full-Mv tuples their
component masks satisfy `sum_i m_i*k_i<=t*w`. This avoids the invalid operation
of dividing 8 GiB advice per matrix. The registered lower bound is nevertheless
far below the target and is not an execution architecture.

An executable GF(2) control now exists:

```text
Compile(W) -> reduced R=E W + row-XOR program
Query(v)   -> Rv -> reverse row-XOR program -> Wv
```

It proves that the missing producer can be stated as an actual finite program
rather than a placeholder, but this implementation retains about 50% work on
its frozen adversary and has adverse program traffic. It is therefore not
admitted. The next architecture must make the direct-query representation
implicit/nonlinear enough that its source-dependent program bits are not fetched
at roughly one bit per original coefficient, while still providing a finite
compiler and native decoder.

## Current causal/global frontier construction — 2026-09-08

See `experiments/causal_global_bridge_20260908/REPORT.md`.

Three bounded modules now exist, none admitted as the universal core:

```text
ordinary HF Llama causal exposure
    basis/GQA token -> native v_proj -> required DynamicCache.values

restricted compiled producer
    binary v_proj columns -> 64-bit packed source + RMSNorm/RoPE metadata
    token address -> exact logits/K/V/RNG successor without dense v_proj call

causal sign/delta adversary
    legal signed-basis token blocks -> independently selected right factors
    -> binary lm-head signed sums -> exact GF(2) row/left parity decoder
```

The restricted producer proves that causal exposure alone is not a runtime
lower bound: special coordinate-query structure can be compiled cheaply. The
sign/delta adversary proves the opposite temporal warning: a dynamic state may
face densely changing legal right factors, so literal source-column delta
maintenance is not a subdense architecture.

The active missing interface is therefore stronger than either bounded module:

```text
Compile(all unchanged checkpoint bytes) -> globally paid representation G
Address(G, current causal input/state) -> finite source-dependent addresses
Decode(G[addresses], input, state) -> exact native dense effects + successor
```

It must preserve native ordered Q4/BF16/FP32 behavior, avoid full scans under
dense right-factor changes, account globally mixed advice without an assumed
per-matrix split, and close all whole-model memory/traffic/arithmetic costs.

The producer may not be implemented indirectly as a Boolean-semiring oracle plus
an unspecified exact lift. The post-persistence Boolean gate proves that a
deterministic black-box Boolean `Mv` -> GF(2) conversion needs at least
`|supp(v)|` complete products, and that one arbitrary-nonlinear Boolean feature
product needs exact feature dimension `2^d-1`. A surviving architecture must
probe/construct a direct GF(2)/native representation or exploit a materially
different global nonlinear source, with those internal bits and costs explicit.

## Current bounded native-global construction — 2026-09-08

See experiments/native_global_transition_20260908/REPORT.md. A/C retain original
ATen matrix kernels and checkpoint references; B calls original body between exact
KV codec boundaries. These are real HF CPU reference bridges, not admitted core
architecture. No whole-model compressed executor, tensor-free source, native-kernel
oracle, or target-cost success is implied. All earlier architecture entries retain
their historical scope; fixed mission and CTC govern acceptance.


## Mission boundary

VORTEX is a runtime, not a retrained target model. It ingests an unmodified supported Hugging Face dense checkpoint and automatically constructs runtime state/metadata.

## Target execution stack

```text
unmodified Hugging Face target checkpoint
        |
        v
checkpoint inspector and automatic runtime compiler
        |
        v
causal executor
        |
        +--> proposal, early-exit, or operation-skipping source
        +--> exact/probabilistic certificate or verifier
        +--> safe commit OR exact correction/fallback
        |
        v
memory scheduler
        - target/draft hot state
        - KV and work buffers
        - RAM/SSD cold state
        - all probes, proposals, verification, and fallback charged
        |
        v
original-target-compatible output contract
```

## Core requirement

The primary runtime must causally reduce or amortize original 405B weight traffic and arithmetic on unseen prompts. Correctness-only components remain auxiliary.

## Mandatory interfaces

### Operation-reduction source

Produces a proposal, intermediate decision, exact execution capsule, or early-exit candidate. It reports causal inputs, target/draft/layer/probe counts, target-specific preprocessing, future information, state bytes, selector cost, and rejected work.

### Certificate/verifier

Declares one contract: deterministic exact, deterministic top-1, bounded logit, probabilistic top-1, or exact block longest-prefix verification.

An exact-reference oracle depth/proposal is not a deployable selector.

### Correction/fallback

Any uncertainty, mismatch, corrupt metadata, or numerical failure executes the exact target work required by the reference contract. No silent approximation.

### Memory/evidence

Separate target/draft weights, KV, work, probe heads, metadata, verification, and fallback. Every run emits revisions, calls/layers/bytes, exact output agreement, selector/oracle labels, timing, memory, and raw checksums.

## Auxiliary components retained

- exact/checksummed mmap pointer VM;
- bounded exact compiler/DAG components;
- CPTC certificate/fault rejection/exact fallback;
- exact block verifier;
- Picard/Anderson numerical reference;
- adversarial triangular and first-token constructions.

## Rejected core mechanisms

See `FAILED_APPROACHES.md`.

Closed through EXP-050:

- range-based partial-sum certification;
- hard target-only Jacobi;
- recursive partial-layer draft;
- target-only continuous fixed-point proposal;
- fixed target-independent external draft as arbitrary-model core;
- tested TinyStories cross-checkpoint pool as practical restricted core;
- proposal-tree continuation from failed single-path sources.

## Closed EXP-050 external-draft path

```text
prompt -> external model cached greedy proposal
       -> one exact target block pass
       -> exact prefix + correction
```

MEASURED favorable result:

```text
p50 matching proposal prefix 0.5
maximum 3
p90 normalized fraction 163.20987654%
matching prefix zero 72/108 rows
Korean/JSON coverage false
```

Universal first-token counterexample returned prefix zero.

Decision: reject core; retain only accounting and verifier references.

## Active EXP-051 layer-finalization architecture

EXP-051 changes the skip axis from future token positions to target layer depth.

```text
exact greedy prefix/current token
        |
        v
embedding
        |
        v
block 1 -> h_1 -> final norm + LM head -> z_1
block 2 -> h_2 -> final norm + LM head -> z_2
...
block L -> h_L -> final norm + LM head -> exact z_L
```

Oracle definitions:

```text
first_match = earliest d with z_d = z_L
suffix_stable = earliest d with z_j = z_L for every j >= d
```

The suffix-stable oracle gives the strongest favorable layer-tail skip. It uses later layer results and is not deployable.

### Logical traffic

For depth `d`:

```text
B_oracle(d) = B_current_embedding_rows
              + sum_{j<=d} B_block_j
              + B_final_norm
              + B_lm_head

fraction(d) = B_oracle(d) / B_full_target_token
```

This favorable accounting pays one LM-head probe and assumes the correct depth is already known. A real selector/certificate can only cost more.

### Oracle Gate before engineering

If suffix-stable oracle median bytes exceed 10%, p90 exceeds 25%, or median block depth exceeds 10%, no tail selector/certificate backend is built.

### Universal boundary

A valid residual target can preserve token `a` through all early layers and add a final residual that flips to `b`. Thus no fixed early depth is universally exact for arbitrary targets.

Empirical oracle results are separately used to decide whether a restricted adaptive certificate deserves work.

### Sound certificate stage

Forbidden until the oracle survives. It must bound omitted attention/MLP residual effects on the final token without executing skipped layers. Intermediate token equality or multi-layer stability alone is not a certificate.

## Resource equations

```text
M_total = M_target_hot + M_kv + M_work + M_probe + M_metadata + M_fallback
B_total/token = B_executed_layers + B_probe + B_selector + B_fallback
C_total/token = C_executed_layers + C_probe + C_selector + C_fallback
```

Target conditions:

```text
M_total <=8 GiB
B_total/token <=1.2 * B_4B
C_total/token <=1.2 * C_4B
```

PROJECTED:

```text
405B Q4 stream 188.592821 GiB
1.2x 4B allowance 2.235174 GiB/token
required fraction 1.185185%
```

Hardware terms remain `NOT TESTED`.

## Safety rule

No path commits outside its declared exact verifier/certificate. Invalid depth, metadata, proposal, probe, selector, or numerical state triggers exact completion/fallback or abort.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## Closed EXP-052 and active EXP-053 architecture

EXP-052 exact witnessed tables are `exact hit OR exact target fallback` auxiliary memoization. EXP-053 compiles bounded quantized target weights and exact arithmetic semantics into a structurally hashed bit-vector/AIG decision circuit. Compile time, nodes, bytes, query touches, reduction, and fallback are mandatory costs.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## Closed EXP-053 and active EXP-054 architecture

EXP-053 exact AIGs are auxiliary bit-level reference machinery. EXP-054 compiles immutable weights into a reduced ordered multi-terminal decision diagram using Shannon branching, exact residual arithmetic states, unique-table reduction, and a fixed weight-derived variable order. Runtime evaluates one root-to-terminal path; compile-state visits, nodes, bytes, query probes, order-search cost, and fallback are mandatory.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## Closed EXP-054 and active EXP-055 architecture

EXP-054 ROMTDD/ROBDD-like diagrams are auxiliary exact decision references. EXP-055 keeps signed modular score arithmetic at word level: compile input columns into exact vector signatures, group identical and optional exact-negated signatures, compute group popcounts, and add scaled score vectors. Group build, bit scans/popcounts, vector arithmetic, bytes, selector metadata, and fallback are mandatory.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## Exact-rank boundary

Modular rank certification is retained as an offline structural audit. Conventional exact low-rank factors are prohibited for matrices certified full rank. The next permitted algebraic route is a different full-rank structured representation, beginning with EXP-059 shift-displacement operators.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## Displacement-structure boundary

The runtime must not promote the tested zero-fill/cyclic diagonal/anti-diagonal generator route for matrices with full certified displacement rank. Certificates remain offline audits. The next permitted exact structural path is sparse streaming conditioned on measured Q4 zero scalars or complete zero blocks.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## Static-zero sparsity boundary

The runtime must not promote static CSR/run/BSR streaming for the measured Q4 population. Exact sparse formats remain conditional auxiliaries. The next permitted sparsity route is causal activation-column skipping based only on exact runtime zeros and fail-closed dense fallback.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## Activation-zero boundary

The runtime must not scan ordinary dense-projection inputs for exact zeros on the measured architecture because the observed population is empty and scanning adds work. Observation hooks remain auxiliary for architectures with explicit hard-zero nonlinearities. EXP-062 is restricted to post-softmax attention probabilities, excluding mask zeros.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## Attention-probability zero boundary

The runtime must not scan post-softmax probabilities for exact zeros on the measured architecture. Structural mask zeros remain a standard attention optimization and are not VORTEX evidence. EXP-063 may inspect cached K/V bit equivalence but must fail closed to ordinary attention when no exact group exists.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## Cached-KV equivalence boundary

The runtime must not maintain exact K/KV grouping on the measured architecture. EXP-064 may compile static output-row prototypes from Q4 weights, but must fail closed to dense row evaluation whenever exact accounting is not favorable.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## Output-row structure boundary

The core runtime must not assume row identity or sparse prototype deltas for generic dense checkpoints. The exact row compiler remains fail-closed auxiliary. EXP-065 may only promote a Kronecker path after certified low rearrangement rank and exact integer reconstruction.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## Kronecker structure boundary

The core runtime must not assume a short exact sum of Kronecker products for generic Q4 dense weights. The modular certifier remains auxiliary. EXP-066 may only advance TT/MPO candidates after all bond-rank witnesses and favorable full accounting pass.

<!-- EXP-066-072A-AUTHORITATIVE-CATCHUP -->
## Current exact-representation boundary

EXP-066 through EXP-070 close classical TT/MPO, exact joint-row reuse, absolute-unread demand bounds, causal exact temporal replay, and short-block local-pattern tables under their registered scopes. EXP-071 does not prove the broader online runtime impossible.

EXP-072A adds a universal boundary for a self-contained hot artifact:

```text
activation x
    -> fixed interpreter + complete checkpoint-derived hot artifact
    -> exact Q4 linear-map output
```

Because standard-basis outputs recover the coefficient matrix, this artifact must encode the arbitrary map injectively. Worst-case 405B Q4 information is `188.98828125 GiB`; it cannot universally fit 8 GiB. Nonlocal arithmetic-DAG synthesis is therefore prohibited as a universal self-contained hot core and retained only as a restricted auxiliary.

The only logically open exact representation interface after this Gate is cold-backed:

```text
8 GiB hot state + activation
    -> fully charged probes into original/lossless cold checkpoint-derived state
    -> exact operation or fail-closed completion
```

No tested scheduler closes its query traffic, arithmetic, latency, or fallback. EXP-073 Stage 1 now fixes the target capacity/link inventory without making private connection information architecture state:

```text
GPU total / free snapshot         8,192 / 8,058 MiB
GPU compute capability           5.2
PCIe current / maximum           Gen1 x16 / Gen2 x16
host RAM total                   23.4983 GiB
local block total                238.4749 GiB, non-rotational ATA
root free capacity               97.6183 GiB
```

The current root free capacity cannot hold the registered `188.9883 GiB` packed 405B-Q4 information, even before scales and overhead. The favorable Gen2 x16 ceiling is `7.4506 GiB/s`, a `25.3656 s` floor for moving that byte-equivalent once. These are capacity/link deductions, not measured traffic. A future cold-backed architecture must either provide separately authorized larger cold storage or a lossless representation inside the measured capacity, and must still prove a query schedule far below a full checkpoint transfer.

Stage 2 must measure actual storage, H2D, loaded-link, and native 4B Q4 behavior before a new physical latency Gate is frozen. It remains separately authorized.

<!-- EXP-074-AUTHORITATIVE-FINAL -->
## Weight-stationary block boundary

The restricted Qwen3.5 surrogate candidate was:

```text
checkpoint-native causal MTP block
    -> unchanged target block verification
    -> group positions by exact routed expert
    -> read each required cold expert page once per block
    -> commit exact longest prefix plus correction
```

This separates three resources:

```text
residency: layer/expert pages may keep peak VRAM below total checkpoint size
traffic:   target pages are amortized only across actually accepted positions
compute:   every active expert multiply remains charged even when weights reuse
```

EXP-074 rejects the MTP-1 plus paging form: it retains `10.0x` the 1B target
traffic under an optimistic fixed route and free draft. The runtime must not
equate disk fit, layer paging, or batched GEMM utilization with 1B-class token
latency.

The reference block scheduler remains auxiliary. Reopening requires a pinned
unchanged checkpoint with exposed native MTP, causal accepted-prefix evidence
long enough to close fully charged p50/p95 equations, and measured router-union
locality. No page scheduler or GPU backend is authorized before those Gates.

For dense 405B, even a zero-cost perfect proposal requires 85 accepted tokens
at the p50 allowance; a 4B draft requires 507. Qwen-specific MTP/router behavior
does not supply a universal dense proposal source.

<!-- EXP-075-AUTHORITATIVE-FINAL -->
## Native MTP surface boundary

EXP-075 confirms that the selected official 0.8B checkpoint has the static
pieces required to instantiate this auxiliary path:

```text
unchanged checkpoint config: one native MTP layer
unchanged weight index:       15 mtp.* tensors
pinned vLLM source:           config rewrite + registry + loader + step reuse
```

The next executable reference must keep four states distinct:

```text
committed target state
proposal-only recursive MTP state
verification target state
post-rejection restored target state
```

Proposal generation may consume only the committed prefix and hidden state.
Target future tokens may be used only by the evaluator/verifier, never as MTP
inputs or variant selection. A mismatch commits at most the exact matching
prefix plus exact correction, discards later proposal state, and restores the
hybrid attention/KV state bit-for-bit or fails closed.

Static key and loader presence does not establish those semantics. No page
scheduler, quantized converter, or GPU backend may be attached until causal
accepted-prefix and rollback tests pass. The architecture remains Qwen-specific
auxiliary screening and does not solve proposal generation for arbitrary dense
checkpoints.

## EXP-078A frozen tangent-macroblock reference boundary

`vortex_runtime/tangent_macroblock.py` contains the pure direct-construction,
charged-cycle, lifetime, aggregation, and decision equations. The throwaway TUI
in `vortex_runtime/tangent_macroblock_prototype.py` exposes the complete cost
state after every user action. The heavyweight reference monkey-patches only the
unchanged checkpoint's MLP executor.

```text
exact prompt prefix
    -> capture complete post-SiLU gate coefficient at the final prompt token
    -> reuse that coefficient in every layer for seven causal positions
    -> preserve candidate recurrent/KV divergence
    -> compare target and candidate logits
```

The reference executes the factorized `up -> frozen coefficient -> down` form
to measure quality. It does not physically construct or accelerate the logical
hidden-by-hidden matrix, implement a sentinel, certify an error bound, restore
an exact cache after divergence, or provide fail-closed deployment. Derived
construction cost and E1 quality observation must remain separate from physical
performance claims.

### EXP-078A closure

The reference boundary was valid, but the state transition failed immediately:

```text
exact anchor -> frozen complete MLP map -> first later token -> quality failure
```

All held-out valid-prefix lengths were zero. The frozen map is therefore removed
from the active architecture before any materializer, sentinel, cache repair, or
kernel is built. A future delta-updated map must be treated as a new component
with an explicit causal delta source and fail-closed state-repair transition.

## EXP-080A Hyperblock arithmetic boundary

The candidate changes only the linear-operator scheduling axis:

```text
non-deployable exact future block X[n,K]
    -> exact tiled rectangular W[m,n] @ X[n,K]
    -> constructive Strassen arithmetic count
```

It does not contain a proposal generator, verifier, correction path, KV state
machine, or deployable future source. Traffic amortization is `1/K` for one
perfect target sweep. Arithmetic reduction is accepted only from fully charged
constructive multiply/add/padding/accumulation counts; a matrix-multiplication
exponent with unit constants is diagnostic.

Failure removes standard Strassen Hyperblocks from the core before a kernel is
built. Passing would authorize only a lower-level exact packed constructor Gate
and would leave the causal-sequential barrier unchanged.

### EXP-080A closure

The exact-control layer passed, but the registered component chain terminates at
the arithmetic Gate:

```text
perfect future X -> K=16,384 -> traffic/workspace upper bounds pass
                                -> standard Strassen arithmetic = 36.111580%
                                -> final allowance = 1.185185%
                                -> reject component
```

Standard recursive Strassen is therefore not part of the active runtime
architecture. The unit-constant `omega=2.371552` row remains a theoretical
target, not a callable component. No proposal, verification, correction,
commit, KV-repair, or exact-fallback transition was added. A future replacement
must expose both a constructive low-constant arithmetic interface and an
independently justified causal source before it can enter the architecture.

## EXP-081A provisional lookup-recovery boundary

```text
current causal x
  -> nonlinear lookup candidate G(x)
  -> residual syndrome recovery
  -> independent field fingerprint
       pass -> commit exact integer projection result
       fail -> original W @ x fallback
```

The fingerprint never manufactures a result; it only rejects an incorrect
candidate. The component is absent from the active runtime until real held-out
recovery coverage clears the fallback equation. Approximate corrected outputs
are diagnostics and may not enter the exact branch.

### EXP-081A closure

The reference transition is algebraically valid and passed all 321 controls,
but it does not enter the runtime architecture. Held-out exact coverage was
only `8.681672%`; charging fallback leaves logical traffic at `92.244986%` of
dense. The six projection rows all accepted `54/622` positions, consistent with
repeated causal prefix state rather than a reusable general residual code.

Retain `syndrome_lookup.py` only as an auxiliary exact candidate verifier and
throwaway research reference. There is no decoder hook, packed projection,
cache transition, scheduler, GPU kernel, or active SRLM component. A future
decision-certificate design must expose explicit proposal, interval state,
page-selection, certificate, commit, and unchanged-fallback interfaces before
architecture admission.

## E0 proof-carrying boundary

Random linear checks may validate a supplied target trace with a small hot
sidecar, but no trace proposer enters the architecture. On one machine the
proposer must still produce every claimed linear result; a dense proposer plus
verification is above 100% of target work and traffic. External proof workers
are outside the no-added-hardware mission. The verifier remains an auxiliary
interface only.

## EXP-082A provisional differential-tree boundary

```text
pinned Q4 matrix
  -> exact row-tree or column-tree compiler
  -> sparse parent-child integer deltas
  -> exact causal x evaluation
       verified tree -> candidate Q4 projection
       malformed/missing state -> unchanged dense Q4 fallback or abort
```

No compiler or runtime component is active yet. Stage 1 contains only a
certified lower-bound analyzer; rejection removes the component before tree
construction. A pass would still require exact reconstruction, direct MatVec
equality, metadata/traffic/storage accounting, build amortization, and later
Q4 output-contract validation before architecture admission.

### EXP-082A closure

The analyzer rejected the provisional component: its favorable terminal-tree
coefficient lower bound is `1.562367394%`, already above the final budget.
Therefore no differential tree compiler, delta store, traversal executor, or
hardware path enters the architecture.

An architecture with synthetic intermediate coefficient vectors would be a
different exact linear circuit rather than this terminal MST. It remains only
an E0 question and must first prove that it does not restate the already closed
exact dictionary/DAG/circuit mechanisms.

## E0 synthetic-circuit closure and provisional query-adaptive boundary

No synthetic tree, generic linear-DAG compiler, cold circuit interpreter, or
delta store enters the architecture. A synthetic edge
`z_v = z_u + (v-u).x` is a static linear-circuit instruction, and general
synthetic sharing was already represented by archived EXP-072B. Cold placement
does not alter that class.

The only adjacent provisional interface is deliberately narrower and has no
implementation:

```text
current committed activation x + unchanged checkpoint
  -> causal query selector
  -> bounded set of lossless cold page/operation identifiers
  -> exact page execution
  -> fail-closed verifier
       verified result -> candidate projection output
       miss/failure     -> charged unchanged dense fallback or abort
```

Selector work, index/state bytes, page probes and payloads, intermediate
values, verification, misses, fallback, compile/build amortization, KV/runtime
state, and every resident byte are mandatory. The interface is not an admitted
Core Candidate until an E0 equation and concrete causal information source
both exist. Authority:
`docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

### Query-adaptive equation boundary

The provisional interface now has a fixed accounting contract but still has no
component implementation. For operations and traffic independently:

```text
R = common + compile/service_tokens
    + coverage*hit
    + (1-coverage)*(miss + dense_fallback)
```

At the p50 logical Gate, a free path needs `98.814814815%` coverage; retaining
the known verifier traffic needs `99.081653739%`. A raw page selector cannot
enter the architecture because it does not determine unread contributions. The
only open subtype is a Coded Causal Cold Source carrying query-selected
checkpoint-derived aggregates or certificates with at least `84.375x`
best-case useful information amplification (`108.891389x` after the verifier).

Peak state is evaluated per mutually exclusive hit/fallback branch under 8 GiB;
cold capacity, build amortization, page metadata, selector scans, and physical
request/transfer latency remain separate mandatory terms. No selector, page
format, decoder, verifier hook, cache transition, fallback engine, or kernel is
admitted until a concrete causal source passes E0. Authority:
`docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`.

## Provisional Causal Residual Atlas boundary

The first concrete Coded Causal Cold Source is admitted only to Gate design:

```text
exact committed prefix pairs (X, W X)
  -> causal orthogonal code Q and image Z = W Q
current committed activation x
  -> a = Q^T x; residual u = x - Q a
  -> center Z a
  -> score/reveal exact cold column pages for W u
  -> outward residual enclosure for every unread page
       declared output certified -> commit
       unresolved/corrupt/nonfinite -> exact dense completion or abort
```

Unlike `OnlineAtlasLinear`, a nonzero residual is not accepted by tolerance.
Unlike EXP-079A, the basis is formed from the current request's committed
prefix rather than fixed DCT directions. Unlike raw page selection, `Z` and
the page bounds carry checkpoint-derived information about omitted work.

No runtime component enters production architecture yet. The favorable
rank-16 screen reserves `0.980995178 GiB` for the two-byte capsule and scans
`1,009` 64-column pages/token. The `0.980995178 GiB` stored capsule incurs
`1.366004944 GiB/token` logical reads because `Q` is used twice. Amortized
traffic/operation fractions are `1.085025716%/0.557958575%`, requiring
`99.899840530%` certificate coverage.
KV, workspaces, numerical-enclosure state, fallback overlap, page service, and
allocator headroom are still mandatory and absent. Only a preregistered pinned
real-weight favorable-oracle Gate may advance this provisional boundary.
Authority: `docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`.

### Preregistered first-decode Gate boundary

The first checkpoint Gate does not admit an Atlas component into the runtime.
It evaluates only this favorable isolated transition:

```text
exact prompt-only rank-16 basis
  + first post-prefill current activation
  + illegal native-anchored exact center
  -> enumerate one 64-column page
  -> patch one layer-11 q_proj or down_proj
  -> unchanged dense suffix and final logits
```

The exact-reference evaluator chooses the top-1-preserving minimum-KL page.
It is not an architecture interface. The current dense output, all page
weights, all other model operations, and page discovery are granted free.

The per-token fast branch exists in this Gate only when both registered
projection branches survive. Eighteen of eighteen token states and every one
of 36 branches must preserve top-1. A failure removes the frozen rank-16/
page-64 path before any legal pair center, bound propagator, selector, cache
transition, fallback engine, scheduler, or kernel is added. A pass authorizes
only those next correctness interfaces under a new Gate.

Authority: `docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`.

### EXP-083A measured boundary

The favorable isolated transition now has real-weight evidence: all 18 token
states and all 36 registered q/down branches contained at least one one-page
patch that preserved native top-1, and the complete Gate reproduced. This
admits **page existence** as an observed property of the pinned population; it
does not admit the evaluator as an architecture component.

The next architecture boundary is deliberately split:

```text
committed prefix pairs only -> Q and legal Z=WQ
current committed x + static metadata -> target-free page choice
legal center + revealed page -> outward output enclosure
  certified -> candidate output
  unresolved/corrupt/nonfinite -> charged exact completion, fallback, or abort
```

No current component implements those interfaces. The exact-reference oracle,
native dense center, exhaustive page logits, and one-projection-at-a-time dense
suffix remain experiment-only. A naive minimum-radius chooser failed two token
states, so it is not admitted. Hardware scheduling and E2 integration remain
blocked behind a separately preregistered correctness Gate.

### Preregistered legal last-down boundary

The next Gate freezes one architecture-shaped transition without admitting it
as a runtime component:

```text
committed prefix (x_s, y_s) pairs
  -> two-pass causal Q_hat/Z_hat + outward pair defect
current layer-23 down input
  -> u = x - Q_hat(Q_hat^T x)
  -> choose maximum-residual-energy 64-column page
  -> Z_hat a + selected exact page contribution
  -> verified unread spectral radius
  -> residual add + final RMSNorm radius + LM-head row margins
       strict winner -> evaluator-checked candidate
       unresolved/corrupt/nonfinite -> immutable dense completion or abort
```

The selector has no weight/output/logit input. The static operator certificate,
pair construction, capsule, proof metadata, vocabulary scan, and fallback are
present in the logical equation. The first valid fallback among 24 new prompts
rejects the path.

This boundary deliberately starts at the final `down_proj`; it does not yet
propagate through attention, DeltaNet, earlier MLPs, multiple projections, or
cache divergence. A pass admits only a backward-layer and position-expansion
Gate. A failure admits no selector rescue. Authority:
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`.

### EXP-083B architecture closure

The preregistered transition is not admitted into the runtime. On the first
untouched row, legal pair construction and target-free page selection worked,
but the outward declared-output certificate required dense completion. The
zero-fallback architecture boundary therefore failed before any backward-layer
composition or E2 integration.

The decisive interface mismatch is now measured:

```text
current residual u                    15.0834838983 L2
selected 64-column residual            8.7190044575 L2
unread residual                       12.3081455865 L2
verified unread radius                16.3298572850
complete down radius                  23.4205200666
candidate pre-RMSNorm center norm     10.7091120605
```

No `CausalResidualAtlas` selector, capsule, proof compiler, cache transition,
fallback engine, scheduler, or kernel is promoted to production architecture.
The favorable EXP-083A oracle remains evidence of page existence only. Any
future architecture proposal must introduce a different causal information
source that changes how unread contributions are known, and must pass a new E0
resource equation before an E1 or E2 component boundary is opened.

## Post-Atlas decision-direction architecture boundary

No decision-dual component is admitted. The exact candidate interface would
have to preserve this boundary:

```text
checkpoint-derived paid primal state  -> WQ
checkpoint-derived paid dual state    -> W^T P
current causal activation/direction   -> a, b, u, r
exact signed result                   -> two cached-span terms + r^T W u
missing/corrupt/uncertified cross term -> unchanged dense fallback or abort
```

The final line is not an implementation detail. A dynamic exact constructor
for a fresh dual direction is dense-equivalent under the audited route, while
the static vocabulary composite exceeds both state and scan budgets. The
runtime therefore contains no `DecisionDirectionalSource`,
`CausalDecisionDualCode`, exact-MIPS index, backward proof compiler, or hybrid
Atlas/dual scheduler.

A future architecture proposal may cross this boundary only by naming the
lossless checkpoint-derived representation that supplies `r^T W u`, defining
its causal build/query/fallback interfaces, and passing complete E0 accounting.
Approximate scoring may be auxiliary only if an independent sound exact path
is charged and fail-closed.

### Separable cross-residual architecture closure

No `MatrixLocalSeparableResidualCode` component is admitted. Generalizing the
two cached spans to arbitrary matrix-local linear covering codes still leaves
coordinate residuals `e,f` and the exact term `e^T W f`. Under an all-query
Cartesian coefficient-probe interface, the complete `8 GiB` hot grant cannot
reduce that raw work below `1.52104775688%`, already above the full p50 target.

The rejected boundary is:

```text
matrix-local row/column linear codes -> cached A^T W and W B images
arbitrary independent query tuple   -> nearest centers + sparse e,f
exact missing result                -> raw e^T W f coordinate probes
```

A future architecture must cross a materially different boundary: genuinely
nonseparable/global advice with charged cancellation and query localization,
or a causal restriction on reachable residual pairs established without
future leakage. Merely increasing basis dimension, changing linear codes, or
moving the same image state to cold storage remains rejected.

### Joint global-advice localization boundary

No global linear-advice component is admitted. Its exact allowable interface
would be:

```text
complete checkpoint w -> one hot linear advice z=G w, rank(G)<=8 GiB bits
block-local rank-one q -> choose a in rowspace(G)
raw cold residual      -> e=q+a, including every outside-block cancellation
exact answer           -> <a,w> + <e,w>
```

Only `rowspace(G) intersect V_i` is usable in block `i` without outside raw
coefficients. A fixed outside set exposes at most one extra local dimension per
coordinate. Query-dependent supports remain a valid but unconstructed degree
of freedom; they cannot be replaced by summed projection dimensions.

The general registered lower bound is `6,407,133` coefficient uses, far below
the `4,785,160,264.82` p50 allowance. Consequently neither an impossibility
boundary nor a runnable component follows. There is no compiler, representation
builder, selector, cold layout, decoder, verifier, fallback engine, state plan,
or physical scheduler for a target-feasible global code.

The next architecture boundary is evidence-only: define whether actual
unchanged causal residual pairs occupy a certified restricted query set. It
must be leakage-free and pre-registered before any new trace/model execution;
it cannot enter the runtime until a complete E0 equation and E1 population
Gate survive.

### Causal Bilinear Span Ledger provisional boundary

No ledger component is admitted into the runtime. The screened interface is:

```text
calibration causal residual pairs -> factorized B-query basis + scalar answers
current exact pair (r,u)           -> factor scan + exact rational witness
  witnessed and independently checked -> combine cached answers
  nonmember/corrupt/nonfinite           -> weighted dense fallback or abort
```

On the registered 405B geometry, a full two-byte factor scan reaches at most
`B=23` after metadata, six verification checks, and amortized build. Its common
traffic/operations are `1.159413233%/0.281295494%`; `B=24` fails traffic before
fallback. The component state is `2.104879502 GiB`, excluding KV, workspace,
fallback overlap, and the source of current pairs/results.

The exact rank-to-miss transition is `misses >= max(0,R-B)`. The frozen last-
down Gate has 36 rows, permits only four registered-down-weighted misses, and
rejects at certified rank 28. Until that population survives, there is no
ledger compiler, numerical decoder, pair extractor, trace proposer, executor
hook, cache transition, scheduler, or kernel.

Even a population pass does not admit the component. A general internal
decision residual requires a paid nonlinear pullback, while linear combinations
of native rounded scalar answers require an exact/outward numerical contract.
Those two interfaces remain mandatory and unimplemented.

### Trace-built query-adaptive code-union closure

No `TraceBuiltBilinearCodeUnion` component is admitted into the runtime. Its
strongest favorable interface was:

```text
calibration traces        -> many exact linear leaves + cached scalar answers
current causal pair       -> free perfect router selects one leaf
exact leaf membership     -> scan only that leaf and combine cached answers
nonmember                 -> unchanged dense fallback or abort
```

This interface is nonlinear in leaf selection, but its answer information is
still the collection of materialized leaf basis directions. A globally
independent population can hit no more queries than the independent directions
built across the union. The registered ceiling is 87,958 directions at
`b=1`, covering only `0.43979%` of 20M independent queries, while required
fallback coverage is `99.999992826%`. The frozen EXP-084A rows also show that
partitioning the full 24-row build span produces zero hits among all five
stored evaluation rows.

Repeated-query or tightly clustered leaves may exist as an auxiliary cache,
never as the exact primary executor. The next architecture boundary requires
an automatic **implicit nonlinear checkpoint source** whose constructor and
query equation expose where exact `r^T W u` information resides without
materializing trace basis directions. No such compiler, representation,
selector, decoder, verifier, fallback engine, scheduler, or kernel currently
exists.

### Exact-field nonlinear path-collapse boundary

No generic algebraic decision-tree or rational-circuit component is admitted.
On an open exact-field domain, one fixed full-dimensional path computes the
complete `r^T W u` rational function; reverse differentiation exposes a static
`W u` arithmetic circuit with constant-factor operation overhead. That is an
archived static arithmetic-DAG interface, not a new query-adaptive source.

The only open architecture slot is therefore a `FiniteWordDiscontinuousSource`
whose exactness and advantage depend essentially on bounded native words:
bitwise packing, modular/floor behavior, native rounding, or discontinuous
addresses. No interface is admitted until its finite alphabet, word model,
constructor, persistent layout, table bound, probe schedule, traffic,
verification, miss/fallback path, and numerical contract close at E0.

### Finite-word rank-one probe closure boundary

No bounded-word rank-one source is admitted into the runtime. The screened
local interface was:

    unchanged matrix block -> all GF(2) two-sided pattern answers
    current left/right bits -> one data-dependent word address per block
    queried entries         -> exact XOR reduction

The equation is exact and discontinuous, but target-fitting query traffic
forces exponential table storage and construction. At 2.5%, the favorable
64-bit layout needs a table 8.660768514174031e11 times the Q4 checkpoint and
78-bit addressing. At the registered 8/675 line, it needs
3.449500717947148e18 times the checkpoint and 100-bit addressing. No compiler,
artifact, loader, decoder, verifier, fallback engine, cache transition,
scheduler, or kernel is admitted for this interface.

Mailman, broadword full scans, Boolean nonemptiness probes, and a free native
rounding table are also excluded as primary information sources. The only open
architecture boundary is the Finite-Word Rank-One Probe Gap: a globally
nonlocal nonlinear exact data structure must first close its complete E0
equation, or a theorem must close that full model. Neither exists.

### Global nonlinear frontier boundary

No `ZeroRectangleNumericSource`, `LimitedIndependenceRankOneIndex`, or
`WholeMatVecScalarOracle` component is admitted. Larsen--Williams stores
globally chosen all-zero rectangles and can answer Boolean nonemptiness with
subquadratic cell probes, but no decoder equation maps that bit to exact
signed/Q4/BF16 rank-one arithmetic. A direct exact-aggregate rectangle summary
would reveal every singleton cell and cannot compress an arbitrary block.

Korten--Pitassi--Impagliazzo's premise is structurally false for the complete
rank-one query code: three queries generated by `u`, `v`, and `u+v` have an
identically zero answer XOR. Whole-output MatVec theorems provide at most an
`n`-fold-weaker scalar consequence. The 2026 dynamic Multiphase result has no
static free-advice mapping.

The only admissible future architecture change remains one uniform,
checkpoint-only compiler with a concrete globally nonlocal nonlinear
numerical representation, an exact decoder equation, finite word/address
semantics, and all state/build/probe/compute/verification/fallback costs.
Forty-way cross-request full-sweep batching yields `1/40` per output, but the
registered 32-token block permits only `1/1280` per token. It is 32x too large
and keeps block `rho=1`; it is a different throughput contract, not a
single-stream component.

### Finite-semiring preprocessing graph boundary

No `WilliamsFiniteSemiringGraph` component is admitted. Its favorable direct
layout is

```text
matrix/input alphabet K + block b
  -> K^b precomputed input-pattern nodes per group
  -> one b-symbol output-pattern value per output group
query vector
  -> choose one node per input group
  -> read ceil(n/b)^2 output-pattern values
  -> semiring-combine a full MatVec result
```

The minimum selected-value traffic is approximately `1/b` of semantic raw
matrix bits, but the persistent edge payload is `K^b/b` times semantic raw.
At registered scale, even the semantically insufficient one-bit model-wide
sidecar is `55,292.57 GiB`; Q4 and BF16 single-query payloads miss the block
budget at theorem-parameter `b=14`. Native BF16/FP32 accumulation cannot use
the proof's regrouping because rounded addition is non-associative.

No compiler, graph artifact, loader, counter state, decoder, native-order
repair, scalar projection, 32-token scheduler, verifier, fallback engine, or
kernel is admitted. A future finite-state transition representation is a new
architecture proposal and must expose its complete state and operation table;
it cannot inherit this theorem by name.

### Global-advice synergy boundary

No per-matrix `CKLTileIndex` or model-wide direct-sum component is admitted.
The rejected interface silently transformed one global nonlinear advice
function

```text
R = f(M_1, ..., M_m), |R| <= 8 GiB
```

into independent local strings `R_i` and summed separately selected hard
queries. XOR advice is an exact counterexample: `R=M_1 XOR ... XOR M_m`
conditionally recovers every `M_i` after the other matrices are known, so no
equal local allocation exists.

A future admissible theorem interface must retain one global `R`, allow all
adaptive cross-matrix probes, and charge every probe needed to unlock shared
information for a jointly composable causal query transcript. A future
constructor must expose those same probes in its physical schedule. There is
currently no representation builder, query algorithm, native numerical
decoder, verifier, fallback engine, 32-token scheduler, or kernel satisfying
that interface.

### Fourier-fiber direct-sum boundary

The proof layer may use the following theorem only for an independently
declared all-linear binary query interface:

```text
arbitrary global advice R(x), |R|<=r
all q in F_2^D
adaptive probes across all blocks
exact <q,x>
  -> 2^(D-r) <= sum(j=0..t,C(D,j)).
```

It is not a runtime component. Standard dense projections expose rank-one or
bounded-rank coefficient masks, not all `2^D` Walsh characters. The 32-tuple
rank-one character-count screen becomes nonrestrictive at 114,194,991 probes,
so no `FourierFiberIndex`, decoder, selector, scheduler, or kernel is admitted.

A future proof interface must lower-bound the restricted character matrix on
large advice fibers. A future execution interface must instead provide a
concrete bounded-word scalar representation and exact decoder. Neither
currently exists.

## Rejected direct factor interfaces: SpikyCut and PowerFold

No `SpikyFactorIndex` is admitted.  The screened direct interface would store
components

```text
S_h = B_h .* (a_h b_h^T)
```

and answer a scalar by two factor dot products per disjoint block.  Its true
logical resource is the total active factor incidence

```text
L = sum_h (|support(a_h)| + |support(b_h)|).
```

The finite E0 description Gate includes arbitrary masks, sparse supports,
real factors, global allocation, and cross-matrix components and still
exhibits sign checkpoints outside the complete model-wide
`L=4,800,000,000` allowance. Consequently no factor loader, partition
decoder, block reducer, native-order repair, cache, scheduler, or kernel is
specified.

No `PowerFoldIndex` is admitted either.  For integer `p`, an entrywise power
of a rank-`r` root expands to `binom(r+p-1,p)` static separable terms and is
therefore the existing low-rank execution interface.  A future architecture
must change the query algorithm itself rather than provide another static
masked-factor description.

## Rejected adaptive interfaces: NearestPair and TrapShift

No `NearestPairLiteralIndex` is admitted. Its proposed interface maps a
rank-one query mask to a cached representative and repairs exactly the
differing checkpoint cells. Even arbitrary joint representatives require an
unaddressable literal answer table under the complete work budget. A future
succinct decoder is a different component and must expose its own information,
storage, probe, and native-order equations.

No `TrapShiftCompiler` is admitted. A trapdoor associated with a sampled mask
does not transfer to an arbitrary shifted checkpoint. The architecture cannot
declare `(W+R)x` free or outsource it to an unspecified average-case solver.

## Rejected direct average-oracle interface: OMEGA-ROWLOTTERY

The Hirahara--Shimizu `AverageOracleAmplifier` is valid in its finite-field
model, but it is not admitted as a free runtime component. Its premise uses
average output-coordinate distance, not a common advantage on every row. The
common-row Fourier bound remains a scoped Gate only.

The concrete direct source `OMEGA-ROWLOTTERY` evaluates selected encoded row
forms exactly and guesses the rest. Exact recovery of an arbitrary
`n`-coordinate result needs at least `n` independent row forms across all
calls; direct evaluation of those forms reads at least `n^2` arbitrary
coefficients. The model-wide one-bit floor is `47.00244140625 GiB`, already
`5.875305x` the hot grant and one complete binary source sweep if cold.

A future `Query-Adaptive Cold Source` must therefore be materially more
succinct than a rank-covering row list and charge its preprocessing structures,
random-instance construction, every probe on every call, list decoding,
verification, numerical lifting, and fallback. The amplifier may then be an
auxiliary correctness layer; it is not itself an answer source.

### Sparse functional dictionary boundary

No `SparseFunctionalDictionary`, functional LDC, or local rank-one truth-table
component is admitted. The only exact binary interface that survived the first
storage/counting screen is

```text
compiled cold cells c_j=<g_j,W>
query mask q -> sparse T(q) with q=XOR(j in T(q),g_j)
answer       -> XOR(j in T(q),c_j).
```

Literal global answers require an address of 39,254,528 bits. Direct `b x b`
rank-one tile tables need `b=10` to reach a favorable 1% coefficient fraction,
but expand a one-bit source by `10,465.29x` (`480.3654 TiB` model-wide).

Near-linear non-systematic storage is not rejected by query cardinality:
counting ceases to decide at 2,020,682 bit-form probes or 494,026 favorable
64-bit words. This is not an upper-bound algorithm. No atom generator,
rank-one-aligned sparse decomposition, metadata index, native numerical cell,
physical layout, or complete runtime equation exists. A random high-girth
dictionary supplies many distinct sparse sums but no reason that they equal
the required masks. `OMEGA-FUNCDICT` therefore remains an unimplemented proof
interface, not an architecture component.

#### Fixed linear decoder closure

The dictionary boundary is narrower than a fixed transform. If its decoder is
`a(q)=Hq` with `GH=I`, then at least `D` decoder rows are nonzero. Under 32
independently selectable binary rank-one tuples, almost every row is active at
least once. Even after granting all 8 GiB as hot one-bit cells and perfect
64-bit cold packing, one hard batch reads at least `41.874345776 GB`, or
`40.8929 ms/token` at 32 GB/s with zero compute.

Consequently no invertible basis, fixed FFT/tensor inverse, or linear syndrome
representative is admitted. Any future `SparseFunctionalDictionary` must expose
a query-dependent nonlinear sparse representative and close its search cost;
arbitrary nonlinear hot advice and causally restricted query sets remain
outside this scoped theorem.

#### Joint factor-envelope boundary

For a batch `q_i=r_i u_i^T`, every query lies in the common space
`R tensor U`, with dimension at most `K^2`. At `K=32` this is a 1,024-slot
restriction target. The product-simplex weight hierarchy is the exact
rank-one intersection law for candidate spans and is validated exhaustively
on all binary subspaces through `2 x 4`.

No `FactorEnvelopeCatalog` is admitted. A registered square has more than
`2^1,046,528` factor-space pairs. No counting-only `SegreDictionary` is
admitted either: even the exact geometry forces only 20,218 bit atoms under
the complete global grant. The open interface is an implicit shared code that
generates the requested restriction and its physical addresses on demand;
that interface has no implementation or native cost closure.

#### Cartesian scalar utility boundary

`R^T W U` is a valid way to share one matrix use across every left/right
pair. If `R` and `U` each have 32 columns, it yields 1,024 exact scalars.
Only the 32 columns of `U` are distinct forward states. The 32 left
directions are measurements of those states, not new successor states, so the
accepted-token denominator remains at most 32. A full near-Shannon BF16 sweep
therefore remains at least `525.1467264 ms/token` at the favorable 32 GB/s
link. Cartesian batching is auxiliary until paired with a partial exact
information source.

#### Nonadaptive XOR-tabulation boundary

No component may claim nonlinear-preprocessing power merely by storing
arbitrary Boolean checkpoint summaries and XORing a fixed query-dependent set
of them. Algebraic-normal-form uniqueness replaces every such summary by its
degree-one coefficient mask while preserving every exact recovery set. This
decoder is exactly the existing sparse functional dictionary.

The remaining binary interface is a deliberately structured, implicit atom
family whose low-weight subset sums contain the Segre/rank-one query set. Raw
capacity first fits the registered fraction for a favorable `23 x 23` block
at six of 529 source positions, but this is not a code construction. Random
atoms miss the required structured masks by exponential margins. No aligned
atoms, decomposer, native lift, joint causal-batch layout, or complete cost
equation exists, so a `SegreAlignedSupercode` is not admitted.

#### Sparse-cover Fourier boundary

The tight local supercode parameters are now rejected by an exact Fourier
Gate, not by random search. Hamming-ball coverage fixes a Krawtchouk character
sum for every rank-one dual matrix. In the `23 x 23`, 619-atom, radius-six
case this requires squared bias at least 277,729, while arbitrary spanning
atoms with all duplicates charged have maximum average
`195,984.95537375606...`.

The first nine capacity-slack-ordered rectangular cases in the registered
`1..128` scan fail the same Gate. `13 x 89` is the first unclosed shape and
requires kernel distance at most 39; it has no atoms or decoder. No
`NearCapacitySegreCover` is admitted, and a future slack cover must expose its
structured short kernel relations, physical decomposer, and native lift.

#### Segre-ruling preimage boundary

The former `13 x 89` frontier is also rejected. For every factor subspace
`A tensor B`, a valid fixed atom cover needs enough low-weight vectors inside
its linear preimage. Information-set projection gives the exact necessary
bound

```text
sum(i<=t,C(S-ab+xy,i)) >= 1+(2^x-1)(2^y-1).
```

The `x=1,y=89` ruling violates it by over four orders of magnitude. The first
combined-Gate survivor is `18 x 36`, not a runtime component. It requires at
least an induced binary `[146,110]` covering code of radius seven and one
shared kernel satisfying all rulings. No `SimultaneousSegreCover` is admitted
until atoms, decoder, native lift, batch union, and costs exist.

#### Determinantal rank-amplification boundary

Any fixed sparse rank-one cover is automatically a radius-`rt` cover of the
complete rank-at-most-`r` determinantal set. Every proposed block must pass
the exact Hamming-ball inequality at every matrix rank. Canonical
representatives must also pass one mixed-weight sampling inequality using the
exact maximum rank-one intersection of every sampled atom span.

These Gates reject the first 770 raw-capacity rectangles. The next parameter
frontier is `30 x 40`, `S=1,404`, `t=14`; it has no atoms or decoder. No
`DeterminantalSparseCover` is admitted until it constructs the rank-one cover,
joint physical batch, native lift, and full costs.

#### Biorthogonal cancellation boundary

The `r*t` radius is not a free worst case.  Minimal decompositions of a fixed
rank-`r` matrix form biorthogonal bases, and their admissible term pairs are
the binary anti-flag graph.  Every fixed sparse dictionary must charge the
support overlaps forced by that graph's spectrum.  Overlaps cancel in XOR.

This rejects the former `30 x 40` frontier and two successors.  The only next
fixed-linear design target admitted by the current exact Gates is the
unconstructed `31 x 43`, `S=1,559`, `t=15` shape.  It is not an architecture
component until it supplies explicit atoms, a sub-dense decomposer, a shared
causal-batch layout, native-order numerical semantics, and complete costs.

#### Recursive biorthogonal cancellation boundary

The one-shot anti-flag overlap charge is reusable. A positive summed
internal-edge lower bound identifies an adjacent pair of rank-one terms whose
representatives share an atom. Removing the pair drops matrix rank by two,
loses at least two support incidences, and leaves the same dictionary contract
on the residual matrix. Therefore every admitted fixed binary atom cover must
satisfy the recursive Gate

```text
Delta_r >= 2 + Delta_(r-2)
R_<=r = max_(j<=r) (j*t-Delta_j).
```

This rejects the former `31 x 43` target at rank 22 and moves the combined
fixed-linear frontier to the unconstructed `31 x 42`, `S=1,523`, `t=15`
shape after 856 rejected capacity-ordered rectangles. No
`RecursiveBiorthogonalCover` is admitted until it supplies explicit atoms,
sub-dense decoding, native-order semantics, joint causal-batch layout, and
complete costs. Adaptive word-valued probes and nonlinear output decoding
remain outside the Gate.

#### Extension-field adaptive-support boundary

Packing matrix columns into `GF(2^m)` does not make adaptive linear summaries
local. Run any deterministic exact decoder on the zero source. Its returned
linear cells are all zero and fix one adaptive support `J`; every perturbation
in their common kernel follows the same path. Exactness therefore forces the
rank-one query direction into `span_GF(2^m)(J)` even when the final decoder is
nonlinear. A `j`-cell field span contains at most `2^j` binary directions, so
every admitted block-local field-linear layout must satisfy

```text
sum_(j=0)^t C(S,j) 2^j >= 2^k.
```

At `k=16,384`, proportional storage needs at least 3,421 field-cell probes,
versus 194 logical probes or 776 after granting all four packed-Q4 traffic
lanes to the one binary plane. Even granting all 551.22 GB of DFloat11 plus
8 GiB to that plane leaves the aggregate relaxed minimum at `10.988948%`,
above the favorable `4.740741%` traffic allowance. No extension-field linear
summary component is admitted. Nonlinear source cells and cells mixing
independent matrix blocks remain outside this Gate and unconstructed.

#### Adaptive packed-linear word boundary

A 64-bit load may expose 64 unrelated binary linear checkpoint summaries, not
only one extension-field multiple. This stronger packing still has an exact
zero-transcript Gate. A zero-source adaptive path using `p` words puts the
requested rank-one mask in a subspace of dimension at most `64p`. The exact
Segre intersection of each such subspace, unioned over all word supports,
rejects seven or fewer words for the `31 x 42` frontier; the first count-
feasible point is eight words. Every block-local rectangle with sides at most
128 misses the registered traffic line even under four-lane Q4 accounting;
the best relaxed point is `118 x 128`, 22 words, at `11/472` traffic.

No packed-linear word component is admitted. A new address scheduler or field
packing does not change this boundary. Nonlinear cell contents and global
cross-matrix encodings remain outside it.

#### Adaptive nonlinear probe-degree boundary

Arbitrary nonlinear stored cells do not evade determinantal capacity. A
depth-`t` adaptive decoder is a cylinder polynomial whose monomials mention at
most `t` cells. Multiplying `r` exact rank-one characters gives every rank-
at-most-`r` character at cylinder degree at most `rt`. Therefore an alphabet-
`A`, `S`-cell block must satisfy

```text
RankLeq(a,b,r) <= sum_(j<=rt) C(S,j)(A-1)^j
```

at every rank. This includes arbitrary nonlinear encoders, value-dependent
addresses, and arbitrary deterministic exact postprocessing. With bit cells,
the `31 x 42` block still needs at least 15 probes.

Nonlinear 64-bit words have a real capacity gap rather than a component. The
smallest side-128 point not rejected at the traffic line is `25 x 108`: 2,700
source bits, 50 padded words, and two probes. The best count point is
`128 x 128` with four probes. Neither point has an encoder or decoder.

The smallest nonlinear seed, `2 x 3` source bits encoded into seven bits with
two adaptive probes, also remains unconstructed. The complete systematic-six-
bits-plus-one-arbitrary-advice subclass is impossible; a fully non-systematic
SMT search timed out and is recorded as inconclusive. No nonlinear probe
component enters the runtime until an explicit finite encoder, both address
functions, exact decoder, native numerical lift, joint batch, and complete
costs exist.

<!-- EXP-088B:START -->
## EXP-088B architecture status

The page-mask object is an offline oracle program descriptor, not an accepted runtime component. It has no causal selector: one checkpoint-static mask is frozen from build states and applied unchanged to holdout states. Architecture promotion is `False`. Until a full-layer existing-ISA executor passes 128 exact steps, the production architecture remains unchanged.
<!-- EXP-088B:END -->

<!-- EXP-089A:START -->
## EXP-089A architecture status

Prefix token IDs plus RNG bytes are accepted as a behavioral-state witness only when `ACCEPT_PREFIX_STATE_BISIMULATION_WITNESS_FOR_TOKEN_DECISION_RESEARCH` is the authoritative result. The production executor architecture is unchanged: replay is a reference oracle, not a runtime component. The next architecture candidate must compute the exact token from this state with an explicit proof and resource trace, then append it; it may use the unchanged suffix only as a fully charged exact state constructor.
<!-- EXP-089A:END -->

<!-- EXP-090A:START -->
## EXP-090A architecture status

The suffix-action object is an auxiliary causal draft component unless the authoritative decision explicitly promotes it. It may never commit state independently. Exact state comes only from the accepted token prefix and the unchanged target verifier's official cache. Architecture promotion: chain=`False`, candidate-tree=`False`. Verification arithmetic remains unreduced.
<!-- EXP-090A:END -->

<!-- EXP-091A:START -->
## EXP-091A architecture status

A Jacobi block may commit only the prefix where input guesses equal unchanged-target proposals consecutively from position one. The official block cache is valid for exactly that prefix. Architecture promotion without compression=`False`; conditional compressed promotion=`False`. Later sweeps are not admitted into the final core at K=128 because their checkpoint traffic exceeds the target.
<!-- EXP-091A:END -->

<!-- EXP-092A:START -->
## EXP-092A architecture status

The block-span object is an oracle correction feasibility test, not a runtime component. It may enter the architecture only after a surviving lower-bound result is followed by a build-frozen basis, causal exact coefficient solver, all-layer native-byte equality, and complete online resource trace. Architecture promotion: `False`.
<!-- EXP-092A:END -->

<!-- EXP-093A:START -->
## EXP-093A architecture status

The static top-k table is an oracle-evaluated candidate source, not a runtime component. It enters the architecture only after the source Gate passes and an exact branch-dependent successor generator commits the same behavioral prefix state under a complete online resource trace. Architecture promotion: `False`.
<!-- EXP-093A:END -->

<!-- EXP-094A:START -->
## EXP-094A architecture status

The Parareal equation is a candidate-source object, not an accepted executor. It may enter the architecture only after a surviving source result is paired with a sound token certificate or exact branch successor, complete prefix-state commitment, and a charged physical resource trace. Architecture promotion: `False`.
<!-- EXP-094A:END -->

<!-- EXP-095A:START -->
## EXP-095A architecture status

The online residual bank and its affine/span programs remain candidate-source objects. Only a causal source with a sound finite-word token certificate, exact prefix-state commitment, and charged physical resource trace may enter the executor architecture. Promotion status: exact-chain `False`, causal-branch `False`, oracle-span `False`.
<!-- EXP-095A:END -->

<!-- EXP-096A:START -->
## EXP-096A architecture status

The FP64 margin certificate is an offline target-seeing capacity object. It enters the runtime architecture only after a causal checkpoint-derived generator emits the same coefficient words before target continuation and a physical resource trace charges scoring, certification, prefix state, and fallback. Promotion: `True`.
<!-- EXP-096A:END -->

<!-- EXP-100A:START -->
## EXP-100A architecture status

The mixed rectangular tensor sequence and cut-depth transformed-weight plan are
research descriptors, not production opcodes. Architecture promotion is
conditional on `INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE` and the active next Gate `EXP-100A control repair`. Until
finite-word closure, a causal block, exact successor state, and existing-ISA
lowering pass, the production executor architecture remains unchanged.
<!-- EXP-100A:END -->

<!-- EXP-102A:START -->
## EXP-102A architecture status

The only evaluated component is a real causal draft/verify loop. Architecture promotion follows `REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE`. No multiplication oracle, free future activation block, or unimplemented transform circuit is an accepted runtime component.
<!-- EXP-102A:END -->

## Nonlinear adaptive word-router architecture boundary — 2026-09-08

No local `NonlinearWordRouter` is admitted. The exact largest-fiber/Segre Gate
now covers arbitrary nonlinear word contents and arbitrary deterministic
value-adaptive addresses. It rejects the former `25x108`,50-word,two-probe
capacity point and every target-feasible side<=128 64-bit local rectangle.

The first single-query local interface not rejected by the cover Gate is:

```text
source block       24x225 or 25x216 binary coefficients
encoded storage    99 padded 64-bit words
query budget       4 word probes = 8/675 of four-lane source bytes
required routing   multiple successive value-dependent address stages
encoder            OPEN
address generator  OPEN
decoder            OPEN
native lift        OPEN
```

Nonadaptive four-word support and a one-value-stage dispatcher are already
rejected. More importantly, the complete rank-one service forces a worst-source
all-query union of at least 84 words, hence some independently-selectable
32-query tuple uses at least 32 distinct words = `64/675`, 8x target. Therefore
this local interface is not admitted even if a four-probe single-query encoder
were found. The unresolved architecture boundary is a legal-causal restriction
or global cross-matrix/nonlocal producer; no placeholder selector, implicit
catalog, free decoder, or response bank may enter. Architecture promotion
remains `False`.
