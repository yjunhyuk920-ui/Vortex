# VORTEX Architecture

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
