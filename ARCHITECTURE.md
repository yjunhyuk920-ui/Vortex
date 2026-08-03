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
