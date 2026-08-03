# VORTEX Architecture

## Mission boundary

VORTEX is a runtime, not a retrained target model. It ingests an unmodified supported Hugging Face dense checkpoint and automatically constructs runtime state/metadata.

## Fixed target execution stack

```text
unmodified Hugging Face target checkpoint
        |
        v
checkpoint/shard inspector and automatic runtime compiler
        |
        v
causal executor
        |
        +--> proposal/information source
        +--> exact or declared probabilistic certificate/verifier
        +--> safe commit OR exact correction/fallback
        |
        v
memory scheduler
        - target/draft hot state
        - KV and work buffers
        - RAM/SSD cold state
        - every proposal, verification, rejection, and fallback charged
        |
        v
original-model-compatible output contract
```

## Core requirement

The primary runtime must causally reduce or amortize original 405B weight traffic and arithmetic on unseen prompts. A component is core only when it participates in that measured cost reduction. Correctness-only components are auxiliary.

## Mandatory interfaces

### Proposal or information source

Produces candidate tokens, soft states, decision metadata, or an execution capsule. It must report:

- causal inputs;
- target/draft/solver pass counts;
- target-specific training or compilation, if any;
- parameter and state bytes;
- future-information use;
- rejected positions and numerical fallback;
- selector cost.

### Certificate or verifier

Declares exactly one contract:

- deterministic exact;
- deterministic top-1;
- bounded-logit error;
- probabilistic top-1 with union-accounted delta;
- exact target block verification with longest-prefix plus first-mismatch correction.

Future-token oracle controls are never deployable evidence.

### Correction/fallback

When proposal matching, certification, metadata, or numerical solving fails, execute the exact target work required by the declared reference contract. No silent approximation and no uncharged correction stream.

### Memory virtualization

Separate:

```text
M_total = M_target_hot + M_draft_hot + M_kv_target + M_kv_draft
          + M_work + M_metadata + M_verify + M_fallback
```

The final target requires `M_total <=8 GiB`; this remains UNVERIFIED.

### Evidence

Every run emits exact revisions, phase/evidence/provenance, target/draft/solver passes, logical and physical bytes where available, exact-prefix distribution, correction/fallback, future-information audit, numerical failures, memory, timing, and output agreement.

## Auxiliary accepted components

- safetensors discovery/slice access;
- compact/checksummed mmap pointer VM;
- atomic runtime-format builder;
- bounded exact decision-index and finite-horizon DAG components;
- CPTC certificate, metadata fault rejection, and exact fallback at E1;
- `vortex_runtime/block_verify.py` exact longest-prefix plus correction verifier at E1;
- `vortex_runtime/fixed_point.py` Picard/Anderson reference and fail-closed numerical machinery at E1;
- hidden triangular adversarial models for universal target-round claims.

None currently supplies the core proposal information required by the final runtime.

## Rejected core mechanisms

See `FAILED_APPROACHES.md`.

Closed families include:

- global/oracle-tight/stratified range-based CPTC;
- hard target-only Jacobi;
- sequential partial-target-layer self-draft with repeated target LM-head work;
- target-only continuous Picard/Anderson block solving;
- proposal-tree expansion from a failed single-path source.

## Retained exact block verifier

```text
exact target prefix + proposed block
        |
        v
one exact causal target block pass
        |
        v
left-to-right comparison
        |
        +--> commit every matching proposal token
        +--> commit exact target token at first mismatch
        +--> discard every later position/state
```

EXP-048/049 committed-output mismatches: zero in their tested scopes.

Verifier arithmetic positive control:

```text
96 perfect future tokens / one target pass = 1.0416667%
```

The verifier is not the proposal source.

## Closed EXP-049 target-only solver architecture

### Tested map

```text
exact prefix embeddings || continuous future block Z
        |
        v
full causal target pass
        |
        v
aligned logits -> top-k soft projection through target embedding
        |
        v
F(Z), residual F(Z)-Z
        |
        +--> damped Picard
        +--> bounded Anderson history
        |
        v
hard proposal -> exact block verifier
```

MEASURED favorable result:

```text
exact-reference-selected p50 prefix 4.5
maximum prefix 6
p90 target-equivalent fraction 168.778596%
Anderson p50 prefix after four passes 1
hard Jacobi p50 prefix after four passes 4
```

Triangular adversarial result:

```text
Picard round prefixes 1,2,3,4
Anderson round prefixes 1,2,3,3
hidden suffix indistinguishability true
```

Decision:

- retain solver/numerical reference only;
- reject target-only fixed-point proposal as core;
- do not tune only top-k/temperature/damping/history/iterations;
- do not build a target-only fixed-point GPU backend from this evidence.

## Active EXP-050 external-draft architecture

EXP-050 changes the information source:

```text
exact prompt
   |                         unmodified external draft checkpoint
   +------------------------> cached sequential draft generation
                                      |
                                      v
                              K-token causal proposal
                                      |
exact target prefix + proposal ------+
        |
        v
one exact target block verification pass
        |
        +--> longest exact proposal prefix
        +--> first-mismatch exact target correction
```

The target remains unmodified and untrained. The external draft is already published and fixed independently of target future tokens.

### Draft pool

For the initial falsification Gate:

```text
Target TinyStories-1M <- Drafts 3M, 8M
Target TinyStories-3M <- Drafts 1M, 8M
Target TinyStories-8M <- Drafts 1M, 3M
```

A reference-selected best draft is an explicitly non-deployable favorable upper bound. A deployable selector remains a separate obligation.

### Universal boundary

A fixed target-independent draft cannot guarantee a nonzero exact prefix for every arbitrary target: for the same prompt, an adversarial target can choose a first greedy token different from the draft.

Therefore EXP-050 reports separately:

- universal exact claim: tested by the first-token counterexample;
- practical restricted-family acceptance: tested by cross-checkpoint proposals.

### Accounting equations

For target parameter bytes `P_t`, draft bytes `P_d`, proposal length `K`, and exact committed tokens `A`:

```text
S_actual/token = (K * P_d/P_t + 1 target verification stream) / A
```

Final 4B-draft/405B-target normalization:

```text
S_projected/token = (K * 4/405 + 1) / A
```

Required target condition:

```text
S_projected/token <=0.01185185185
```

Even with a completely correct proposal:

```text
4/405 + 1/K <=0.01185185185
K >=507
```

Thus an actual 4B draft requires at least 507 consecutive exact proposal tokens before other overhead. The older 85-token minimum applies only to a zero-cost proposal.

### Tree expansion

A proposal tree is forbidden until the single-path fixed pool survives its early Gate. Every branch and target-scored node must be charged. A tree cannot be used to hide position-zero divergence.

## Resource equations

```text
B_total/token = B_draft + B_target_verify/A + B_selector
                + B_correction + B_metadata

C_total/token = C_draft + C_target_verify/A + C_selector
                + C_correction

T_token >= max(B_total/effective_bandwidth,
               C_total/effective_throughput,
               serial_latency_floor)
```

PROJECTED same-bit target comparison:

```text
405B Q4 target stream: 188.592821 GiB
4B Q4 draft/baseline stream: 1.862645 GiB
1.2x allowance: 2.235174 GiB/token
required target-equivalent fraction: 1.185185%
```

Hardware terms remain `NOT TESTED` until Phase D.

## Safety rule

No path commits outside its declared exact verifier or certificate. Invalid revision, metadata, proposal, numerical state, selector, or accounting triggers exact correction/fallback or abort.
