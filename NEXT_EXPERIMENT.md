# Next Experiment

## Closed Gate — EXP-048

Authoritative evidence:

```text
results/exp_048/summary.json
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
artifact 8850040445
artifact SHA-256 67c587da36b968f9c38e0a7774ea03cecd2ad2d7d274d3e83c833c56529c3443
```

MEASURED:

```text
B1 perfect future oracle:
  96 tokens / 1 target pass
  exact mismatches 0
  target-equivalent fraction 1.0416667%
  future information true
  deployable false

B2 hard Jacobi:
  p50 58 target passes / 32 exact tokens
  p50 fraction 181.25%
  p90 fraction 193.75%
  max matching prefix 3

B3 partial-layer self-draft:
  18 cases, 54 fixed variants
  exact mismatches 0
  future information uses 0
  p50 committed tokens / verification 1
  max matching prefix 1
  minimum fully accounted fraction 1333.463%
  p90 fully accounted fraction 2893.843%
```

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

B1 proves block verification can meet the logical traffic target when a long proposal is already correct. B2/B3 prove that hard Jacobi and sequential early-layer drafting do not produce that proposal cheaply. B4 tree expansion is not continued from failed B3.

## EXP-049 — Anderson-Accelerated Continuous Block Fixed-Point Gate

### Mechanism change

EXP-049 removes the separate per-token draft loop entirely.

Represent a future block of `K` unknown token states as continuous token embeddings `Z`. Execute the unmodified target model over the exact prefix plus all `K` soft positions in one causal batched pass. Convert the resulting future logits into a sparse soft embedding update and solve the causal block fixed point using a small number of damped Picard or Anderson-accelerated iterations. Harden the block to tokens and apply the retained exact longest-prefix-plus-correction verifier.

No target weights are modified. No training, learned adapter, future generated token, or external draft model is allowed in deployable conditions.

### Why this directly addresses EXP-048

- B3 repeatedly executed early layers and a full LM head once per proposed token; EXP-049 executes each target layer/head once per solver iteration across the whole block.
- B2 hard Jacobi propagated discrete guesses slowly; EXP-049 tests whether continuous residual information and Anderson mixing can move useful information farther than one token per iteration.
- The exact block verifier remains unchanged, so any incorrect hard proposal is safely truncated and corrected.

### Continuous map

For exact prefix token embeddings `E(p)` and future soft states `Z_0...Z_{K-1}`:

```text
L(Z) = target_logits(E(p) concat Z)
P_i  = top-k-softmax(L_i / tau)
F(Z)_i = sum_{v in top-k_i} P_i(v) * token_embedding(v)
R(Z) = F(Z) - Z
```

Logit alignment remains `prefix_length - 1 + i`.

The deployable solver may use only current prefix state, fixed initialization metadata, prior solver iterates, and current target outputs.

### Conditions

#### S0 — hard Jacobi baseline

Reuse EXP-048 B2 with every target pass charged.

#### S1 — damped continuous Picard

```text
Z_{r+1} = (1 - lambda) Z_r + lambda F(Z_r)
```

Pre-register `lambda` values and top-k/temperature settings. No post-hoc unrestricted tuning.

#### S2 — Anderson acceleration

Use a bounded history `m` and solve the small residual least-squares problem in float64 on CPU reference code. Apply regularization, coefficient clipping, finite checks, and fail-closed fallback to S1.

Pre-register history sizes `m in {2, 4, 8}` and at most four solver iterations for the early Gate.

#### S3 — exact future-state oracle

Initialize `Z` from exact future tokens only to validate map alignment and the theoretical best hardening/verifier path. S3 is future-aware and non-deployable.

#### S4 — adversarial triangular models

Construct finite causal models where token `i` reveals a transformation of exact token `i-1` and arbitrary initialization is wrong. Use them to test the worst-case one-new-guaranteed-position-per-round barrier.

### Initialization

Deployable fixed choices:

- repeated exact next-token embedding obtained from the first solver pass;
- repeated last-prefix token embedding;
- fixed zero/mean embedding control.

Initialization may not use reference continuation or held-out target tokens.

### Accounting

For every state/condition record:

```text
block_size
solver_iterations
target_solver_full_streams
exact_verification_full_streams
correction_streams
soft_topk_projection_bytes_and_ops
anderson_history_bytes
matching_prefix
committed_tokens
rejected_positions
future_information_used
numerical_fallbacks
```

Primary logical metric:

```text
target_equivalent_stream_fraction =
    (solver target streams
     + exact verification streams
     + correction streams
     + separately normalized projection cost)
    / exact committed tokens
```

Projection/Anderson cost may not be hidden merely because it is smaller than the target model.

### Causal triangular lower-bound audit

EXP-049 must state and test the following worst-case claim:

> For arbitrary causal dense models, a target-only synchronous block solver with no external future information cannot guarantee more than one new exact token position per black-box causal target round in the worst case.

Required work:

1. formalize the target interface and guarantee being claimed;
2. construct an adversarial finite causal model family;
3. prove indistinguishability of later positions before predecessor resolution under the declared interface;
4. show hard Jacobi attains the one-position-per-round bound on the construction;
5. determine exactly which assumptions continuous embeddings/Anderson violate or do not violate;
6. keep the theorem separate from empirical average-case checkpoint results.

A valid worst-case impossibility result does not fabricate an empirical failure, but it changes what can be claimed for the fixed “arbitrary model + exact output” objective.

### Small-checkpoint corpus

Use the same three pinned TinyStories checkpoints and six held-out families as EXP-048 for direct comparison. Add at least two deterministic adversarial toy causal models and low/high entropy synthetic controls.

Early block sizes:

```text
K in {64, 128, 256}
solver iterations in {1, 2, 4}
```

Respect each checkpoint's maximum position length. Record excluded states rather than silently truncating the Gate.

### Pre-registered early rejection Gate

Reject target-only continuous fixed-point proposal generation as the core path if any condition holds:

```text
exact verifier mismatch >0
future information in S1/S2 >0
NaN/Inf or unhandled Anderson instability >0
best S1/S2 p50 matching prefix after <=4 solver passes <16
best S1/S2 p90 target-equivalent stream fraction >10%
S2 does not improve p50 matching prefix over S0 by at least 4x
best checkpoint acceptance materially worsens with model size
adversarial triangular model violates any claimed universal >1-position/round guarantee
```

A proved worst-case one-position-per-round barrier rejects universal exact target-only fixed-point acceleration for the declared arbitrary-model contract, even if a few average-case prompts improve.

### Promotion Gate

A solver may advance to a complete Phase-C runtime only if:

```text
zero exact mismatches
zero deployable future information
zero unhandled numerical failures
p90 target-equivalent stream fraction <=1.185185%
p50 committed tokens satisfies the dynamic pass-count requirement
nonzero success across every held-out family
non-degrading size trend
no contradiction with the exact claim scope
```

Dynamic requirement:

```text
committed_tokens >= ceil(total_target_equivalent_streams / 0.01185185)
```

Examples before projection overhead:

```text
2 total streams -> >=169 tokens
3 total streams -> >=254 tokens
4 total streams -> >=338 tokens
5 total streams -> >=422 tokens
6 total streams -> >=507 tokens
```

### Strongest counterexamples

- a token-copy chain that reveals exactly one position per round;
- alternating-token and cryptographic/hash-like predecessor maps;
- Korean/code/identifier prompts with early mismatch;
- soft states whose top-k support excludes the exact token;
- Anderson coefficient explosion or oscillation;
- a long apparent soft convergence with hard prefix length zero;
- block lengths near context limits;
- exact verification mismatch at position zero.

### Evidence boundary

Before a complete small-checkpoint solver path is frozen:

```text
Phase: A/B
Evidence ceiling: E1
complete real operation replacement: false
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec: NOT TESTED
```

### Next exact action

Create a new branch after PR #58 merges and implement in this order:

1. formal triangular lower-bound statement and adversarial toy models;
2. model-independent S1/S2 solver with numerical fault tests;
3. exact hardening and retained block verifier integration;
4. pinned TinyStories runner and raw evidence workflow;
5. pre-registered Gate decision and durable state update.
