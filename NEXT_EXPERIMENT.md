# Next Experiment

## Closed Gate — EXP-047R

Authoritative evidence:

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

MEASURED:

```text
3 pinned trained dense checkpoints
18 held-out current-token states
wrong accepts 0
bound violations 0
C1 exact-state oracle median 100%
C1 exact-state oracle p90 100%
C2 median 100%
C2 p90 100%
C2 best 254/256 = 99.21875%
```

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

The strongest favorable range-only oracle failed the pre-registered 10%/25% Gate by a wide margin. C3 variance-adaptive tuning is not the next experiment because it would attempt to rescue a mechanism class already rejected by its exact realized range oracle.

## EXP-048 — Causal Block Verification Amortization Gate

### Mechanism change

EXP-048 does not skip scalar matrix tiles by partial-sum certification. It changes the cost structure:

> stream the target model weights once across a block of proposed future positions, verify all positions in one exact causal target pass, and divide that full target stream by the number of exactly accepted tokens.

The deployable proposal path must be causal and training-free. The target checkpoint remains unmodified.

### Exact target traffic requirement

From the fixed same-bit projection:

```text
405B Q4 full target stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 0.01185185
```

With a zero-cost perfect proposal, one full target verification pass must accept at least:

```text
ceil(1 / 0.01185185) = 85 tokens
```

Any real draft cost increases the required accepted block length.

### Declared algorithm

For an unmodified causal dense checkpoint:

1. maintain the exact committed prefix and KV state;
2. use a training-free partial-layer self-draft to propose `K` tokens causally;
3. concatenate the proposal to the committed prefix;
4. execute one exact full-target teacher-forced block pass with a causal mask;
5. compare target argmax tokens with the proposal left to right;
6. commit only the longest exactly matching prefix;
7. at the first mismatch, commit the exact target token and discard later proposal state;
8. preserve exact greedy decoding output by construction.

No future generated token may enter the deployable proposal path.

### Conditions

#### B0 — exact sequential greedy baseline

One full target pass per token. Correctness control.

#### B1 — perfect-proposal oracle upper bound

Use exact future greedy tokens only to prove the maximum possible block-verification amortization and to validate accounting. This is non-deployable, future-aware, and cannot count as evidence for the runtime proposal mechanism.

#### B2 — existing Jacobi baseline

Reuse the existing exact Jacobi implementation as a control. Do not relabel its target passes as one-pass block verification and do not hide failed iterations.

#### B3 — causal partial-layer self-draft

Generate proposals using an early prefix of the same checkpoint layers, the checkpoint's own normalization/output head, and no learned adapter. Sweep a small pre-registered set of layer fractions and block lengths. All draft weight traffic, target verification traffic, rejected positions, KV rebuilds, and correction passes are charged.

#### B4 — causal tree variant only after B3

A bounded proposal tree may be tested only if B3 shows meaningful acceptance. Every expanded node and target-scored position is charged. No future-token or reference-continuation routing is allowed.

### Traffic accounting

For each verification cycle record:

```text
accepted_tokens
proposed_tokens
draft_layer_equivalent_streams
target_full_streams
correction_target_streams
rejected_scored_positions
```

Primary derived metric:

```text
target_equivalent_streams_per_accepted_token =
    (target_full_streams
     + correction_target_streams
     + draft_layer_equivalent_streams)
    / accepted_tokens
```

This is a logical same-bit weight-stream metric, not wall-clock speed. Hardware latency remains Phase D.

### Phase A/B implementation target

Use at least three pinned small trained dense causal checkpoints with layer access supported by the reference implementation. Use held-out prompts across English, Korean, code, mathematics, structured output, and ordinary narrative when tokenizer/model capability permits.

The Phase A/B runner must:

- reproduce exact sequential greedy tokens;
- prove B1 block verification equals the sequential baseline;
- implement B3 without training or checkpoint modification;
- record all proposal/verification passes and positions;
- save raw per-cycle JSONL, exact revisions, environment, and checksums;
- keep oracle and deployable results in separate fields;
- fail closed on any token mismatch not explained by the declared correction rule.

### Pre-registered early rejection Gate

Reject partial-layer self-draft block amortization as the core path if any condition holds on held-out generation:

```text
exact output mismatch >0
future information in deployable path >0
p50 accepted tokens per target verification <16
p90 target-equivalent stream fraction >0.10
acceptance trend worsens materially with checkpoint depth/size
accounted B3 cost is not lower than exact sequential B0
```

These thresholds are deliberately lenient. Passing them does not establish the final 1.185% requirement; it only permits a stronger Phase-C Gate.

### Promotion Gate

A candidate may advance toward real operation replacement only if:

```text
zero exact mismatches
zero future information
p50 accepted block >=85 after full cost accounting
p90 target-equivalent stream fraction <=0.01185185
nonzero success across all declared held-out task families
model-size trend is non-degrading
```

The `>=85` requirement is not negotiable under the current same-bit traffic objective unless another independently measured mechanism reduces the full target stream cost.

### Strongest counterexamples

Include:

- low-entropy repetitive text;
- high-entropy code and identifiers;
- Korean and English switching;
- arithmetic with brittle token dependencies;
- prompts whose early-layer and final-layer argmax disagree immediately;
- long blocks with a mismatch near the first position;
- EOS and structured-format boundaries;
- deliberately bad fill tokens for Jacobi controls.

### Evidence boundary

Before an actual runtime replaces sequential target decoding on a real checkpoint:

```text
Phase: A/B
Evidence ceiling: E1
405B: NOT TESTED
8 GiB VRAM: NOT TESTED
CUDA/PCIe/SSD/TTFT/tokens/sec: NOT TESTED
```

### Next exact action

Create a new branch and pre-register:

```text
docs/research/EXPERIMENT_048_CAUSAL_BLOCK_VERIFICATION_AMORTIZATION.md
experiments/exp_048/
tests/exp_048/
results/exp_048_candidate/
.github/workflows/exp_048_gate.yml
```

First implement B0/B1 accounting and exactness tests, then B2 existing Jacobi control, then the smallest B3 partial-layer self-draft. Do not start B4 before B3 survives its early rejection Gate.
