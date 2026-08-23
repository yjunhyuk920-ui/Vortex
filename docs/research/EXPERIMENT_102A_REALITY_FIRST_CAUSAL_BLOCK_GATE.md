# EXP-102A — Reality-First Causal Draft/Verify Gate

## 1. Why this Gate replaces EXP-101A

EXP-101A granted a perfect future activation block and made transforms, additions, bytes, workspace, repair, and `N/A` free. Those assumptions cannot occur in the deployed runtime. EXP-102A changes the first question from “does a multiplication count look small under an oracle?” to:

> Can an actual causal draft and an actual unchanged target produce a long exact committed block after every required operation and state transition is executed and charged?

The multiplication-only EXP-101A run is withdrawn from promotion. It may not become an authoritative scientific result under the reality-first contract.

## 2. Three materially different principles considered

### A. Real causal draft plus exact block verification — selected

An unchanged small public model generates a greedy candidate from the exact committed text. The unchanged target verifies all proposed positions, crops rejected target KV, executes the exact mismatch token, rebuilds the draft state from the exact committed text, and compares against an incremental target reference.

This directly measures the missing variables:

\[
A,\qquad N/A,\qquad T_{\rm draft},\qquad T_{\rm verify},\qquad T_{\rm repair}.
\]

### B. Shared-input projection bundles — rejected at E0

Q/K/V and gate/up can concatenate weights because they share an input. This shares some transforms but does not introduce a lower bilinear-rank source. EXP-100A’s dominant leaf work remains, so it has no independent route from `38.25%` to the final target.

### C. Whole-layer nonlinear superinstruction — deferred

This changes the correct unit, but no finite-word executable instruction, causal compiler, or complete resource path currently exists. Under the reality-first contract, a missing instruction cannot be supplied for free.

Principle A is selected because it is fully executable, causal, exact/fail-closed, and cheapest to falsify on real public checkpoints.

## 3. Frozen public models

Target:

```text
HuggingFaceTB/SmolLM2-360M
revision f8027fd0eaeea54caa13c31d31b9fdc459c38b49
BF16 eager CPU
```

Draft arms:

```text
same_family:
  HuggingFaceTB/SmolLM2-135M
  revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
  direct tokenizer IDs

cross_family_text_bridge:
  roneneldan/TinyStories-33M
  revision 72da9b32b3fda7cb82db29b75f88ae5ea0d980e8
  tokenizer EleutherAI/gpt-neo-125m
  revision 6cb0d322a3a484e99667e7cb240e22f1ac036b99
  decoded text is retokenized by the target tokenizer
```

The same-family arm is a real but restricted upper rung. It cannot establish universal arbitrary-checkpoint promotion by itself. The cross-family arm tests the model-independent text interface.

## 4. Exact online step

For exact committed prefix `p`, draft `D`, target `F`, and requested block `K`:

1. `D` greedily generates real tokens from `p`.
2. A same-tokenizer arm uses the actual IDs. A cross-tokenizer arm decodes text and retokenizes with the target tokenizer. If the target prompt-token prefix is not stable, the bridge fails closed.
3. `F` executes the complete candidate block with its real KV cache.
4. The longest exact greedy prefix is accepted.
5. On mismatch, rejected target cache positions are cropped and the actual target mismatch token is executed.
6. The draft state is rebuilt from the exact committed text. Rebuild time and cache bytes are charged.
7. An independent incremental target reference generates the same number of committed tokens.
8. Token bytes and complete terminal target KV hashes must match.

No future target token participates in candidate generation.

## 5. Complete accounting

The authoritative row records:

- draft prefill and every draft autoregressive step;
- draft decode, text bridge, target retokenization, and all packing/argmax/comparison time;
- every target candidate position;
- `N/A` including mismatch repair;
- target verification and repair wall time;
- real same-tokenizer cache crop/mismatch replay or cross-tokenizer full rebuild wall time;
- incremental target baseline wall time for the same committed population;
- target/draft parameter bytes;
- draft and target cache bytes;
- process peak RSS;
- exact token and terminal-state equality.

No compression credit is used for promotion. One raw target sweep must be amortized by actual committed tokens.

## 6. Frozen build/holdout protocol

The only executable block length is frozen before results:

```text
K = 96
```

`K=64` is excluded because it cannot satisfy the raw p50 minimum `A>=85` even under full acceptance.

Selection sees only English, code, and repeated-pattern build cases. Math, structured JSON, and Korean prompts are untouched holdouts.

The raw traffic prerequisites are derived without compression:

\[
A_{p50}^{\min}=\left\lceil\frac{1}{0.011851851851851851}\right\rceil=85,
\]

\[
A_{p95}^{\min}=\left\lceil\frac{1}{0.014814814814814815}\right\rceil=68.
\]

## 7. Promotion Gate

For each arm on every holdout case:

```text
exact token equality                     true
exact terminal target KV equality        true
minimum committed tokens                 >= 85
p50 speculative/baseline wall time       <= 1.2
p95 speculative/baseline wall time       <= 1.5
p95 N/A                                  <= 1.5
```

Decisions:

```text
PROMOTE_REAL_CAUSAL_DRAFT_BLOCK_TO_LARGER_SCALE_FULLY_CHARGED_GATE
RETAIN_SAME_FAMILY_CAUSAL_DRAFT_AS_RESTRICTED_AUXILIARY_NOT_UNIVERSAL
REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
INVALID_REALITY_FIRST_CAUSAL_BLOCK_CONTROL_FAILURE
```

Universal promotion requires both same-family and cross-family arms. A same-family-only pass is restricted auxiliary evidence.

## 8. Stop rule

On rejection, do not sweep prompts, K, temperature, sampling, tokenizer cleanup, model revisions, or acceptance thresholds. Reopening requires a materially different causal information source that is executable and fully charged.

## 9. Claim boundary

This is an E2/E3 CPU mechanism Gate. It is not 405B execution, physical 8-GiB residency, target GPU performance, storage bandwidth, or same-machine 4B-Q4 acceptance evidence. Those remain `NOT TESTED`.
