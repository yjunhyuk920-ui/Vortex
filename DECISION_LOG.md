# VORTEX Decision Log

Append-only decisions. Authoritative run identities are read from committed result JSON.

## D-001 — Final target fixed

Arbitrary public unmodified Hugging Face dense model; runtime only; real 405B; <=8 GiB VRAM; original contract preserved; 4B-class user experience; independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

GitHub Actions CPU cannot be labeled target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, power, or peak-VRAM evidence.

Status: ACTIVE. Phase D NOT TESTED.

## D-003 — Phase A/B/C/D adopted

Status: ACTIVE.

## D-004 — E0–E7 adopted

Status: ACTIVE.

## D-005 — MEASURED/DERIVED/PROJECTED/UNVERIFIED separation adopted

Status: ACTIVE.

## D-006 — mmap/index/DAG components are auxiliary

Evidence: PR #50/#52/#54.

Status: ACTIVE.

## D-007 — Raw prefix enumeration rejected as core

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED.

## D-008 — Exact future-DAG accepted only as body compression

Evidence: 64->38 nodes; causal held-out start coverage 0%.

Status: AUXILIARY.

## D-009 — Core research must skip or amortize original operations causally on unseen prompts

Status: ACTIVE.

## D-010 — EXP-047 CPTC selected and executed

Status: EXECUTED.

## D-011 — EXP-047 correctness primitive accepted at E1

`results/exp_047/summary.json` records workflow `30793232558`, 525 cases, zero wrong accepts, zero fallback/reference mismatch, zero independent-bound mismatch, and 15/15 adversarial exact fallbacks.

Status: ACCEPTED E1 PRIMITIVE ONLY.

## D-012 — Global-range CPTC-v1 not promoted

MEASURED: 4/525 certified, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python reference about 8.6–9.1x full summation.

PROJECTED: required target fraction 1.185%; positive control remained 8.817x above it.

Status: REVISE.

## D-013 — Oracle-tight/stratified audit required before further CPTC backend work

Status at registration: EXECUTED BY EXP-047R.

## D-014 — Freeze completed experiment measurements

Completed workflows must not silently overwrite frozen evidence. Reproduction writes to isolated output.

Status: ACTIVE REPRODUCIBILITY RULE.

## D-015 — EXP-047R completed with valid E1 evidence

Authority: `results/exp_047r/summary.json`, workflow `30795946233`, artifact `8848886335`.

MEASURED: three pinned checkpoints, 18 states, zero wrong accepts/bound violations, C1 oracle median/p90 100%, C2 median/p90 100%, C2 best 99.21875%.

Status: EXECUTED, E1.

## D-016 — Range-based CPTC rejected as core

The exact realized min/max oracle evaluated 100% at median and p90 versus pre-registered limits 10%/25%. Further range metadata or C3 variance adaptation cannot rescue the declared mechanism without changing class.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

Status: REJECTED AS CORE; AUXILIARY PRIMITIVE RETAINED.

## D-017 — Next core Gate changed from weight skipping to block amortization

A zero-cost perfect proposal requires at least 85 exact tokens/full target pass to meet the projected 1.185% target-equivalent fraction.

Status: EXECUTED BY EXP-048.

## D-018 — EXP-048 exact block verifier accepted at E1

Authority: `results/exp_048/summary.json`, workflow `30798936320`, artifact `8850040445`, ZIP SHA-256 `67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9`.

The longest-prefix plus first-mismatch correction verifier preserved exact greedy output in reference tests and committed cases. Predictions after first mismatch were never committed.

Status: ACCEPTED E1 AUXILIARY PRIMITIVE.

## D-019 — Perfect block proposal proves arithmetic only

B1: 96 future-aware exact tokens/one target pass, fraction 1.0416667%, exact 18/18, deployable false.

Status: ACCEPTED NON-DEPLOYABLE UPPER BOUND ONLY.

## D-020 — Hard Jacobi and partial-layer self-draft rejected as core

B2 hard Jacobi p50 58 target passes/32 exact tokens, p50 181.25%, p90 193.75%.

B3 partial-layer self-draft p50 committed tokens 1, maximum matching prefix 1, minimum accounted fraction 1333.463%, p90 2893.843%.

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

Status: B2/B3 REJECTED AS CORE; VERIFIER RETAINED.

## D-021 — Next Gate removed the sequential draft loop

EXP-049 tested continuous soft-token block states, damped Picard, bounded Anderson acceleration, exact hardening/verification, and an adversarial triangular dependency audit.

Status: EXECUTED BY EXP-049.

## D-022 — EXP-049 reference solvers and lower-bound harness accepted at E1

Authority:

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

MEASURED correctness/safety:

```text
9 EXP-049 tests passed
repository validation passed
18 checkpoint cases
selected exact mismatches 0
selected future-information uses 0
unhandled numerical failures 0
S3 alignment failures 0
```

Status: PICARD/ANDERSON REFERENCE AND FAIL-CLOSED NUMERICAL MACHINERY ACCEPTED E1 AUXILIARY.

## D-023 — Target-only continuous fixed-point generation rejected as core

EXP-049 allowed the exact reference to select the best pre-registered S1/S2 variant, block size, and pass count per case.

MEASURED:

```text
oracle-best p50 exact proposal prefix 4.5
oracle-best maximum prefix 6
oracle-best p90 target-equivalent fraction 168.778596%
S0 hard Jacobi p50 prefix after four passes 4
S2 Anderson p50 prefix after four passes 1
S2/S0 improvement 0.25x
model medians 4.5 / 5.0 / 4.0
```

The favorable selector failed the 16-token and 10% early thresholds.

Status: PERFORMANCE GATE FAILED.

## D-024 — Hidden triangular target-round barrier accepted within declared interface

Two adversarial causal chains recorded Picard prefixes `1,2,3,4` and Anderson prefixes `1,2,3,3`. Two targets with different hidden suffixes produced indistinguishable suffix transcripts before predecessor resolution.

Claim scope:

- one synchronous black-box causal target block evaluation per round;
- exact prefix, fixed initialization, all prior states/outputs, and arbitrary continuous/Anderson arithmetic allowed;
- no external future information.

Within this scope, a solver cannot universally guarantee more than one newly resolved exact token per target round for every arbitrary causal target.

Status: ACCEPTED E1 ADVERSARIAL LOWER-BOUND CONSTRUCTION; NOT A CLAIM ABOUT EVERY AVERAGE CHECKPOINT PROMPT.

## D-025 — EXP-049 core decision

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Exactness, causality, numerical safety, and model-size trend passed. Prefix, traffic, Anderson-improvement, and universal-barrier Gates failed.

Status: REJECTED AS CORE.

## D-026 — Next Gate imports target-independent external advice

EXP-050 will use other already published, unmodified checkpoints as causal draft models and retain exact block verification. It will charge one complete draft stream per proposed token and one target verification stream per block.

Universal boundary: a target-independent draft can be contradicted at the first token by an arbitrary target. Practical fixed-pool evidence is therefore separate from the universal claim.

PROJECTED 4B-draft/405B-target arithmetic:

```text
draft cost per token = 4/405 target streams
required total fraction = 0.01185185185
minimum completely correct proposal length after draft cost = 507 tokens
```

Status: ACTIVE NEXT GATE — EXP-050.
