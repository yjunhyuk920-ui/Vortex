# VORTEX Research Handoff

## Current branch handoff

Latest completed local Gate: **EXP-103A Hierarchical Native-Rounding Page Certificate**.

Decision:

```text
REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
```

Evidence:

```text
results/exp_103a/local/result.json
results/exp_103a/local/checksums.sha256
docs/research/EXP_103A_LATEST_RESULT.md
```

Deterministic core:

```text
d912ff35377f34d76926c9ca1a887a92c51eb0f69899b8621340650b67b7c363
```

Validation:

```text
200,000 randomized cases
53,676 certified pages
0 certificate mismatches
6 focused tests passed
byte-identical deterministic rerun
all-ones finite-word negative control: 100% pages read
Gaussian dense control p50/p95: 100% / 100% pages read
```

Interpretation: finite precision creates real no-op products, but a page-summary executor must skip roughly 99% of weight pages after paying metadata. An arbitrary legal dense row can force every page to matter. Do not reopen this family by changing page size, exponent encoding, or threshold; reopening requires a value-changing cross-page computation, not a finer no-op selector.

## Next action

Run EXP-104A locally: checkpoint-native resident slice transducer. Freeze the actual <=8-GiB representation before checkpoint execution, then require real causal accepted prefix, exact target token/state, and full cost accounting. Actual target-scale `A >=339` is required unless exact sweep bytes are reduced.

## Operating rule

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Do not duplicate the validated experiment in GitHub Actions unless the user explicitly asks.
