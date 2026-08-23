# VORTEX Research Handoff

## Latest completed local Gate — EXP-104A

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

Evidence:

```text
results/exp_104a/local/result.json
results/exp_104a/local/checksums.sha256
docs/research/EXP_104A_LATEST_RESULT.md
```

Deterministic core:

```text
dcdce92c76ca84154d12e87cbfd904e1fdfe637cdbd0c8ea51d55e13605a6f93
```

Validation:

```text
7 focused tests passed
registered 405B population matched exactly
byte-identical rerun
compile PASS
checksum ledger PASS
```

Decisive fact:

```text
ideal Q4 full head + one layer = 2,644,508,672 bytes/token = 1.322254336x native 4B Q4
real Q4 full head + one layer  = 2,809,790,464 bytes/token = 1.404895232x native 4B Q4
required p50                   = <=1.2x
```

Acceptance cannot amortize the draft scan because every proposed token requires one autoregressive resident pass.

## Next action

Run EXP-105A locally. Compare a genuinely causal sub-full-head index, a single-scan multi-token transducer, and a cross-matrix decision-bit program. Implement only a principle with an explicit fully charged route below the `2.4 GB/token` p50-equivalent byte floor.

## Operating rule

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

No duplicate GitHub Actions run unless the user explicitly requests it.
