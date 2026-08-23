# VORTEX Research State

## Fixed target

Arbitrary public, unmodified Hugging Face dense 405B-class checkpoint; executor-only; one 8-GiB GPU; exact declared token and successor-state contract; same-machine native-4B-Q4 warm latency target `p50 <=1.2x`, `p95 <=1.5x`.

## Operating mode

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

GitHub Actions and duplicate hosted reproduction are not required unless the user explicitly requests them.

## Latest authoritative completed Gate — EXP-103A

```text
REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
```

EXP-103A implemented an exact BF16/FP32 native-rounding page certificate. It passed 200,000 randomized soundness cases with zero mismatches and positive controls, but the byte Gate failed decisively:

- one exponent byte per page consumes `1/(2Q)` of BF16 checkpoint traffic;
- the remaining p50 allowance requires roughly `98.86%–99.60%` of full pages to be skipped for `Q=64..1024`;
- a legal 16,384-wide all-ones BF16 row reads and executes `100%` of pages/terms at every registered Q;
- 64 Gaussian BF16 dense rows also read `100%` of pages at p50/p95.

The certificate remains auxiliary only. Public-checkpoint execution was not run because the universal finite-word byte Gate was already decisive.

## Previous authoritative Gate — EXP-102A

```text
REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

Real public draft/target checkpoints preserved exact token and terminal KV state, but holdout minimum committed tokens were only `1` and `2`, far below the registered raw-traffic requirement.

## Active next Gate — EXP-104A

`Checkpoint-Native Resident Slice Transducer Reality Gate`.

The next candidate changes the causal information source rather than tuning external drafts or no-op certificates. It must construct a real draft from a fully charged <=8-GiB resident subset/derived draft-only representation of the unchanged target checkpoint, stream exact embedding rows as needed, and verify with the unchanged exact target.

Before any backend work it must pass:

1. a complete static 8-GiB ledger, including draft weights/head, KV, workspace, metadata, and fragmentation;
2. actual public-checkpoint causal accepted-prefix measurement;
3. exact target token and terminal-state equality;
4. target-scale requirement `A >=339` for the raw BF16 p50 traffic floor, unless it measurably reduces the target sweep bytes;
5. fully charged draft, verification, rebuild, repair, and `N/A` costs.

## Not tested

- complete 405B target execution;
- physical complete 8-GiB allocation;
- target SSD/PCIe/HBM behavior;
- target CUDA/SASS;
- same-machine native-4B-Q4 p50/p95 acceptance.
