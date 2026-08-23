# VORTEX Research Handoff

## Latest completed local Gate

**EXP-105A — Activation-Ordered Exact Residual-Bound Head Tournament**

```text
REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE
NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE
```

Evidence:

```text
results/exp_105a/local/result.json
results/exp_105a/local/checksums.sha256
docs/research/EXPERIMENT_105A_ACTIVATION_ORDERED_EXACT_HEAD_TOURNAMENT_GATE.md
docs/research/EXP_105A_LATEST_RESULT.md
```

Deterministic core:

```text
17cebccb51a2fbdadf871767151962b47e462bd14caeba1707cd66a64a468598
```

Validation:

```text
8 focused tests PASS
12 random Q4/Q8 queries x 6 block sizes, zero mismatch
runtime winner independent of dense reference helper
structured early-pruning positive control PASS
late-decision exact control reads 100% of head
byte-identical rerun PASS
SHA-256 ledger PASS
GitHub Actions not run
```

Interpretation:

- query-adaptive exact row elimination is possible on structured heads;
- dense random controls saved some weight payload, but selector state erased the saving;
- a legal late-decision head forces complete head reads;
- adding the minimum complete 405B-width layer yields `2.815515648 GB/token`, above the `2.4 GB/token` p50 budget;
- head-only bytes are not a complete executor because no causal state source or `A>=339` continuation was implemented;
- no-guess autoregressive dependencies serialize one weight-bearing traversal per token unless a transition operator or branches are supplied.

Do not reopen by changing only block size, activation ordering, residual-norm threshold, state-byte accounting, or the synthetic distribution.

## Next action

Run EXP-106A locally: depth-complete width-thin checkpoint surrogate. Freeze a complete training-free narrow representation and cost ledger first. Only then run a public checkpoint and measure actual accepted prefix plus exact target state.

## Operating rule

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Do not duplicate the locally validated experiment in GitHub Actions unless explicitly requested.
