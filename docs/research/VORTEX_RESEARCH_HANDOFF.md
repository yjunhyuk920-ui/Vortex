# VORTEX Research Handoff

## Current authoritative local result

EXP-107A froze three prior-screen batches and executed the two cheapest arithmetic kills.

```text
REJECT_CURRENT_680_EXPLICIT_FMM_CATALOG_COMPOSITIONS_AS_10X_CORE
REJECT_REGISTERED_NAIVE_RANK46_APA_INTERPOLATION_EXACTIFICATION
FINITE_STRASSEN_CALCULUS_DIRECT_SUM_EXTRACTION_CHEAP_KILL_ONLY
```

Evidence:

```text
experiments/exp_107a/reproduce.py
experiments/exp_107a/config.json
results/exp_107a/local/result.json
results/exp_107a/local/checksums.sha256
docs/research/EXPERIMENT_107A_RESEARCH_PRIOR_AND_CURRENT_FMM_ENVELOPE_GATE.md
docs/research/EXP_107A_LATEST_RESULT.md
```

Deterministic core:

```text
8401542be0615e32d25600fd35499535bf9f5a5e834bc43e65dbfc5614399c6e
```

Validation:

```text
canonical reproduce.py self-test PASS
6 independent focused tests passed
byte-identical deterministic rerun
Python compile PASS
SHA-256 ledger PASS
integrity failures []
GitHub Actions not run
```

## Scientific interpretation

The current 680-scheme exact FMM catalog cannot be rescued by pure Kronecker/mixed recursion or independent partitioning: an unrealistically favorable unit-constant envelope already retains `12.515087006%` arithmetic. The complete target needs normalized exponent `<=2.7700683089`, while the current catalog minimum is `2.792481250`.

Rank-46 APA is algebraically strong enough only before exactification. Generic 22-evaluation interpolation is rejected, but finite nontrivial direct-sum extraction remains open.

## Next action

Run EXP-108A locally. Produce one finite exact Strassen-calculus/direct-sum schedule and count all operations and constants. Do not build a kernel unless it crosses the `10%` model-wide arithmetic line. If it fails, continue the repeated three-principle batch loop rather than ending with no survivor.
