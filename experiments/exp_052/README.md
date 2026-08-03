# EXP-052 — Exact Advice Tradeoff

This experiment audits exact target-specific prefix/KV-state advice under complete build, fallback, reuse, index, and storage accounting.

It is CPU E1 evidence. It does not execute 405B, measure 8 GiB VRAM, or replace an unseen-state target operation.

## Run

```bash
python -m pytest -q tests/exp_052
python scripts/run_validation.py
bash experiments/exp_052/run_current_env.sh
```

## Isolated reproduction

```bash
bash experiments/exp_052/reproduce.sh
```

The default reproduction output is `results/exp_052_reproduction`; frozen authoritative evidence must never be overwritten.

## Expected decision values

```text
CONTINUE_ENUMERATIVE_EXACT_ADVICE_TO_OPERATION_REPLACEMENT
REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY
```

Scientific rejection is a valid completed run. Integrity, exactness, causality, missing-state, and provenance errors are workflow failures.
