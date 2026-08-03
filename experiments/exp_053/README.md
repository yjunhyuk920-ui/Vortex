# EXP-053 — Bit-Exact Decision-Circuit Compiler

This Phase-A/B experiment compiles bounded binary-activation signed modular linear top-1 operators directly from weights into structurally hashed AIG circuits.

Runtime input truth tables are forbidden as the representation. Exhaustive input enumeration validates finite-domain equality only.

## Run

```bash
python -m pytest -q tests/exp_053
python scripts/run_validation.py
bash experiments/exp_053/run_current_env.sh
```

## Isolated reproduction

```bash
bash experiments/exp_053/reproduce.sh
```

Frozen authority must never be overwritten. CPU timing is environment-dependent; exact circuits, hashes, node counts, exhaustive assignments, and Gate decisions are primary fields.
