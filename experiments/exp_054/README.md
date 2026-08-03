# EXP-054 — Reduced Ordered Decision Diagrams

This Phase-A/B experiment compiles bounded signed modular linear top-1 operators into exact reduced ordered multi-terminal decision diagrams.

Natural and deterministic weight-magnitude orders are both compiled and charged. A compile ceiling is a scientific fallback row, not an infrastructure exception.

## Run

```bash
python -m pytest -q tests/exp_054
python scripts/run_validation.py
bash experiments/exp_054/run_current_env.sh
```

## Isolated reproduction

```bash
bash experiments/exp_054/reproduce.sh
```

No real Transformer operation or target hardware is measured.
