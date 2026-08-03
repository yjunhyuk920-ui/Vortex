# EXP-051 — Layer Finalization and Tail-Skip Gate

Model-independent tests:

```bash
python -m pytest -q tests/exp_051
```

Pinned CPU oracle audit:

```bash
bash experiments/exp_051/run_current_env.sh
```

Isolated reproduction:

```bash
bash experiments/exp_051/reproduce.sh
```

The runner performs prompt prefill plus one exact warm-up generated token, then records 64 incremental warm decode states per target/prompt. Every target block is executed. Intermediate states are passed through the original GPT-Neo final norm and LM head; the final hidden-state entry is already final-normalized and is not normalized twice.

Suffix-stable depth and per-state depth selection use complete later-layer reference outputs. They are non-deployable E1 upper bounds, not real operation replacement.

405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, and real skipped-layer execution are NOT TESTED.
