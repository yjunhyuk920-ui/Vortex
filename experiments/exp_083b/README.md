# EXP-083B — Legal Pair and Outward-Bound Last-Down Gate

EXP-083B implements the immutable one-shot contract in
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`.

It uses the already pinned unchanged Qwen3.5-0.8B BF16 payload and a new
SHA-pinned population of 24 prompts. The current layer-23 `down_proj` candidate
is built only from committed prefix input/image pairs, selects one page from
the current residual without a weight/output/logit argument, and must certify
the final native greedy token with outward bounds. The first valid unresolved
row executes dense completion and rejects the frozen path.

Implementation commit:
`ecf753d7352bcca47767d0fe91c46d84cca66b59`. Frozen config SHA-256:
`2c8bdf12535e18327f0a4116e8f9dc4b918f900208d405972cfa421764bdd5c1`.

Canonical one-shot Windows command, only after the source commit exists:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083b\run_experiment.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_083b
```

Independent evidence verification reads the pinned weight but performs no
model forward:

```powershell
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083b\verify_results.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_083b `
  --write-report
```

The runner/verifier use the pinned virtual environment's own packages. Adding
the repository-level `.deps` package directory to `PYTHONPATH` would shadow
that lock with an older Transformers build and is therefore invalid.

Before the canonical command there is no expected scientific decision. A pass
would authorize only a separately preregistered backward-layer/position Gate.
It is not E2 operation replacement, physical performance evidence, or evidence
for 122B/405B.

## Result

The one-shot run stopped on `legal_holdout_english_01`: rank 16 and target-free
page 0 were valid, but the strict final certificate was unresolved and exact
dense completion executed. Controls, leakage failures, and false accepts were
all zero. The authoritative decision is
`REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH`.

Independent no-forward verification passed and rebuilt deterministic-core
SHA-256
`57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.
Evidence is under `results/exp_083b`.
