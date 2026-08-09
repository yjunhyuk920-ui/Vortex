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
$env:PYTHONPATH = ".;.deps"
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

Before the canonical command there is no expected scientific decision. A pass
would authorize only a separately preregistered backward-layer/position Gate.
It is not E2 operation replacement, physical performance evidence, or evidence
for 122B/405B.
