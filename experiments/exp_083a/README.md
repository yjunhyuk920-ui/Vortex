# EXP-083A -- Causal Residual Atlas First-Decode Gate

EXP-083A executes the immutable contract in
`docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`. It uses only the
already pinned unchanged Qwen3.5-0.8B payload and the 18 evaluation prompts
from EXP-076.

For each prompt, prompt-only top-16 SVD bases are built for layer-11 `q_proj`
and `down_proj`. The first post-prefill decode call enumerates every contiguous
64-column page under the explicitly non-deployable native-anchored center. A
single valid branch failure triggers the frozen scientific rejection and stops
the run.

Canonical Windows command:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083a\run_experiment.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_083a
```

Independent evidence replay:

```powershell
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083a\verify_results.py `
  --output-dir results\exp_083a `
  --write-report
```

Until a result bundle is committed, status is `SOURCE IMPLEMENTED; NO MODEL
RESULT`. A pass is only permission to test a legal committed-pair center and
outward-rounded bound. It is not E2 or evidence for 405B performance.
