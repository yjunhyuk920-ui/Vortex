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

The authoritative primary and reproduction bundles both return
`PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE`. They preserve
18/18 token states and 36/36 branches with mean/p95 KL
`0.007225545020063708/0.037660752986209814`; both independently rebuild the
same deterministic-core SHA-256
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`.

This pass is only permission to test a legal committed-pair center and
outward-rounded bound. It is not a target-free selector, E2 dense-operation
replacement, physical speed evidence, or evidence for 405B performance.
