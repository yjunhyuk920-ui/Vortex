# EXP-076 runner

This directory contains the preregistered CPU reference Gate for the unchanged
`Qwen/Qwen3.5-0.8B` checkpoint at revision
`2fc06364715b967f1860aea9cf38778875588b17`.

The model payload is intentionally excluded from Git. Download is allowed only
after the source/preregistration commit:

```powershell
& '.deps\exp076-venv\Scripts\python.exe' experiments\exp_076\download_checkpoint.py
```

Canonical local run from Git Bash:

```bash
EXP076_PYTHON='C:/dincAI/Vortex/.deps/exp076-venv/Scripts/python.exe' \
EXP076_MODEL_DIR='C:/dincAI/Vortex/.deps/exp076-model' \
EXP076_OUTPUT_DIR='C:/dincAI/Vortex/results/exp_076' \
bash experiments/exp_076/run_current_env.sh
```

The runner uses one causal K=64 native-MTP path per prompt and derives the
registered K values `2,4,8,16,32,64` from its prefixes. Build prompts select one
fixed K by the committed minimax rule; only held-out evaluation prompts decide
the Gate.

The reference keeps committed target, recursive proposal, verification, and
post-rejection states separate. Verification and commit-state reconstruction
operate on independent hybrid-cache clones. Any future-token read, wrong
accept, cache mutation, replay mismatch, malformed manifest, or non-finite
metric fails closed.

This is not a vLLM performance benchmark. CPU time and RSS are environment
observations. The 35B/122B models, private Ubuntu server, GPU backend, page
scheduler, Phase D, and dense-405B claims remain prohibited.

## Result

The canonical run selected `K=4` and rejected the candidate. Held-out
accepted-prefix p05/p50/p95 was `0/4/4`, versus required p05/p50 minima of
`9/11`; 2/18 cases accepted no proposal token. All registered causality,
acceptance, committed-cache, and rollback controls passed. See
`results/exp_076/summary.json` and the authoritative experiment document for
the complete claim boundary.
