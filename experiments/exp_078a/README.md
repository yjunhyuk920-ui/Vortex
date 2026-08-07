# EXP-078A runner

This is a deliberately favorable, throwaway lifetime prototype for an exact
anchor-conditioned MLP macro-operator. It answers one question: does the exact
operator induced by the last prompt token remain accurate even for the next
seven causal positions?

Inspect the construction-amortization state interactively with one command:

```powershell
.deps\exp076-venv\Scripts\python.exe -m vortex_runtime.tangent_macroblock_prototype
```

Run the pinned unchanged-checkpoint Gate from the repository root:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_078a\run_experiment.py --model-dir .deps\exp076-model --output-dir results\exp_078a
```

The runner does not download anything, contact the private Ubuntu target, or
physically materialize a macro matrix. The candidate reference executes the
factorized up/frozen-coefficient/down expression so quality can be measured;
direct macro-construction and hot-application costs are derived separately.
