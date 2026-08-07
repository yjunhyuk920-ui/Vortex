# EXP-081A -- Syndrome-Recovered Lookup MatVec

Closed throwaway prototype for one question: can an eight-stage nonlinear lookup
candidate leave an output error that a rank-eight syndrome code recovers on at
least 99.75% of held-out causal Transformer projection calls?

The pure finite-field logic lives in `vortex_runtime/syndrome_lookup.py`; the
tiny interactive shell lives in
`vortex_runtime/syndrome_lookup_prototype.py`. The batch runner embeds controls
and writes the evidence bundle. No production kernel is implemented.

Canonical command:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_081a\run_experiment.py --output-dir results\exp_081a
```

Result: `REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH`. The exact
finite-field controls passed `321/321`, but held-out weighted exact coverage was
only `8.681672%` versus the `99.75%` Gate. The deterministic core is
`8621f6357536b6fc3396872668484c52103e28d2af8291b96575bc4d007c2ccc`;
authority is `results/exp_081a/summary.json`.
