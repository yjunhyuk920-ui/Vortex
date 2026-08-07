# EXP-081A -- Syndrome-Recovered Lookup MatVec

Throwaway prototype for one question: can an eight-stage nonlinear lookup
candidate leave an output error that a rank-eight syndrome code recovers on at
least 99.75% of held-out causal Transformer projection calls?

The pure finite-field logic lives in `vortex_runtime/syndrome_lookup.py`; the
tiny interactive shell lives in
`vortex_runtime/syndrome_lookup_prototype.py`. The batch runner embeds controls
and writes the evidence bundle. No production kernel is implemented.

Planned canonical command:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_081a\run_experiment.py --output-dir results\exp_081a
```

No result exists at preregistration time.

