# EXP-082A -- Exact Differential Spanning-Tree Gate

Preregistered cheapest-kill experiment. Stage 1 computes a certified lower
bound on the best row/column Hamming spanning tree from exact 32-coefficient
block patterns. It must stop before building a tree if coefficient work alone
already exceeds the final target fraction.

The runner implements only Stage 1 plus exact small controls. No model-scale
MST constructor or sparse runtime is present. No result exists before the
source commit and clean evidence run.

Planned canonical command:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_082a\run_experiment.py --output-dir results\exp_082a
```
