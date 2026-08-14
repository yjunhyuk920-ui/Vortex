# EXP-082A -- Exact Differential Spanning-Tree Gate

Preregistered cheapest-kill experiment. Stage 1 computes a certified lower
bound on the best row/column Hamming spanning tree from exact 32-coefficient
block patterns. It must stop before building a tree if coefficient work alone
already exceeds the final target fraction.

The runner implements only Stage 1 plus exact small controls. No model-scale
MST constructor or sparse runtime is present. The result was produced from the
clean source commit and recorded separately in the evidence commit.

Authoritative result:

```text
REJECT_DIFFERENTIAL_SPANNING_TREE_FROM_CERTIFIED_LOWER_BOUND
weighted coefficient lower bound  1.562367394%
target                             1.185185185%
```

The lower bound already fails before any sparse metadata or memory cost, so
Stage 2 and the tree constructor were not run.

Canonical command:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_082a\run_experiment.py --output-dir results\exp_082a
```
