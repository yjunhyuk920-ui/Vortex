# Preregister the Cheapest Surviving Gate

Type: task
Status: resolved
Blocked by: none

## Question

What is the smallest theorem, exact certificate, favorable oracle, or pinned
real-weight measurement that can decisively promote or reject the surviving
mechanism before a backend, large model, or target-hardware implementation?

## Answer

Freeze the [Causal Residual Atlas Cheapest Real-Weight Gate](../../../docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md): 18 pinned evaluation prompts,
teacher position 1, prompt-only rank 16, layer-11 `q_proj` and `down_proj`, one
64-column page selected by exhaustive exact-reference top-1/minimum-KL oracle.
The two widths expose 16/56 pages and 1,296 candidates. The target coverage
equation permits zero token failures: 18/18 overall, 3/3 per family, and 36/36
branches, with mean/p95 KL `<=0.02/0.05`. One failure rejects the registered
rank-16/page-64 path; a pass authorizes only a legal pair-center and outward-
bound Gate. Seven contract tests pass. No checkpoint was run and no experiment
number was assigned.
