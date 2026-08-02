# Session handoff

Last updated: 2026-08-02 (Asia/Seoul)

## What was done

The initial executable VORTEX prototype was assembled and validated locally before the repository upload.

Local verification command:

```bash
python -m pytest -q
python scripts/run_validation.py
```

Observed result at handoff:

```text
7 passed
validation script completed successfully
```

The committed validation report records:

- exact progressive LM-head certification for all recorded synthetic trials;
- exact disk-backed LM-head certification for all recorded tiny-checkpoint trials;
- exact Jacobi/sequential greedy sequence equality for all recorded tiny-checkpoint trials;
- a Llama 3.1 405B tensor-size plan;
- approximately 4.9 seconds total validation time on the handoff environment.

## Current code boundary

Working path:

```text
HF safetensors -> tensor discovery/slicing -> bounded cache
-> exact streamed tiny Llama -> progressive disk LM head
-> exact token certification
```

Not yet implemented:

```text
progressive internal Q/K/V/O and MLP projection execution
-> nonlinear bound/refinement propagation
-> fused GPU backend
-> real 8GB hardware benchmark
-> real 405B run
```

## Exact next task

Do not begin with a new speculative architecture document. Begin in code.

1. Define a linear operator interface in the runtime.
2. Refactor the exact Llama path so each projection can be supplied by an operator.
3. Add an instrumented progressive operator for one internal projection, starting with `o_proj` or `down_proj` because their output directly returns to residual width.
4. Run exact-versus-progressive comparisons on the tiny checkpoint.
5. Record per-layer residual fractions and final-token equality.
6. Expand to gate/up and then Q/K/V, adding nonlinear repair logic only after the linear measurements exist.

## Important observations

- A low-bit coarse LM head often selected the same token in the recorded tests, but certification cost varied significantly by bit width.
- Six-bit base precision produced far lower residual refinement than four-bit in the recorded synthetic run.
- Exact Jacobi decoding worked, but committed blocks were small; it is not yet the required hundred-token amortization mechanism.
- Fitting tensor windows under 8GB is only the residency problem. The project remains blocked on reducing or amortizing internal model traffic and compute.

## Files to update in the next session

- code and tests for the implemented milestone;
- `validation_results.json` if metrics change;
- this handoff file;
- `docs/ARCHITECTURE.md` for structural changes;
- `docs/ROADMAP.md` when a milestone gate is passed.
