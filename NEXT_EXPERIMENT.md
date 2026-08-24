# Next Experiment — EXP-107A

## Exact Distinct-State Shared-Weight-Sweep Gate

### Why this is next

EXP-106A closes fixed-width linear/coordinate/low-rank all-layer surrogates as
the arbitrary-checkpoint causal core. Resource-feasible narrow bridges collide
states; full-width bridges restore the original resource problem. The next
mechanism must therefore share **value-changing computation across distinct
states**, not assume that the states are equal or compressible.

### Three materially different principles

1. **Coded exact branch-superposition linear front end — selected first.** Pack
   several distinct causal branch states into a coded matrix, stream each target
   weight tile once, compute exact linear outputs, and decode every branch before
   the first nonlinearity. Charge the branch matrix, encoding/decoding,
   matrix-matrix MAC/adds, weight reads, output writes, and every evaluated
   branch `N`.
2. **Finite-word nonlinear-cell coalescence.** Merge work only when branches
   share an exactly identical RMSNorm/SiLU/rounding control cell. Charge cell
   discovery, normalization statistics, coefficients, split points,
   intermediates, and post-cell branch separation. Sign-pattern or hash equality
   is insufficient.
3. **Symbolic token-embedding contribution DAG.** Share checkpoint-static
   token-dependent subexpressions while prefix-dependent terms remain explicit.
   Charge program bytes, cold probes, online execution, nonlinear barriers,
   fallback, and exact target verification. It must differ from the rejected
   self-contained artifact/table family through a concrete cold-backed decoder.

### Cheapest local Gate

Before public-checkpoint execution, implement an exact finite-word branch-matrix
control and record:

```text
actual distinct branches N
useful committed tokens A
N/A
weight-tile bytes
branch activation/input/output bytes
MAC/add/encode/decode operations
nonlinear separation point and cell count
per-branch KV/state bytes
peak workspace <= 8 GiB
p50-equivalent byte and arithmetic roofline
```

Batching alone is not progress: reject if arithmetic remains proportional to
all branches and `N/A` cannot be <=1.5. A coded computation survives only if it
removes real branch-equivalent work before nonlinear separation while preserving
exact finite-word outputs.

After local validation, commit source, config, result, logs, checksums, README,
and handoff; do not duplicate the experiment in GitHub Actions unless explicitly
requested.
