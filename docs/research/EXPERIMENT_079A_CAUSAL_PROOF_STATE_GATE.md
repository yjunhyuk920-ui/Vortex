# EXP-079A -- Causal Proof-State DCT Block-Zonotope Gate

## Status

Preregistered. No model output has been interpreted. The source commit must be
recorded before the pinned checkpoint runner is executed.

## Question

Can one concrete Causal Proof-State Machine (CPSM) primitive make complete MLP
down projections compatible with the final 405B/4B-equivalent budget while
remaining fail-closed?

The primitive keeps a tiny exact image of a procedural pilot subspace hot. The
unrepresented original checkpoint remains cold and is not discarded: it is a
deferred exact dependency represented by disjoint output-row L2 proof balls.
Current-token activations may trigger complete cold row-block reads. A token may
commit only after its downstream decision proof closes; otherwise the unchanged
projection is the exact fallback.

EXP-079A is the cheapest necessary Gate. It does not yet implement the nonlinear
proof propagation needed to certify a token.

## Mechanism

For a dense down projection `W` with `m` output rows and `n` input columns, let
`U` be the first `k` orthonormal DCT-II directions. `U` is procedural and needs
no checkpoint-specific stored basis.

```text
A = W U
E = W - A U^T
W x = A(U^T x) + E x
```

The hot sidecar stores `A` and one conservative Frobenius norm for each disjoint
row block of `E`. For row block `r`:

```text
||(E x)_r||_2 <= ||E_r||_F ||x||_2
```

Because row blocks occupy disjoint output coordinates, their squared L2 radii
sum without replacing them by independent scalar intervals. Reading a complete
cold row block resolves that block exactly. The local proof state is therefore
a product of correlated L2 balls, not an uncorrelated per-logit range table.

The candidate forward uses a deliberately impossible selector: it first sees
the complete exact residual and retains the row blocks with the greatest actual
residual L2 energy. A separate favorable oracle removes the largest certified
block radii when reporting the minimum possible local sound radius. The two
oracles may choose different blocks. This can only help the candidate and avoids
rejecting it because of a poor selector implementation.

## Difference from closed families

- EXP-047/047R and EXP-068 accumulated independent absolute unread bounds. This
  Gate keeps every coordinate within one row block correlated as an L2 set and
  never reads a complete low-bit base first.
- EXP-058 rejected exact low rank. Here the pilot is not claimed to reconstruct
  `W`; `E` remains an exact cold dependency with charged whole-block reads and a
  mandatory fallback.
- EXP-069 required an exact prior-activation span hit. Here every miss survives
  in the proof state rather than being silently approximated or called a hit.
- EXP-077A dropped nonzero intermediate channels. Here no residual is declared
  absent: it is bounded, selectively resolved, or sent to exact fallback.
- EXP-078A reused a prior token's frozen operator. The pilot is static checkpoint
  data and every current causal activation gets new coefficients and bounds.

This reopens only the cold-backed DCT-pilot/block-zonotope interface. A failure
does not prove every correlated certificate or online matrix data structure
impossible.

## E0 scorecard

1. **Target-scale upside.** The hot image costs approximately `4k/n` of an
   ideal packed-Q4 matrix when stored in FP32. At fixed `k/n`, pilot application
   and image traffic remain a small constant fraction. Complete cold row blocks
   consume the remainder of the final budget.
2. **Novel information source.** The current causal activation and selectively
   queried original row blocks supply query-time information absent from a
   self-contained hot artifact.
3. **Evidence basis.** DCT orthogonality gives an exact algebraic split and the
   Frobenius inequality gives a sound local enclosure. Whether real Transformer
   activations and weights make it useful is deliberately unassumed.
4. **Scaling reason.** A fixed pilot rank becomes cheaper as width grows; the
   target projection instead preserves the small-checkpoint rank fraction, so
   useful pilot dimension grows with width. Spectral usefulness at scale remains
   unverified and is not a success premise.
5. **Universality.** The basis, sidecar, bounds, and row layout are generated
   mechanically from any dense matrix; there is no training, adapter, prompt
   table, or user-authored model rule.
6. **Correctness closure.** An open proof cannot commit. Exact full projection,
   cache repair, and token replay are mandatory fallback semantics.
7. **Resource closure.** The Gate counts pilot-image bytes, norm metadata,
   per-matrix metadata, proof records, pilot/application operations, and selected
   ideal-Q4 row reads. Selector, downstream propagation, and fallback are
   explicitly free favorable grants; consequently a pass cannot establish a
   deployable runtime.
8. **Cheapest falsification.** Run the unchanged local 0.8B checkpoint on the
   frozen 192-decision corpus. If the dual oracles still miss quality, local
   proof-radius, or byte thresholds, stop before any nonlinear interpreter,
   fallback engine, kernel, download, or target-server work.

## Registered budget equations

Final target-equivalent fractions are fixed before the run:

```text
p50 = 1.2 * 4 / 405 = 0.011851851851851851
p95 = 1.5 * 4 / 405 = 0.014814814814814815
```

For `L` equal down matrices:

```text
Q4_total            = ceil(P_total * 4 / 8)
sidecar/matrix       = m*k*4 + ceil(m/R)*4 + 32 bytes
cold row-block       = ceil(R*n*4/8) bytes
proof state          = L*ceil(m/R)*16 bytes
charged traffic      = all sidecars + L*integer cold blocks
charged operations   = L*(n*k + m*k + ceil(m/R) + selected_rows*n)
```

The small Gate uses `k=4`, `R=4`, all 24 SwiGLU down projections, and the actual
text parameter count. Every other target weight, operation, and byte is free,
so the full whole-model allowance is concentrated on this one matrix family.

The shape-only 405B projection uses `P=405B`, `L=126`, `m=16,384`, `n=53,248`,
`k=59`, and `R=64`. It must keep sidecar plus proof records below the complete
8 GiB envelope and charged logical traffic/operations below the corresponding
p50/p95 fractions. This is not measured VRAM or throughput.

## Registered quality and proof Gates

The unchanged BF16 control must match all 192 frozen target decisions. The p50
arm is authoritative and must satisfy all of:

- charged logical traffic no greater than `1.185185185%` of total target Q4;
- held-out top-1 agreement at least 99%;
- every required family at least 95% top-1 agreement;
- held-out mean target-to-candidate KL at most 0.02;
- held-out p95 KL at most 0.05;
- p50 and p95 of the separately optimized minimum local sound-radius ratio no
  greater than 1.0 of the exact MLP-output L2 population.

The `1.0` radius screen is intentionally weak: a local uncertainty ball as large
as the signal is not a token certificate. Passing merely authorizes construction
of a nonlinear margin propagator. The p95-budget arm is diagnostic and cannot
rescue a failed p50 arm.

## Decisions and stop rule

```text
PROMOTE_TO_NONLINEAR_CAUSAL_PROOF_PROPAGATION_GATE
REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH
INVALID_CAUSAL_PROOF_STATE_CONTROL_FAILURE
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

Scientific failure prohibits post-result rank, DCT-basis, block-size, layer,
prompt, family, or fraction sweeps around this mechanism. Do not implement a
nonlinear interpreter, selector, CUDA kernel, 122B/405B run, or target-server
benchmark after failure. A continuation requires a new information source that
directly addresses the measured failure.

## Strongest counterexample

For a dense matrix whose energy is nearly isotropic outside the pilot and a
dense activation orthogonal to the pilot, every row-block radius is similar.
Resolving roughly one percent of blocks leaves nearly all residual energy open.
The synthetic reference checks soundness and fail-closed transitions; the real
Gate measures whether the checkpoint behaves materially better.

## Causality and evidence ceiling

No future generated target token is visible. The quality oracle observes only
the current projection's exact residual, which is unavailable to a deployable
selector and is explicitly granted free.

Evidence is capped at Phase B/C E1 on one unchanged small checkpoint. Sampling,
complete-model proof propagation, fallback timing, real Q4 fidelity, CUDA,
physical I/O, 8 GiB peak VRAM, wall-clock, 122B, arbitrary dense 405B, and E2-E7
remain **NOT TESTED**.
