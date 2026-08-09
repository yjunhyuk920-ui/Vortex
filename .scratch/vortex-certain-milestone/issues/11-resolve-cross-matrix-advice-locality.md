# Resolve Cross-Matrix Advice Locality

Type: research
Status: resolved
Blocked by: Break the Bilinear Cross-Residual Barrier

## Question

Can one checkpoint-derived advice state of at most 8 GiB serve the registered
sequence of matrix-local exact Bilinear Cross Residual queries in a genuinely
nonseparable way, rather than paying a direct sum of per-matrix images?

In an explicitly declared coefficient- or word-probe model, either derive a
finite localization/direct-sum theorem that charges cancellation of
other-matrix content and outside-block probes, or specify a concrete global
code whose constructor and query recover each `r_i^T W_i u_i` exactly. The
complete equation must include representation, build amortization, query
localization, operations, logical and address-granular traffic, hot and cold
state, verification, miss, and fallback below the registered `1.185185185%`
target.

Do not reopen matrix-local separable linear covers, static synthetic circuits,
exhaustive tables, future target traces, external compute, training,
checkpoint modification, or hidden dense discovery. Do not assign an
experiment number or run a model/hardware Gate until a concrete E0 route
survives.

## Answer

Global linear advice has an exact localization law, but the resulting general
lower bound is insufficient to reject the target.

In the systematic linear coefficient-use model, write a block-local query as
`q_i=a_i+e_i`, where `a_i` is in the one global advice rowspace. Since `q_i`
is zero outside block `i`, every outside component of `a_i` occurs identically
in the charged raw residual `e_i`. With no outside probes, useful advice is
exactly `U intersect V_i`, and those shortened dimensions sum to at most the
global advice dimension. For a fixed outside support `T`, rank-nullity gives
local dimension at most `dim(U intersect V_i)+|T|`.

Projection ranks cannot replace this account. The finite counterexample
`U={(x,x)}` projects fully into two blocks while its shortened dimension is
zero in each. Query-dependent supports also form a union of fixed-support
images, so the theorem does not silently promote into a direct sum.

A global span-to-cover consequence remains valid. The `13/50` entropy witness
certifies covering radius `104,974,453,310`; every registered block tuple is a
sum of at most `16,384` aligned rank-one tuples. Therefore an arbitrary
independent registered query tuple needs at least:

```text
coefficient uses                    6,407,133
fraction of dense                   0.001586914271%
complete p50 allowance              1.185185185185%
allowance / bound                   746.848904934x
ideal packed payload                800,896 bytes
```

The last row is not address- or hardware-granular evidence. The target is far
above the lower bound, so no general impossibility follows; no concrete global
code supplies a complete constructor/query/runtime equation either.

Decision:

```text
ESTABLISH_GLOBAL_LINEAR_ADVICE_LOCALIZATION
REJECT_FREE_CROSS_MATRIX_PROJECTION_REUSE
DO_NOT_CLAIM_TARGET_SIZED_DIRECT_SUM_OR_GENERAL_IMPOSSIBILITY
KEEP_NO_SURVIVING_CANDIDATE
OPEN_CAUSAL_QUERY_RESTRICTION_CERTIFICATION_AT_E0
```

Nine focused tests and all 476 repository tests pass. No model, checkpoint,
server, GPU, experiment number, or hardware action occurred. Evidence:

- `docs/research/E0_CROSS_MATRIX_ADVICE_LOCALITY.md`
- `vortex_runtime/cross_matrix_advice_locality.py`
- `tests/test_cross_matrix_advice_locality.py`
- `results/e0_cross_matrix_advice_locality/summary.json`
- summary SHA-256 `e8957ad1a1e89e61ab7fb55b23df399e5d0bc066a4922d58a005012983c2aa14`
