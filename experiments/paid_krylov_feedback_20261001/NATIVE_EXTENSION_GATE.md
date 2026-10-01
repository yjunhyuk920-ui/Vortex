# Native extension gate after the constructive primitive

This is a source-grounded missing hypothesis, not an admitted new core or a
request to rerun the already tested native graph.

## Actual preserved source boundary

The pinned Llama-family reference source is already preserved in
../native_global_transition_20260908/results/reference_sources. In
layer_forward.txt, input normalization precedes attention, an actual residual
addition follows, then post-attention normalization, MLP and a second residual
addition. In attention_forward.txt, original q/k/v projections act on those
current hidden states, RoPE acts on q/k, and the new k/v are passed to cache
update before attention and the original o projection.

Consequently the second layer's new KV depends on the first layer's nonlinear
attention/MLP result, not just the current token. This was already evidenced by
../causal_cut_20260907/REPORT.md; repeating that counterexample is not new research.
The existing native graph/demand executors also retained every matrix MAC, so
ordinary tracing or backward slicing has already failed to supply this source.

## Precise hypothesis worth testing only after a producer is supplied

For a fixed original native cut t, seek a source-derived bit-linear state chart
K_t, cheap transition J_t, narrow innovation injection B_t and executable
eta_t, with

  bits(F_t(decode(K_t z), a, rho))
    = K_(t+1)[J_t z XOR B_t eta_t(z,a,rho)],

on every legally reachable represented state. Construct an equally explicit
original observer, and include KV/layout/control/RNG or their proved exact
parallel recurrence. Fixed bounded padding/shape handling, initial encoding,
chart changes and every required readback count. The already implemented
synthetic primitive is the special case of a constant chart, one innovation
bit, three scalar observers and an AND/XOR eta.

The new content needed is an original-source native instruction program for
eta_t and the observer that avoids the eliminated dense region, with a paid
size/query bound. Small output rank, cached prefixes, a field-exact guard, or
running the original region then subtracting its linear part is insufficient.

The cheapest falsifiable check for a supplied proposal is:

1. Freeze the original cut, ABI, reachable-state invariant, full observation
   roots and the candidate constructor/eta/observer program before results
2. Verify the native intertwining identity and original output/RNG/state relation
   on the supported smallest case, including the first source-generated legal
   nonzero innovation; use exact words, not fitted real residuals
3. Audit the candidate's source reads and instruction dependency graph: if eta
   or observer calls the original eliminated matrix region, the proposal has
   not supplied the source. Stop before model/scale runs
4. If a rank-k linear-bit innovation subspace is claimed, k+1 independent native
   second-difference vectors from legal quadruples refute that specific chart;
   absence of a witness does not prove closure or cheap construction

A particularly tempting candidate is to use old KV as the linear backbone and
new KV as the narrow innovation. That is legitimate state bookkeeping, but the
preserved reference source shows that its unmodified innovation producer is the
whole causal layer sequence. Without a newly supplied producer, that candidate
is not new and does not qualify for another numerical experiment.

No concrete cheap native eta/observer constructor for the original HF body has
been found in this assignment. The formal relation alone must not be reported
as a solution or as three new qualifying principles. Continue source discovery;
do not expand the parity toy or relabel append-only cache writes as acceleration.
