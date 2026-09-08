# Continuation, frozen before consumer-quotient execution

Date: 2026-09-08. Parent 66c01825635a357f86f9dfa5369088d7a239c92d.
The previous uncommitted source and its initial hashes are evidence, not new work.
The old source visits all selected position IDs, and its ordered continuation
reads all already-produced terms. Neither is a qualifying core executor.

Next concrete construction: specialize joint information to the fixed downstream
consumer family, rather than construct the entire pattern histogram on each query.
For observed patterns P, express each supplied Boolean function f as a constant
plus a linear combination (over F2) of selected centered functions b_j on P.
Construct bitplanes b_j(p_i) from the actual original source. For query mask q,
read packed words of these planes and compute the needed parities. Store the
actual output coefficient masks and constants. No ideal histogram or predictor.

This is a strengthened implementation of flow A, not a fourth novel principle.
It uses elementary function-space linear algebra and static partial evaluation.
B (global native relation elimination) and C (persistent conjugate state) remain
unconstructed; no new cheap separator/chart is assumed and no B/C benchmark ran.

Target link: tests whether a producer can preserve exactly the cross information
needed by known consumers without a positional-ID scan. It does NOT establish
that input-dependent native product terms are already available, that a fixed
consumer family represents all legal continuations, or that ordered FP reductions
are interchangeable with parities. Original model O1-O6 therefore stay OPEN.

Intended exact identity: f(p)=c_f XOR sum_j d_fj*b_j(p) on every observed p.
Then XOR_i q_i*f(p_i)=c_f*parity(q) XOR sum_j d_fj*XOR_i q_i*b_j(p_i).
Uniform constructor: enumerate observed patterns only, evaluate supplied ANFs,
Gaussian eliminate the centered rows, and materialize the selected bitplanes.
No enumeration of all current input masks is needed by the constructor.

Cost route: if rank r is much less than source word width k, immutable source
plane reads can be r/k of the original. This is NOT a full-work inequality;
query-mask traffic, coefficient masks, output, loading, generating planes, all
temporaries and initialization must also be charged. If all identity-bit
consumers are retained, their rank can prohibit this route for this format.
No assumption r<<k is adopted. This route is conditional and not core-admitted.

Before results: use original 24 cases/all256 masks; also k=8,16,32,
N=1024, seeds11,29, families (single parity of pairwise ANDs) and (same function
plus every original input bit). Include patterns 0 and every unit bit so the
all-bit family is not spuriously low rank. Query masks: zero, full, one-bit,
alternating, and 12 deterministic random masks. Do not increase these sizes
after results. Compare against an independent direct ANF evaluation, inspect
file and query costs, and retain each actual encoded file and query result.

If retained functions restore rank or full native ordering is still missing,
do not tune rank thresholds, drop required consumers, or call it a model engine.
The next essential gap then is a native, input-dependent producer whose cost is
small even with all required output/RNG/state obligations retained. This plan
does not claim that gap has a solution. Hardware/public model: NOT TESTED.
