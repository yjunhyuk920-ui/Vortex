# Constructive correlation-source continuation — 2026-09-08

Baseline: 66c01825635a357f86f9dfa5369088d7a239c92d, PR146.
Mission and CTC are unchanged. No target or public-model runtime result is claimed.

## Intended final theorem (not established)
For every checkpoint in the fixed public dense405B scope, every legal prefix,
input and RNG state, construct a finite unchanged-semantics executor preserving
the declared native output and all future successor-state behavior, with full
peak allocation <=8GiB and same-machine native4BQ4 p50<=1.2x,p95<=1.5x.
O1-O6 remain OPEN until the full construction and sufficient bounds exist.

## Three compared information flows

A. Nonlinear cold cells composed through an explicit joint-correlation source.
Replace the unjustified step 'child answers determine parent answer'. Generate
the actual joint-pattern partition, and evaluate all parent functions from
query-restricted pattern multiplicities. Selected for a bounded construction
because it addresses the missing input to E0_NONLINEAR_RECURSIVE_LIFT_GATE.
The intended >=10x route would require source + query work <=0.1 of direct
evaluation including pattern construction, IDs, queries and parent evaluation.
No such inequality is assumed. No qualifying generic speedup is claimed.

B. Move original native arithmetic into a globally contracted Boolean relation.
Eliminate intermediate variables using truth-table factors, retaining output,
RNG and live next-state roots. Finite elimination has sum(2^scope) table work;
a small induced scope would be needed but is not supplied. Compare exact
scope/cost against native forward evaluation before any compiler expansion.

C. Store a transformed causal state permanently and execute only its transition.
Need a constructed E,T',O' with E(T(s,u,r))=T'(E(s),u,r), O'=O, and original
RNG consumption. Prior nonlinear-coordinate controls already show this can work
for planted inverse pairs, not that it is available for an original Transformer.
No new native conjugacy or cheap output reconstruction is assumed.

These are distinct continuation principles, NOT three newly qualifying
discoveries. Existing mathematical tools and prior exclusions retain scope.

## Fixed bounded tests, before implementation/results

Correlation source: k-bit tuples at N sites, arbitrary Boolean function supplied
as ANF monomials (not a table of every checkpoint/input). Mask is r outer u over
GF(2). Compile the source once; never access raw child arrays during query.
Generator groups exact observed tuples, stores patterns and site identifiers.
The query computes pattern parities by visiting selected sites; parent answers
are XOR over occupied patterns. All reads and evaluation work are charged.

Test dimensions k in {2,4,8,16}, shape 4x4; all 256 pairs of 4-bit masks;
families diverse deterministic pseudorandom, repeated, constant; seeds 11,29.
Parent functions: every single bit, all quadratic monomials, and their XOR.
Independently compare direct source evaluation; mutate files to test rejection.
Record file sizes and logical byte references, NOT wall-clock performance.

The known two-site AND witness must be reproduced and repaired using joint
pattern counts. Do not credit re-proving that witness as a new theorem.

After A, implement an exact ordered reduction over pattern identifiers to test
whether an unordered histogram suffices for original FP32 arithmetic. Use the
fixed [2^24,1,-2^24,1] / [2^24,-2^24,1,1] witness and exact dyadic reference.
Preserve original positions for the valid constructor; count every ID read and
native reduction. This is a genuine paid continuation, not a free native lift.

No matrix-size/seed sweep after a failure, no SAT-timeout enlargement, no public
weights download, no target-server action, no Actions, no CUDA benchmark.
At a session boundary preserve all open obligations and an actionable next
construction. Neither this preregistration nor its tests complete the mission.
