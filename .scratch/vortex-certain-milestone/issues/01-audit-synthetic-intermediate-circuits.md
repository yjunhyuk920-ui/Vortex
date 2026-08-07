# Audit Synthetic-Intermediate Exact Circuits

Type: research
Status: resolved
Blocked by: none

## Question

Does an exact Hamming/linear circuit with compiler-created Synthetic
Intermediates define a materially new cold-backed execution class after
EXP-053, EXP-054, EXP-072A/072B, and EXP-082A, and can any surviving class pass
an E0 fully charged route toward `1.185185185%` without dense discovery?

## Answer

No for the static class. A synthetic Hamming/Steiner edge
`z_v = z_u + (v-u).x` expands into a static exact linear straight-line
program; a general shared linear DAG is more permissive and was already the
archived EXP-072B class. Moving that program to cold storage changes residency,
not its information source. It therefore fails E0 novelty and has no favorable
fully charged operation/traffic closure.

EXP-082A alone remains inconclusive for Steiner nodes because
`SMT >= MST/2` yields only `0.781183697%`. Published random-hypercube Steiner
and binary linear-circuit bounds add adverse distributional evidence, not a
finite pinned-checkpoint certificate.

The materially distinct class left open is a Query-Adaptive Cold Source whose
causal selector chooses a query-dependent subset of lossless cold information
and charges selection, probes, misses, execution, state, verification, and
fallback. Full audit:
[`E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`](../../../docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md).
