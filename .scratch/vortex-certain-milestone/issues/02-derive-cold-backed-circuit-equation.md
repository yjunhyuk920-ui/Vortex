# Derive the Query-Adaptive Cold-Backed Equation

Type: research
Status: resolved
Blocked by: none

## Question

For the Query-Adaptive Cold Source left open by the synthetic-intermediate
audit, what are the tightest favorable selection, operation, cold-traffic,
hot/intermediate-state, compile-amortization, verification, miss, and fallback
equations at the registered 405B shapes, and what minimum hit/coverage and page
granularity would be required to remain below `1.185185185%`?

## Answer

For operations and traffic independently, the complete favorable equation is
`R = g + kappa/N + rho*h + (1-rho)*(m+1)`. Even a zero-cost fast path requires
`98.814814815%` exact coverage at the p50 conservation Gate; the known
six-check verifier raises this to `99.081653739%`. The corresponding minimum
useful information amplification is `84.375x`, or `108.891389x` after that
verifier.

The p50 byte ceiling is `2.228263889 GiB/token`: with other costs free it
permits `584,126` 4-KiB, `36,507` 64-KiB, or `2,281` 1-MiB payload pages. Raw
Q4 page omission cannot determine unread dense contributions; a surviving
source must use a causal checkpoint-derived code or certificate and charge its
selector, decode, state, build, verification, misses, and fallback.

Full derivation and tested calculator:
[`E0_QUERY_ADAPTIVE_COLD_EQUATION.md`](../../../docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md).
