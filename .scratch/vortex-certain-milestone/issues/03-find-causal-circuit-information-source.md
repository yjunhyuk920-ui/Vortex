# Find a Causal Circuit Information Source

Type: research
Status: resolved
Blocked by: none

## Question

What information available from the current committed prefix and unchanged
checkpoint can select a query-dependent subset of exact cold pages or
operations without executing an equivalent dense pass, reading target future
tokens, training an adapter, or hiding a full checkpoint scan?

## Answer

[The Causal Residual Atlas source audit](../../../docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md)
identifies exact committed-prefix `(x, W x)` pairs as a concrete coded causal
source. They form `Q` and `Z=WQ`; current out-of-span work remains a bounded,
progressively revealed cold residual with exact fallback. A favorable rank-16
405B screen has a nonempty logical budget window but needs
`99.899840530%` certificate coverage, so only the cheapest pinned real-weight
Gate is unblocked. No Surviving Candidate or positive milestone exists yet.
