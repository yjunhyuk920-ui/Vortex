# Validation matrix — theory-closure/native coding

2026-09-05. [Detailed scope and proofs](docs/research/THEORY_CLOSURE_NATIVE_CODING.md).

| Item | Status and exact meaning |
|---|---|
| RN32 nonadditivity/permutation witness | DERIVED and reproduced, not all GPU reduction topologies |
| Finite-state continuation separation | Written proof in declared gradual-underflow addition API |
| Registered independent checks | 752 pairs, 3,000 libm.fmaf-versus-integer additions, zero mismatches |
| BF16 store and invalid input checks | Passed within the same reference scope |
| Input/result format | Seed/edge list plus recorded uint16 output pairs; full witnesses regenerated and hashed |
| Full O1-O6 theory | NOT ESTABLISHED; no cheap universal generator |
| Public checkpoint / full KV/RNG backend | NOT TESTED |
| 405B / total 8 GiB / GPU / baseline latency / TTFT | NOT TESTED |
| Full repository tests / GitHub Actions | NOT RUN |

[Summary](results/theory_closure/summary.json), [raw observations](results/theory_closure/observations.json), [replay and environment](results/theory_closure/validation.json).
The mathematical proof is not inferred from finite test counts. Persistence does not promote scientific status.
[Previous matrix preserved unchanged](docs/research/history/pre_theory_closure_20260905/VALIDATION_MATRIX.md).
