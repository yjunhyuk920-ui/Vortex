# EXP-100A latest result — Explicit Rectangular FMM Gate

## Identity

- source commit: `0348e87fa385630a84043727ddb9a34c032df4ac`
- raw result: `/home/runner/work/Vortex/Vortex/results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json`
- AlphaTensor catalog commit: `1949163da3bef7e3eb268a3ac015fd1c2dbfc767`
- AlphaTensor catalog Git blob: `5ff45960f86da6237f105f78aaa82d29cb18c30e`
- deterministic core: `e72ec22a41216b3b3a91c3989c5327ee7492bfb356bf4d799c2af1b97cd2ab1e`

## Authoritative decision

`INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE`

## Integrity and catalog

- catalog keys inspected: `93`
- eligible exact integral small-coefficient factorizations: `59`
- retained tensor orientations: `128`
- deterministic integer controls: `132`
- control mismatches: `0`
- integrity failures: `['empty_direct_shape_search']`

## Best explicit staged-transform plan

- block length: `8192`
- arithmetic ratio: `38.251649686367%`
- cold-byte fraction/token: `1.184508591014%`
- favorable FMM workspace: `6.121550854 GiB`
- compiled static representation: `109460.330654640 GiB`
- first 10x arithmetic Gate: `False`
- final p50-equivalent arithmetic Gate: `False`
- p50 cold-byte Gate: `True`
- favorable workspace screen: `True`
- all direct 10x joint-pass blocks: `[]`
- all direct final-p50 pass blocks: `[]`

## Best free-transform rank oracle

- block length: `4096`
- arithmetic ratio: `13.010262621991%`
- cold-byte fraction/token: `0.016276041667%`
- favorable workspace: `1.103682553 GiB`
- first 10x arithmetic Gate: `False`
- all oracle 10x joint-pass blocks: `[]`

The oracle grants every factor transform, packing operation, and transform scratch
for free. It is algebraic headroom, not an executor result.

## Next gate

### EXP-100A control repair

Repair only the recorded identity, tensor-reconstruction, registered-input, or deterministic-control failure. Do not interpret an invalid run scientifically and do not change the frozen search or thresholds while repairing infrastructure.

## Claim boundary

The run verifies the pinned public factorization catalog and derives a bounded
405B-shape resource ledger. A causal future block, finite-word transformed-weight
closure, sound native-repair selector, existing-ISA packed kernel, complete
successor state, TARGET-W execution, physical 8-GiB allocation, and same-machine
4B p50/p95 remain `NOT TESTED`.
