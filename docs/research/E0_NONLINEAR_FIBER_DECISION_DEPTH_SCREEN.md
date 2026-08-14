# E0 Nonlinear-Fiber Decision-Depth Screen

## Question

Can arbitrary nonlinear preprocessing beat the linear covering witnesses at
the first exact rank-one parity instances?

Fix one preprocessing value and let `C` be the corresponding set of binary
matrices. For every query `q = r tensor u`, a query algorithm may adaptively
read matrix coordinates and must return `<q,W>` for every `W in C`. Define

```text
d(C) = max over rank-one q of the minimum coordinate-decision-tree depth
       of <q,W> restricted to W in C.
```

This is a finite toy of one arbitrary nonlinear advice fiber. It grants free
address computation and a different optimal adaptive tree for every query.

## Complete `2 x 2` result

`scripts/prototype_rank_one_fiber_depth.py` enumerates all `65,535` nonempty
subsets of the sixteen binary `2 x 2` matrices. There are ten distinct
rank-one coefficient masks, including zero.

| fiber cardinality | minimum possible `d(C)` |
|---:|---:|
| 1 | 0 |
| 2--4 | 1 |
| 5--8 | 2 |
| 9--16 | 4 |

The minimax values are identical when all sixteen linear queries replace the
ten rank-one queries. At the power-of-two optima the printed witnesses are
ordinary affine parity-code fibers; exhaustive nonlinear selection gives no
smaller depth.

The sharpest small warning is also limited: every `2 x 2` fiber containing
more than half of all matrices has a rank-one parity that forces all four
coordinate probes. This is not extrapolated to large dimensions.

## Affine `2 x 3` control

The same program enumerates all `26,387` distinct affine fibers in six
ambient bits. For fiber sizes `1,2,4,8,16,32,64`, the exact minimax depths are
respectively

```text
0, 1, 1, 2, 2, 3, 6.
```

This is the expected linear covering behavior. It supplies no hidden
cross-row decoder.

## Decision and scope

```text
OMEGA-FIBERDT SMALL-NONLINEAR SHORTCUT: REJECTED
REASON: COMPLETE 2x2 SEARCH FINDS ONLY AFFINE-COVERING OPTIMA
HIGH-DIMENSIONAL NONLINEAR FIBER THEOREM: OPEN
GENERAL FINITE-WORD RANK-ONE PROBE GAP: OPEN
```

This experiment is not a Transformer lower bound. It does not cover large
nonlinear fibers, global 8 GiB advice, cross-matrix probes, cold encodings,
native BF16/FP32 semantics, or the registered 405B budget. Its role is only
to prevent a false claim that tiny nonlinear adaptivity already supplies a
new constructor.

The relevant general data-structure boundary remains Chakraborty, Kamma, and
Larsen, *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector
Multiplication*: <https://arxiv.org/abs/1711.04467>. Its numerical/parity
lower bounds do not close the globally advised target regime, while its fast
upper construction uses hereditary Boolean zero rectangles and does not
return exact numerical sums.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe scripts\prototype_rank_one_fiber_depth.py
```

No model, checkpoint, backend, download, hardware, or experiment-number action
is part of this E0 prototype.
