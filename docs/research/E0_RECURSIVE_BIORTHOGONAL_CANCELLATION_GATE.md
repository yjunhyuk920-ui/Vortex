# E0 Recursive Biorthogonal Cancellation Gate

Status: **STRUCTURALLY VALID / PRIOR FIXED-LINEAR FRONTIER REJECTED**

This is an E0 theorem and exact finite calculation. It performs no model or
hardware run. Structurally valid conditions were established. Large-model
performance remains unverified.

## 1. Question

The one-shot biorthogonal Gate finds a decomposition with forced XOR
cancellation, but then stops. Its positive internal-edge bound actually
identifies a stronger reusable object: an adjacent pair of admissible
rank-one terms whose fixed representatives share an atom coordinate.

The question is whether that pair can be removed and the theorem applied to
the residual matrix without changing the dictionary contract.

## 2. Pair peeling

For a fixed rank-`r` matrix `M`, let `q,q'` be adjacent vertices in its
anti-flag graph. They extend to a minimal biorthogonal decomposition and

```text
rank(M-q-q') = r-2.
```

If their representatives share one coordinate, then

```text
wt(f(q) XOR f(q')) <= 2t-2.
```

The residual matrix uses the same fixed dictionary and the same radius-`t`
representative guarantee. If `Delta_r` denotes forced incidence cancellation
for every rank-`r` matrix, induction therefore gives

```text
Delta_r >= 2 + Delta_(r-2).
```

When no shared-coordinate pair is forced, an arbitrary adjacent pair still
gives `Delta_r >= Delta_(r-2)`. Odd rank can remove one arbitrary term and
gives `Delta_r >= Delta_(r-1)`.

## 3. Why the shared pair exists

For atom coordinate `c`, let `A_c` be the anti-flag vertices whose
representative contains `c`. The previous spectral calculation proves an
exact lower bound on

```text
sum_c e(A_c).
```

Whenever that lower bound is positive, at least one `A_c` contains an edge.
Its endpoints are an admissible adjacent pair sharing coordinate `c`. No
chromatic-number conjecture or sampled graph search is used.

At every even rank the executable Gate takes the stronger of:

```text
the one-shot overlap-to-cancellation knapsack,
the recursively peeled adjacent pair plus the rank-(r-2) bound.
```

It then uses

```text
R_r    = r*t - Delta_r,
R_<=r = max_(0<=j<=r) R_j
```

before comparing the exact Hamming-ball and determinantal-set sizes.

## 4. Decisive witnesses

For the prior `31 x 43`, `S=1,559`, `t=15` frontier, positive spectral edge
bounds begin at rank 18. Recursive peeling gives

```text
Delta_18 = 2
Delta_20 = 4
Delta_22 = 6
```

At rank 22 this reduces the radius from 330 to 324. The exact
`Ball(1559,324) / RankLeq22(31,43)` ratio is

```text
0.4293893830152846...
```

so the former frontier is impossible. The same Gate rejects `32 x 42` and
the registered capacity-order scan now closes its first 856 rectangles.

The first unclosed fixed-linear case becomes

```text
31 x 42, S=1,523, t=15,
best recursive ratio = 38.10086306517199... at rank 24.
```

This is not a construction.

## 5. Claim boundary

The theorem covers a fixed binary linear atom map and any nonlinear locator
that returns a short XOR representative in that map. It does not cover
adaptive word-valued probes, nonlinear output decoding, native BF16/FP32
semantics, a shared 32-query physical union, or a complete Vortex runtime.

The result closes a proof gap; it does not supply the target architecture.
Do not search nearby fixed-linear rectangles without a new structural
equation. The active construction path remains a genuinely adaptive
finite-word decoder or a lower bound that covers it.

## 6. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_recursive_biorthogonal_cancellation_gate.py `
  --output-dir results\e0_recursive_biorthogonal_cancellation_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_recursive_biorthogonal_cancellation_gate -v
```
