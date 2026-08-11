# E0 Segre Ruling Preimage-Sphere Gate

## Verdict

```text
REJECT THE PRIOR 13 x 89 SPARSE-COVER FRONTIER EXACTLY
PROMOTE 18 x 36 AS THE NEXT FIXED-LINEAR CONSTRUCTION TARGET
REQUIRE, BUT DO NOT CLAIM, A [146,110] BINARY RADIUS-7 COVERING CODE
REQUIRE ALL SEGRE RULINGS TO WORK THROUGH ONE SHARED KERNEL
KEEP ADAPTIVE FINITE-WORD DECODERS OUTSIDE THIS THEOREM
NO SURVIVING CANDIDATE
```

No checkpoint, model, download, backend, server, or hardware action occurred.
This is an exact finite-dimensional obstruction for a fixed binary linear atom
map. It is neither a native numerical implementation nor a general adaptive
cell-probe lower bound.

## Elementary explanation

Imagine that a code has 1,353 switches. It claims that every answer in a
`13 x 89` rank-one puzzle can be made by turning on at most 13 switches.
Instead of checking every answer, freeze the left factor. This leaves one
straight family containing exactly `2^89` answers.

All switch patterns that land in this family form a 285-dimensional binary
space. A basic information-set argument says that such a space can contain at
most

```text
sum(i=0..13,C(285,i)) = 10,451,278,437,645,598,672,024
```

patterns using at most 13 switches. But the family needs

```text
2^89 = 618,970,019,642,690,137,449,562,112
```

different patterns. The available fraction is only
`0.0000168849509765...`. Thus `13 x 89` is impossible without searching for
atoms. The next combined-Gate survivor is `18 x 36`; that is a target to
construct, not evidence that a construction exists.

## 1. Preimage theorem

Let

```text
phi : F_2^S -> F_2^(a x b)
```

map a selected atom subset to its XOR. Covering all rank-one masks forces
`phi` to be surjective because the matrix units are rank one. Hence

```text
D = ab,                 dim ker(phi) = r = S-D.          (1)
```

Choose factor subspaces `A <= F_2^a` and `B <= F_2^b` with dimensions `x`
and `y`. Their tensor product has dimension `xy`, so

```text
P = phi^(-1)(A tensor B),
dim P = r+xy.                                             (2)
```

The distinct rank-at-most-one masks inside this tensor subspace number

```text
Q_xy = 1+(2^x-1)(2^y-1).                                (3)
```

If the dictionary covers every such mask at selection weight at most `t`,
then `P` must contain at least `Q_xy` distinct ambient words of weight at most
`t`.

## 2. Information-set sphere bound

For any `k`-dimensional binary subspace `P <= F_2^S`, choose `k` coordinate
positions on which projection is one-to-one. The projection is a bijection
from `P` to `F_2^k`, and deleting coordinates never increases Hamming weight.
Consequently

```text
|P intersect Ball(S,t)| <= sum(i=0..t,C(k,i)).            (4)
```

Combining (2)--(4) gives the exact necessary condition

```text
sum(i=0..t,C(S-ab+xy,i)) >= 1+(2^x-1)(2^y-1)             (5)
```

for every `1 <= x <= a`, `1 <= y <= b`. The special case `x=1,y=b`
is one complete Segre ruling: fix one nonzero left factor and vary every right
factor.

This proof does not assume random atoms, systematic storage, distinct atoms,
or a particular decoder for locating the subset.

## 3. Exact rejection of `13 x 89`

The prior first Fourier survivor had

```text
a=13, b=89, D=1,157, S=1,353, t=13, r=196.
```

For the large ruling `x=1,y=89`, its preimage dimension is

```text
k = r+b = 285.
```

Equation (5) fails by a factor of about 59,224:

```text
Ball(285,13) = 10,451,278,437,645,598,672,024
required       = 618,970,019,642,690,137,449,562,112
ratio          = 0.000016884950976589718...
```

Therefore no fixed dictionary with these parameters can cover all binary
rank-one masks. This strictly advances the Fourier Gate: its slack prevented a
pointwise character contradiction, while one internal ruling already lacks
enough low-weight preimages.

## 4. First survivor of the combined scan

Rectangles with sides `1..128` were sorted by the exact global ratio
`|Ball(S,t)|/|Segre(a,b)|`. Each was tested first by all restricted tensor
conditions (5), then by the prior Fourier second moment. The first ten are
rejected; the eleventh and first unclosed shape is

```text
a=18, b=36, D=648, S=758, t=7, r=110,
|Ball(758,7)| / |Segre(18,36)| = 1.5546241552963183...
```

None of its `18*36` restricted factor-dimension checks violates (5). The
minimum ratio occurs at the full `18 x 36` space, where it equals the global
ratio above. The Fourier second moment also does not reject it. Neither result
constructs the required atoms.

## 5. Ordinary projected code required by one ruling

For the large 36-dimensional ruling, `P=phi^(-1)(L)` has dimension

```text
n = r+b = 146.
```

Choose an information-set projection of `P` to 146 coordinates. The shared
kernel maps to a binary linear code with parameters

```text
[n,k] = [146,110],       codimension 36,
covering radius <= 7.                                    (6)
```

Indeed, every coset corresponds to one ruling point, and its promised
weight-seven preimage projects to a coset representative of no larger weight.
The elementary sphere condition does not reject (6):

```text
Ball(146,7) = 255,108,299,740
2^36        = 68,719,476,736
density     = 3.7123143518692814...
```

This is a useful construction Gate, not a solution. One ordinary
`[146,110]` radius-seven covering code is only necessary. The same 110-
dimensional kernel must induce suitable covers for every one of the
`2^18-1` left-factor rulings while also aligning their cosets with the
bilinear `u tensor v` labels. Independent covering codes cannot simply be
chosen for each ruling.

## 6. Claim boundary and next task

The theorem covers fixed binary linear atoms and selection weight. It does
not cover data-dependent adaptive addresses, arbitrary word-valued decoders,
native Q4/BF16/FP32 rounding, a physical 32-query union, KV/state handling, or
the complete runtime cost equation.

The next cheapest constructive task is now precise:

1. build a parity-check description of a `[146,110]` binary code with
   covering radius at most seven, or prove the exact parameters impossible;
2. if it exists, test whether one common 110-dimensional kernel can satisfy
   all `18 x 36` Segre rulings rather than one ruling in isolation;
3. only after that, supply a succinct subset decoder, shared 32-query physical
   union, and native numerical lift.

Until the simultaneous construction exists, `18 x 36` is an unclosed
mathematical frontier and `NO_SURVIVING_CANDIDATE` remains correct.

## 7. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_segre_ruling_preimage_sphere_gate.py `
  --output-dir results\e0_segre_ruling_preimage_sphere_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_segre_ruling_preimage_sphere_gate -v
```
