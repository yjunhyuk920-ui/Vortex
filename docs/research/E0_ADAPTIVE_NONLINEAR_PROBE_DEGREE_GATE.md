# E0 Adaptive Nonlinear Probe-Degree Gate

Status: **STRUCTURALLY VALID / FIRST HONEST NONLINEAR-WORD GAP ISOLATED**

This is an E0 theorem and exact finite calculation. It extends a prior
fixed-linear capacity inequality to arbitrary nonlinear block-local encodings.
It does not construct a runtime and does not prove a global cell-probe lower
bound.

## 1. Elementary idea

Think of every stored cell as a box whose content may be **any** function of
the checkpoint. The decoder may open a box, look at its value, and use that
value to choose the next box.

A decoder opening at most `t` boxes is a depth-`t` decision tree. Every leaf
mentions at most `t` box coordinates, so the exact answer function can be
written as a polynomial whose monomials mention at most `t` boxes. This does
not assume that the stored contents are linear.

For a rank-one mask `q`, write its parity character as

```text
chi_q(W) = (-1)^<q,W>.
```

If a rank-`r` matrix is `Q=q_1+...+q_r`, then

```text
chi_Q(W) = product(i=1..r) chi_q_i(W).                   (1)
```

Multiplying `r` depth-`t` answer polynomials gives box-support degree at most
`r*t`. Distinct matrix characters are linearly independent. Therefore the
space of all degree-`r*t` box polynomials must be large enough to contain
every rank-at-most-`r` matrix character.

## 2. Exact theorem

Let the encoding have `S` cells, each with alphabet size `A`. The vector space
of functions whose monomials mention at most `d` distinct cells has dimension

```text
V_A(S,d) = sum(j=0..d) C(S,j)(A-1)^j.                  (2)
```

Exact recovery of every rank-one parity with at most `t` adaptive probes
forces, for every matrix rank `r`,

```text
number of a x b matrices of rank <= r
    <= V_A(S,min(S,r*t)).                               (3)
```

All matrix units are rank one. Hence exactness already forces the arbitrary
encoding to be injective; no linear or systematic premise is hidden here.

For one-bit cells, (2) is the ordinary Hamming-ball size. Thus the old
determinantal capacity inequality remains valid even when the stored bits,
address path, and final Boolean logic are all nonlinear.

## 3. Exact bit-cell consequences

The smallest visible capacity gap is:

```text
source                 2 x 3 binary matrix
stored bits            7
linear sparse minimum  3 probes
general nonlinear degree minimum not rejected  2 probes
```

The last line is not a construction. It is the first exact place where a
genuinely nonlinear adaptive encoding could improve on every linear
dictionary, so it is the cheapest synthesis target.

The first implementable subclass has also been closed exactly. Store all six
source bits systematically and let the seventh bit be an arbitrary Boolean
function of the complete matrix. Symbolic enumeration of every depth-two
adaptive decoder turns each decoder into pointwise constraints on the 64-bit
truth table of that advice function. Every one of the six hard queries has 14
possible constraint cubes, but no mutually consistent choice exists across
all rank-one queries. Thus the systematic-plus-one-nonlinear-bit seed is
impossible; only a fully non-systematic seven-bit encoding remains open.

An exact SMT model for that fully non-systematic case was run for 600 seconds.
It returned `unknown` due to timeout. The timeout is recorded as inconclusive,
not as evidence of impossibility or existence.

At the registered fixed-linear frontier:

```text
source                 31 x 42 = 1,302 bits
stored bits            1,523
minimum arbitrary nonlinear bit probes not rejected = 15
14 probes first fail at matrix rank 16
```

Thus merely replacing each linear atom by a nonlinear bit cannot move that
frontier below fifteen probes.

## 4. Arbitrary nonlinear 64-bit words

For 64-bit cells, set `A=2^64`. Give the candidate the whole proportional
8 GiB advice share, pad every partial word to 64 fully useful bits, and divide
traffic by four favorable Q4 lanes.

For `31 x 42`, equation (3) gives:

```text
padded words                                24
one adaptive word probe                    rejected
first probe count not degree-rejected       2
favorable traffic                          16/651
                                            = 2.4577...%
latency-derived target                     8/675
minimum/target                             450/217
                                            = 2.0737...x
```

Unlike the packed-linear Gate, the nonlinear calculation cannot reject all
larger rectangles. Across all 8,256 shapes with sides at most 128:

```text
capacity-feasible at or below target       4,257 shapes
smallest-area survivor                     25 x 108
source bits                                2,700
word probes                                2
traffic                                    8/675 exactly

best traffic-capacity point                128 x 128
word probes                                4
traffic                                    1/256
```

These numbers prove only that low-degree capacity no longer rules the shapes
out. They supply no encoder, no address rule, and no decoder.

## 5. What changed and what remains

The new theorem closes the objection that nonlinear stored bits automatically
escape determinantal rank amplification. They do not. Adaptivity is already
included in the decision-tree polynomial.

The precise constructive target is now:

```text
Encode every 25 x 108 binary matrix into 50 padded 64-bit words,
then answer every rank-one parity with two adaptive word reads.
```

Passing raw degree capacity does not make this object likely. A valid object
must include an explicit finite encoder, both address functions, the exact
output function, succinct metadata, and a native numerical lift. The first
smaller seed is the arbitrary seven-bit/two-probe `2 x 3` problem.

Global cross-matrix encodings, a shared physical union for 32 queries, and
Q4/BF16/FP32 rounding remain outside this block-local theorem.

## 6. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_adaptive_nonlinear_probe_degree_gate.py `
  --output-dir results\e0_adaptive_nonlinear_probe_degree_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_adaptive_nonlinear_probe_degree_gate -v
```
