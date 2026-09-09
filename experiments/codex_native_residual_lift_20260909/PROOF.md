# Exact counts from two adaptive BF16 native queries

This proves an auxiliary arithmetic reduction. It is not a fast native executor,
a legal-HF-state controllability theorem, or a model-wide O3 closure.

For every binary B of shape m by n, binary v of length n, 1<=n<=16384 and m>=1,
let b=n.bit_length()=ceil(log2(n+1)), D=n+mb. Construct fixed BF16 C with
C[:,:n]=B and C[i,n+ib+j]=-2^j; all other bias coefficients are zero.
Before either result is observed, this entire matrix is fixed.

The ABI is FP32 products and FP32 additions in any binary tree, or FP32 FMA
accumulation, followed by one RNE BF16 output store. No low-precision partial
sums are allowed. Define BF16 as sign/8-bit exponent/7-bit fraction and FP32
as sign/8-bit exponent/23-bit fraction, with round-to-nearest ties-to-even.
Each scale alpha is positive normal BF16 with unbiased exponent e in[-100,100].

Call0 uses x0=alpha0*(v,0). From actual output y0, use finite integer RNE division
to compute k0_i=clamp(round_even(y0_i/alpha0),0,n).
Call1 uses x1=alpha1*(v,bits(k0_0),...,bits(k0_(m-1))).
Return k_i=k0_i+round_even(y1_i/alpha1), with exact-ratio integer rounding.
True Bv is never used to form either query. In particular no rounding witness
or selector is supplied by an oracle other than the explicitly paid native calls.

## Exact intermediate invariant

Write alpha=M*2^(e-7), where128<=M<=255 is an integer. Each nonzero product is
alpha times an integer coefficient. In query0 every partial coefficient lies
between0 andn. In query1 any subset of positive unit terms and negative binary
bias terms has absolute coefficient at most k+k0<=32768.
Thus every product and possible partial has significand integer magnitude
at most255*32768<2^23, within FP32's24 significant bits.

Every nonzero partial has magnitude at least alpha>=2^-100; the largest is
below2^116. All products and partials are normal finite FP32. Induction on any
binary reduction tree therefore proves exact FP32 addition; serial FP32 FMA
has the same invariant. A reduced-precision accumulator is outside this ABI.

The exact pre-store results are alpha0*k and alpha1*(k-k0), where k=(Bv)_i.

## Residual recovery

For positive z0=k*alpha0, BF16 RNE relative error is at most1/256 in this normal
range. Hence |y0/alpha0-k|<=k/256<=64. The endpoints k+-64 are integers; nearest
integer rounding cannot escape their closed interval. Clamping to[0,n] cannot
increase distance from true k in that interval. Therefore |k-k0|<=64.
This is stronger than the preregistered conservative bound65.

For r=k-k0, the second normalized BF16 error is at most |r|/256<=1/4<1/2.
Thus exact-ratio RNE returns r, and k0+r=k. The parity follows by integer mod2.
For k=0 or r=0, either signed zero maps to integer zero. This decoder does not
claim that its integer output replaces either native output word or native state.

## Finite decoder and termination

Decode BF16 sign, significand and exponent. Move exponent difference into an
integer numerator or denominator, compute quotient and remainder, compare twice
the remainder with the denominator, choose the even quotient on equality, and
restore sign. Actual admitted ratios use small bounded integers; there is no
floating division whose rounding could silently change a tie. lift.py uses this
algorithm; exact Fraction arithmetic is an independent validation oracle only.

All loops are finite: mD matrix initialization, mn source copying, mb bias writes,
two fixed padded FP32 trees and O(mb+n+m) input/decode work. Allocation failure
is not covered by a resource-success theorem and is not a fast fallback.

## Direction and remaining obligations

The implication is: access to two native calls for this structured C enables
exact counts/parity. It does NOT imply a native decoder from a cheap GF2 oracle.
It still executes the dense body twice. BF16 output256 and257 can collide after
the first call; the second deliberately pays a new native effect to recover them.

This statement does not establish that arbitrary vectors x0/x1 occur as legal
hidden states of an unchanged public HF405B, or that a full token protocol has
the required RNG/KV/all-continuation correspondence. It cannot be composed into
such a theorem by assumption. Global nonlinear advice is not split per matrix.

The registered-shape embedding audit admits only9,342,985,041 independent source
bits (1.0876666104GiB). That information fits inside the global8GiB allowance.
Consequently this embedding cannot exclude a globally mixed representation that
retains the whole variable source there. No target lower-bound improvement is
claimed, and further growth of this arithmetic control is not the next core.
