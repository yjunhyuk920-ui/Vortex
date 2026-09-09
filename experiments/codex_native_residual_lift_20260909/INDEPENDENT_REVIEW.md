# Independent review: two adaptive native BF16 count queries

## Verdict

**VALID IN THE DECLARED ABI; no arithmetic counterexample found.** For every binary row of `B`, the two calls recover the exact integer count and therefore its parity. The proof gives the stronger bounds

`|k-k0| <= 64` and `|BF16(alpha1*(k-k0))/alpha1 - (k-k0)| <= 1/4`,

which imply the preregistered `<=65` and `<=65/256<1/2` bounds. This conclusion requires separate FP32 products followed only by FP32 additions/FMA, an RNE BF16 output store, and an exact-rational integer-rounding routine. It does not apply to reduced-precision partial sums or an ordinary floating-point division used in place of that routine.

## Proof

Fix a row and write its true count as

`k = sum_l B[i,l] v[l]`, so `0 <= k <= n`.

A positive normal BF16 scale with unbiased exponent `e` has the exact form

`alpha = M * 2^(e-7)`, where `128 <= M <= 255` is an integer.

All entries used here are exact BF16 values: `0`, `1`, `-2^j`, and `alpha` or `+0` in an input lane. The largest bias is `2^(b-1) <= 16384` because `b=n.bit_length()=ceil(log2(n+1))`.

### Native FP32 arithmetic is exact before each BF16 store

In the first query, every nonzero product is exactly `alpha`. Every partial sum is `A*alpha` with `0 <= A <= n`. Its integer coefficient obeys

`A*M <= 16384*255 < 2^22`.

In the second query, the exact dot product is

`alpha1 * (k - sum_j bit_j(k0) 2^j) = alpha1*(k-k0)`.

An arbitrary partial tree node can contain any subset of the positive unit terms and negative bias terms. Its coefficient therefore satisfies

`|A| <= k+k0 <= 2n <= 32768`, hence `|A|M < 2^23`.

FP32 has 24 bits of significand precision. Thus each exact product and each possible exact partial sum is representable in FP32. Induction over any binary sum tree proves that every addition is exact. The same invariant proves exactness for any serial FP32 FMA accumulation order, because each fused product-plus-accumulator result is representable.

There is no exponent failure. A nonzero partial has magnitude at least `alpha`, whose exponent is at least `-100`, and the largest partial is below `2^116`, whose exponent is at most `115`. Both lie strictly inside the normal finite FP32 range. Products by the largest bias also stay in this range. Cancellation can produce zero but no nonzero value smaller than `alpha`, since every partial remains an integer multiple of `alpha`.

### First call leaves a small integer residual

Before the first BF16 store the row value is exactly `z0=k*alpha0`. If `k=0`, the decoded value is zero. Otherwise let `E=floor(log2(z0))`. RNE BF16 rounding has absolute error at most `2^(E-8)`. Since `k<=2^14`, `alpha0<2^(e+1)`, and `alpha0>=2^e`, we have `E<=e+14`, so

`|y0/alpha0-k| <= 2^(E-8)/alpha0 <= 64`.

Rounding this exact rational to the nearest integer cannot move it outside the integer interval `[k-64,k+64]`. Projection onto `[0,n]` cannot increase distance from `k`, because `k` itself is in that interval. Therefore

`|k-k0| <= 64`.

This proves the preregistered `<=65` residual bound with one count to spare.

### Second call recovers the residual exactly

Let `r=k-k0`, so `|r|<=64`. Before the second store the native result is exactly `r*alpha1`. Write `alpha1=M*2^(e-7)`. The integer coefficient to be stored is `|r|M <= 64*255 = 16320 < 2^14`. Rounding an integer below `2^14` to eight significant bits changes it by at most `32`. Consequently

`|y1/alpha1-r| <= 32/M <= 32/128 = 1/4 < 1/2`.

RNE of the exact ratio therefore returns `r`, including for negative residuals. The final result is `k0+r=k`, and `k mod 2` is the exact parity.

The preregistered weaker estimate also follows: `1/4 = 64/256 <= 65/256 < 1/2`.

### Exact-rational RNE is finite and signed-zero safe

The decoder need not perform floating division. Decode each nonzero BF16 value as a sign, an integer significand, and a power of two. After moving the exponent difference to the numerator or denominator, compute integer quotient and remainder. Compare twice the remainder with the denominator; on equality choose the even integer. Reapply the sign. This terminates on bounded-size integers and implements RNE of the exact ratio.

Map both `+0` and `-0` to rational zero before this procedure. Negative bias entries multiplied by zero can create `-0`, and an arbitrary zero-only tree may preserve that sign, but the sign then has no effect on `k0`, the residual, the clamp, or parity. An implementation that rejects `-0` or treats it as a negative count would fall outside the proved decoder.

## Scope and paid costs

The matrix is fixed before either output is observed. The second input is allowed to depend on `y0`; it contains `v` again plus the row-private binary digits of `k0`. Neither input construction uses the unknown true count `k`.

This is an O3 arithmetic reduction from exact binary counts/parity to two native queries. It is not a fast native decoder. It performs two complete augmented matrix-vector evaluations and reads the source matrix twice. The conceptual dense matrix has shape `m x (n+m*b)`, so an ordinary dense representation pays `m(n+m*b)` BF16 entries and each call processes that full shape, including the cross-row zeros. A structured implementation can store the original `mn` binary entries plus `mb` nonzero biases and perform the bias correction directly, but that is a specialized representation whose indexing, arithmetic, and storage must be charged; it still pays two source-scale `Bv` evaluations. The adaptive control also pays decoding `m` first outputs, clamping them, forming `mb` bias bits, constructing the second input, and decoding `m` second outputs.

No `>=10x` work or traffic reduction follows. The lemma does not construct arbitrary legal Hugging Face hidden states that realize these query vectors, preserve RNG/KV or every continuation, implement an arbitrary native kernel, or establish behavior on target hardware. It does not permit globally mixed nonlinear advice to be split independently per matrix. Prior catalog, free-witness, full-scan, and dense-delta families remain excluded as fast cores.

Under `CONSTRUCTIVE_THEORY_CONTRACT.md`, this closes only the narrow finite-word count/parity arithmetic step. It does not close the full O3 causal state contract, O1 construction, O4 total resource bounds, or O5 405B hardware limits and latency. The preregistered ledger therefore remains `O1-O5 OPEN / O6 PARTIAL`, and this lemma does not turn the rejected fast-core claim true. The fixed mission remains arbitrary unmodified public HF 405B-class checkpoints, total GPU peak allocation at most 8 GiB, the native-4B-Q4 p50/p95 and TTFT requirements, and full original-output/successor-state preservation. The user has no 405B hardware, so no real-405B hardware claim is available. Overall theory and hardware status must remain `NOT_ESTABLISHED` and `NOT_TESTED` unless supported by separate evidence.
