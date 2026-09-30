# Native scalar chart construction: preregistration

Before implementation tests, 2026-09-30. This is a separately sealed continuation
of the prior guarded-binade and fixed-itinerary packages; their files and results
are not changed here.

## Claim and scope

Construct the complete map of a fixed finite sequence of scalar IEEE-style RNE
additions/FMA exact products without enumerating initial FP32 words. Initial
finite words, signed zeros and infinities are admitted; NaNs and nonfinite FMA
operands are explicitly excluded. Products are finite exact dyadics with a sign
bit for exact zero. No actual GEMM schedule, HF state or accelerator is claimed.

Initial charts are signed normal binades, signed subnormal intervals, signed-zero
singletons and infinity singletons. Intersect each chart's monotone PeriodicMap
with disjoint exact-sum regions via its explicit inverse. Separate signed-zero
rounding basins at +/-half a minimum subnormal, exact cancellation at zero, normal
binades, and inclusive overflow tails. Nonconstant charts must never contain an
untagged zero output. Constant charts use the declared scalar reference and are
charged. Every product and input coefficient remains read and paid.

Use an ordered region cursor to avoid all-charts times all-regions search.
Canonicalize genuinely constant intervals and merge only provably identical
adjacent charts. Record products/operand reads, chart visits, region advances and
intersections, inverses/quotient terms, map updates, scalar rounds, peak/final
charts and relevant bit widths. The sufficient construction remains O(BN^2)
bounded multiword actions; repeated updates are not model-once preparation.

## Frozen checks

- Tiny format p=4, exponent_bits=3: every non-NaN initial word, including both
  infinities and signed zeros, for every length-three sequence from
  {-17,-1,-1/32,-1/64,-0,+0,1/64,1/32,1,17}
- Compare against independently rounded nearest-neighbor scalar references
- Verify chart coverage/no overlaps, chart invariants and complete successor
  word after every prefix, not just the final result
- Deterministic FP32 boundary population: both zeros/infinities, extreme finite
  values, min/max subnormals, and immediate neighbors of signed powers of two at
  exponents {-126,-125,-75,-1,0,1,64,126,127}
- Fixed FP32 streams cover no-op zeros of both signs, exact cancellation, half-ulp
  ties, underflow products, overflow, huge positive/negative finite addends and
  repeated binade crossings
- A BF16 operand wrapper must form exact products without premature FP32 rounding
  and charge each input/weight access. Recheck the two fused/separate witnesses
- No FP32 enumeration, model run, GPU, Actions or target latency benchmark

Success requires zero mismatches. Preserve any initial failure before correction.
No >=10x/core promotion follows even if every check passes. The independent
heterogeneous product/summary producer remains missing.
