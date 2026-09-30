# Fixed-binade guarded native transition reference: preregistration

Recorded before the numerical/property checks on 2026-09-30. This work is a
bounded Phase A/B auxiliary construction, not an admitted core, model experiment,
HF equivalence claim, or target performance result. No root ledger is edited here.

## Intended claim and scope

For a binary round-to-nearest-ties-to-even format of precision p, a fixed normal
binade grid h, and one sign, an ordered sequence of exact dyadic addends has an
exact guarded representation by two integer offsets and one integer interval
per incoming accumulator parity. FMA addends are the exact products, without an
intermediate product rounding. Guards are constructed from the addends, not
provided by an oracle. Function composition may be regrouped; numeric additions
may not.

The independent reference rounds exact Fraction values to the declared binary
format. Reduced-width exhaustive checks test the construction. FP32/BF16 edge
cases check actual exponent ranges, normal/subnormal transitions, signed zero,
overflow and underflow. This declares an IEEE-like scalar FMA ABI; it does not
identify any actual ATen/GEMM/tensor-core reduction schedule with that ABI.

## Frozen guard requirements

- Positive normal binade [2^e, 2^(e+1)] and h=2^(e-p+1)
- Both parities, non-ties, half-ulp ties, negative addends and negative accumulators
- The lower endpoint's predecessor gap is h/2 except at the minimum normal
  exponent, where subnormals retain gap h
- The upper endpoint has successor gap 2h; it may be included only when finite
- At the maximum exponent, the overflow midpoint is excluded because RNE rounds
  that midpoint to infinity
- Negative-binade guards are reflected positive guards with endpoint closure
  reflected too
- Zero, signed zero, subnormal incoming accumulators, NaNs, infinities, and any
  guard escape use the original scalar instruction; they are not silently
  included in the two-offset normal-binade representation
- Guard construction/intersection, product formation/classification, shifts,
  carries, normalization, state decoding and every escaped instruction are paid

## Frozen checks

1. Exhaust every supported reduced-format normal accumulator, both signs and
   every binade, against a fixed exact dyadic addend grid spanning boundaries
2. Exhaust all three-addend sequences over a fixed half-ulp/adversarial alphabet,
   and compare accepted cases with independent sequential RNE
3. Compare left fold, right fold and balanced guarded-summary composition;
   validate domain equality and outputs, not only final values
4. Check first-escape execution against native sequential rounding, including
   crossings, cancellation, zero and overflow
5. Check concrete BF16 x BF16 -> FP32 fused-vs-separate underflow and overflow
   witnesses; test ties and adjacent boundary values using exact Fractions
6. Check sign reflection and maximum/minimum exponent boundary guards

Success means zero accepted-domain mismatches and exact escape execution on this
frozen population. The proof must state why it holds beyond that population.
A failed test stops the claim pending correction; retain initial failure evidence.

## Cost and promotion gate, frozen before tests

Let N be the number of scalar terms. The direct construction reads N coefficient
labels and N input positions and forms/classifies N exact products before or while
building summaries. It is O(N) with a larger finite constant than an FMA stream.
One stream-only first-escape implementation has at most N escapes and does not
avoid any coefficient. The compact map is not a compact query-dependent producer.
No model/backend experiment or core promotion is allowed without a different,
explicit non-enumerative producer with >=10x complete work/traffic elimination.
All target hardware, same-machine native-4B-Q4 latency, complete 8 GiB allocation,
KV, TTFT, workload quantiles, and full O1-O6 closure remain untested/open.
