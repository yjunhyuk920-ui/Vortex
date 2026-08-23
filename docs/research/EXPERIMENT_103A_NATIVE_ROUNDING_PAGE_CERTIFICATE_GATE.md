# EXP-103A — Hierarchical Native-Rounding Page Certificate Gate

## Question

Can an unchanged BF16 dense checkpoint avoid reading and multiplying almost all
weight pages because the native finite-precision accumulator proves every
product in a page is an exact no-op?

## Three materially different principles screened

1. **Hierarchical native-rounding page certificates — selected.** Reverse the
   assumption that every dense product must be evaluated. A one-byte maximum
   BF16 exponent summary is stored per output-row/input-page. At runtime the
   exact FP32 rounding cell and activation magnitude certify a page only when
   every separate multiply/add term is strictly unable to change the current
   accumulator.
2. **Prefix-transition monoid binary lifting — rejected before implementation.**
   It composes token transitions instead of running them sequentially, but an
   exact resident transition table is the already-closed self-contained
   artifact/table family and supplies no charged sub-dense decoder.
3. **Deferred mismatch state closure — auxiliary only.** It moves correction
   state materialization to the next sweep. It can remove one repair call but
   cannot increase the accepted prefix or reduce the principal checkpoint
   sweep.

## Exact ABI

The reference operation is fixed left-to-right:

```text
p_j = RNE_FP32(BF16(w_j) * BF16(x_j))
s_j = RNE_FP32(s_{j-1} + p_j)
```

For accumulator `s`, let `I(s)` be the strict interior of its IEEE binary32
round-to-nearest-even cell. A page is skipped only if the exponent metadata and
runtime activation maximum prove every rounded product lies inside
`I(s)-s`. The accumulator therefore remains exactly unchanged for every term.

The implementation charges the metadata bytes. It does not grant free weight
reads, future activations, a perfect selector, compression, or a changed
reduction order.

## Traffic Gate

For page width `Q`, one metadata byte per `2Q` BF16 weight bytes costs

```text
metadata_fraction = 1 / (2Q)
```

With the registered p50 checkpoint traffic fraction
`0.011851851851851851`, the remaining full-page read allowance is

```text
q_max = target_fraction - 1/(2Q).
```

Across `Q=64..1024`, the mechanism must skip more than 98% of all pages; for
practical `Q=128..512`, it must skip about 99%.

## Decisive finite-word control

A legal dense BF16 row and activation consisting entirely of ones is executed
in the declared ABI. Every `1*1` addition changes the FP32 accumulator through
16,384 terms, so every page is read at every registered page size. This is not
an oracle or an asymptotic projection; it is an executable finite-word input.

The same implementation also runs 200,000 randomized soundness cases, a tiny-
product positive control, and 64 BF16 Gaussian dense rows.

## Promotion and rejection

Promotion required zero exactness mismatches and a universal route to the
registered traffic fraction. The all-ones finite-word row forces 100% page
reads, so the universal byte Gate fails before kernel or checkpoint work.

```text
REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
```

The exact certificate may remain an auxiliary within a future block kernel,
but it cannot be the arbitrary-checkpoint core.
