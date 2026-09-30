# Corrected internal inverse accounting

## Error and preserved evidence

Independent audit correctly found that the initial NativeCharts inverse counters
covered region-intersection calls only. When a map's grid grew, the sealed
PeriodicMap.then also executed two inverse calls, which bypassed those counters.
Output checks passed, but the reported total inverse counts were incomplete.

No original result was overwritten. Validations 01/02 remain intact. The source,
report and checksum manifest used for validation 02 are retained under
`results/source_02`; its constructor SHA256 is
`d26afa010bc289894c9cb6606a913af15e292a125ee973be82810c7705890dee`.
The independent audit's own receipt is retained under results when supplied by
the parent. The parent sealed crossing.py is unchanged.

## Correction

`NativeCharts.update_mapping` calls the same PeriodicMap.then with the same map,
grid and addend. It charges its fixed implementation's work:

- Grid growth: two inverse calls, two times the current level count in quotient
  terms, one internal map evaluation and one grid-round call
- No grid growth: one grid-round call per existing level

Intersection and growth inverse counts are separately recorded and summed into
inverse_calls/inverse_level_quotients. Native-word Format.round calls remain a
separate category from uniform-grid round_grid calls. No mathematical output,
state behavior, region, input domain, bound or acceptance threshold changes.

## Corrected native-stream totals

For the eleven registered FP32 streams (44 appended addends):

    intersection inverses: 47,515 calls; 105,036 quotient terms
    growing-grid internals: 4,766 calls; 11,050 quotient terms
    TOTAL: 52,281 calls; 116,086 quotient terms
    growing-grid updates/internal map evaluations: 2,383
    all internal grid-round calls: 38,818
    separate Format.round calls: 10,147

The revised totals exactly match the independent audit. All previous noninverse
counters and semantic validation populations compare unchanged to validation 02.

## Independent dynamic validation

Correction preregistration preceded validation 03. Its test-only spy wraps the
actual PeriodicMap.inverse method and observes calls/level counts directly.
On EVERY NativeCharts.append, actual deltas must equal the constructor's recorded
deltas; subtotal identities are also asserted. The original methods are restored
in a finally block. The production constructor does not use the spy.

Across the full frozen suite, the spy checked 3,050 append steps and observed
164,931 calls with 347,198 quotient terms. Accounting mismatches: zero.

The existing 342,000 exhaustive tiny-prefix comparisons, 3,520 selected FP32
comparisons, 192 BF16-wrapper comparisons, zero/sign/overflow witnesses and API
rejections still pass with zero output mismatches. The new log/result is
`validation_03.log` / `results/validation_03.json`.

This correction increases the charged cost. Every original product/read remains
necessary for this constructor. CORE_ADMISSION=false and no 10x, model/kernel,
GPU or latency claim changes. The corrected package is local evidence; publication
and remote verification remain the integrating parent's responsibility.
