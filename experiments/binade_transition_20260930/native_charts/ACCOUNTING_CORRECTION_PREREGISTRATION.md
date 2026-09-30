# Internal inverse accounting correction: preregistration

Recorded before correction validation 03, 2026-09-30. Independent audit identified
that NativeCharts.inverse counts intersection inverses, but the sealed
PeriodicMap.then implementation performs two more inverse calls whenever its grid
grows. Those internal calls and their level quotient terms were omitted from the
previous inverse counters. This is an accounting error; no output mismatch or
native semantic change is alleged.

Preserve validations 01/02 unchanged and retain the source/report/manifest used for
validation 02 under results/source_02. Do not edit the sealed parent crossing.py.

Correction: a local NativeCharts map-update wrapper will charge exactly the two
internal inverse calls on h>peak_grid and their 2*len(levels) quotient terms. Keep
separate intersection/growth subtotals and total counters; also identify internal
grid-round/evaluation work. The same PeriodicMap.then remains the executed map
operation, with identical operands and output.

Repeat the unchanged positive validation population. Add a validator-only dynamic
spy on PeriodicMap.inverse and compare its observed calls/level terms with the
constructor counters on EVERY appended step, including BF16 wrappers. This spy
must restore all original methods afterward. It is an independent accounting
cross-check, not part of the constructor or output algorithm.

Success requires unchanged zero-mismatch output results and exact equality of
instrumentation counts to observed calls. Preserve any failed first attempt.
No baseline/model/backend test or performance promotion is authorized. Every
coefficient/product remains paid and CORE_ADMISSION remains false.
