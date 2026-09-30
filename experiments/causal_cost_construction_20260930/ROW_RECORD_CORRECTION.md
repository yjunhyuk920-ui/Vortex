# Row-record clarification — 2026-09-30 15:35 UTC

The eight-byte runtime record does not store every exact row norm. Eligibility
is decided before narrowing the field, with fully paid bounded multiword
constructor scratch. For N<=16384, two five-uint64-limb arrays suffice for the
exact norm in units2^-133 and its integer rescaling, plus scalar decoding/shift
state. Their scans, shifts, comparisons and writes are explicitly charged.
Larger widths require a re-derived/enlarged limb bound.

Only eligible finite nonzero rows store exact uint32 S_i and int16 v_i.
Ineligible rows store a flag and unused zero sentinels, never a wrapped norm,
and always execute original fallback. All-zero and nonfinite rows have separate
flags and unused zero fields. Flag dispatch precedes numeric use. Zero inputs
likewise have no valuation; their special path does not evaluate one.

The previous NATIVE_FIELD_WRAPPER.md is preserved byte-for-byte at
history/pre_row_record_correction/NATIVE_FIELD_WRAPPER.md, SHA256
106c3141af1d9cc9161c71e6de070fe9aee364cca6e6f595c88b278dc1ec8c03.
The original pre-architecture-correction snapshot also remains untouched.

This clarifies representation and charges constructor work; the representability,
centered-decoding proof and coverage failure are unchanged. It supplies no cheap
source, coverage guarantee or core admission. No runtime/model/GPU experiment,
root-ledger edit, commit or publication occurred.

ROW_RECORD_CHECKS.json records this revision's wrapper hash. The earlier
CORRECTION_CHECKS.json is preserved as evidence of the preceding architecture-
neutrality revision; its wrapper hash is historical, not this revision's hash.
