# Review of proposed conservative fixed-array instruction bound

The proposed bound is defensible as a derived bound for an explicitly specified fixed-array 64-bit RAM rendering, not the Python implementation's observed cost:

    Q = n^2 + 4n
    U_build <= 64 [Q + 2rn + n(r+1) + r^2 + 10r + 10n + 64].

Assume 1<=n<=64, 0<=r<=n; fixed-size arrays for packed rows, Krylov columns, pivot vectors, labels and validity; one instruction per bounded-word load/store, arithmetic/bit operation, index computation, comparison and branch; and one instruction for 64-bit popcount. Source/serialized data service is separately charged, with original/source/transformed/workspace buffers retained in the memory inventory. Dynamic allocation, Python objects/dictionaries, process startup, filesystem and physical transport service are outside this RAM instruction term, not free.

A defensible debit schedule is:

- Q source-word validation and packing bodies
- rn Krylov-power row bodies and rn certification row bodies
- n(r+1) pivot reduction slot bodies
- r^2 literal certification decoder-column bodies
- 3r observer transport, 3r observer-identity checks, r decoder serialization, r power-call overhead, r certification outer overhead, and r+1 elimination outer overhead: at most 10r+1 bodies
- At most 10n bodies for fixed-array initialization and source row/array outer handling
- At most 63 additional fixed header/setup/validation/output bodies, absorbing the extra elimination body

Each body can be rendered in fewer than 64 of the specified instructions. Keep the pseudocode/operation-model alongside this debit schedule so the bound has an actual implementation target. Serialized basis-word byte writes can be emitted in a fixed eight-store body under this allowance. Assertions/comparisons must be counted; dynamic exception formatting need not be in successful legal-source execution.

Two required representation details:

1. Do not treat Python bit_length as free. During the descending pivot scan, remember the first set bit at an invalid pivot slot. Later lower-pivot eliminations cannot modify this higher bit, so the remembered slot is exactly the leading nonzero residual bit. This avoids a separate unpriced scan. A six-stage bounded bit-scan would also fit, but should be explicitly charged.
2. Form the r-bit mask as all-ones when r=64, otherwise (1ULL<<r)-1. Never execute an undefined 64-bit shift by 64. All basis-coordinate shifts otherwise use indices <=63.

For the maximum n=r=64, the bound is 1,417,216 modeled word instructions. This large conservative number is not a measurement, a cycle count, a baseline ratio or a target-latency claim. Initializer, online step, requested readbacks, logging and any later imported-state solve are separate schedule terms. Counts/bytes for external I/O do not by themselves establish its service-time upper bound.

This addendum reviews the proposed fixed-array schedule on 2026-10-01. Original replay checks concern verify.py SHA256 cb7bb422116b8e24487ace672a227b6e78709a63f98452f30d406e6d53add654 and native_reference.c SHA256 1219f1a953c9ac9374db95131f1e5c0e7e9741cc5a44f54691b2cd8d21120337. The independent script's import path points to the live original experiment, and its receipt records these input hashes. A portable historical replay must point the import at the retained file with that hash; it must not silently import a later revision and call it the same reviewed execution.
