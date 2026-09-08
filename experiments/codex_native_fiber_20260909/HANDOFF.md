# Next exact obligation

Read [REPORT](REPORT.md), [preregistration](PREREGISTRATION.md),
[obligations](obligations.json) and [validation](VALIDATION.json).

Do not infer native injectivity from algebraic full rank. Independent bijective
word encodings preserve fibers, and the dense I+11^T family has a native fiber
of size at least 256^(n-1) in the declared FP32 ABI. The later proof-only binary
corollary directly separates GF2 full rank from native injectivity as well.

This closes only a local pure-copy native lift premise. It does not eliminate
global cross-wire/side-state encodings or nonbijective exact continuation
representations. The side-state bit count is only for an optional injective
lift; it is not a native KV lower bound and cannot be summed per matrix.

Next core: explicitly construct the paid native dense query/state map, allowing
native collisions and correctly retained side state, without hiding full dense
execution in encoding/decoding. Existing source reconstruction, free witness,
full scan, catalog, row replay and arbitrary sparse-delta premises stay excluded.
Do not repeat the collision-box enumeration as primary progress.

The shared original checkout belongs to another active writer; inspect actual
remote/head/status and preserve their current work. This package was built in
an independent checkout and integrated only by normal branch updates.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false;
O1-O5 OPEN; O6 PARTIAL. No qualifying new three-principle round is claimed.
No 405B-capable hardware is currently available. The goal remains active and
unachieved. Native HF reachability/RNG/KV, 405B/GPU/latency/TTFT remain untested.
