# V1 source preservation

The exact pre-repair LF-byte SHA-256 identities are recorded in `SHA256SUMS`.
They are preserved with immutable `scalar_results_v1`; the pre-repair defect
was `leaves = [0.0] * store.cols` in `project_fp32` and acceptance of
`max_support=3`, which summed only its first two active values.

The v1 source bytes were NOT retained. Only the hashes and immutable result
were saved; this limits independent reproduction of the historical defect.
Primary static inspection identified the dropped term before the repair.
The later regression checks the full result and that v2 refuses the input;
it is NOT an execution record of the original v1 code. The static example is:
`[[0x3f80,0x3f80,0x3f80]]` and three `0x3f80` inputs had full result `0x4040`
but the v1 sparse routine returned `0x4000` when called with `max_support=3`.
