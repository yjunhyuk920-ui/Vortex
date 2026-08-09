# Run the Causal Bilinear Rank Gate

Type: experiment
Status: open
Blocked by: Certify Causal Bilinear Query Restriction

## Question

Do the pinned Qwen3.5-0.8B last-`down_proj` causal Bilinear Cross Residual
query tensors occupy an exact span small enough for the registered `B<=23`
factor-scan ledger, or does their certified held-out rank reach the frozen
rank-28 rejection threshold?

Implement and execute only the preregistered Gate in
`docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md`. Before any model row,
pin the runner, checkpoint/config/weight/prompt hashes, six build prompts x
four genuine post-prefill positions, 18 disjoint evaluation prompts x two
positions, prompt-only two-pass MGS side rank 16, winner/strongest-competitor
decision direction, exact float32 dyadic query construction, primes
`65521/65519/65497`, exact rational hit witnesses, family controls, result
schema, checksums, and stop decisions.

Fail closed on any leakage, prefix-replay mismatch, nonfinite pair, false hit,
or fingerprint failure. Stop scientific execution as soon as rank 28 is
certified or more than four exact build-ledger misses are inevitable. A pass
requires all 36 evaluation rows, rank at most 27, at most four exact misses,
at least five exact hits in every six-row family, and every witness/control
passing.

Do not change span dimension, side rank, primes, prompts, positions, decision
direction, or hit semantics after seeing model data. Do not contact the Ubuntu
server, download another model, start hardware work, implement a general dense
VJP extractor, or promote a pass beyond the paid extractor/native-semantics
Gate. Do not reopen Atlas, activation replay, matrix-local codes, or free
cross-matrix projection reuse.
