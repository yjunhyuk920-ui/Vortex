# Validation matrix — 2026-09-07

[Previous matrix preserved byte-for-byte](docs/research/history/pre_signed_orbit_20260907/VALIDATION_MATRIX.md). No historical gate is promoted. [Current report](experiments/signed_orbit_20260907/REPORT.md).

| Item | Evidence and exact limit |
|---|---|
| Finite constructor/query | Magnitude leaves, ordered signed-orbit DAG, SORB1 file and Python/C execution; W not retained by query, information remains in code |
| Native proof | Structural induction for separate FP32 multiply + balanced adjacent FP32 RNE tree + final BF16 RNE; not actual HF/CUDA/FMA ABI |
| Zero/nonfinite refinement | Exact mirror-zero rule; local mirrored add for nonfinite nodes; fixed CPU NaN semantics only |
| Original registered suite | 24 cases, 22 executions / 2 preflight cost refusals; 176 queries / 54,272 coordinates match |
| C follow-on | Same old queries plus 16 new; 192 queries / 87,040 coordinates match; four separate adversarial cases |
| Independent arithmetic | 16,000 integer-oracle primitive comparisons and 72 small whole outputs match |
| Favorable controls | Planted Walsh / row-sign permutations, dense and full-rank; not public learned models |
| Costs | 2048 code 2.442%, logical data at most about 6.11%; word+FP envelope about 18.85%; no complete 10x or latency proof |
| Generic controls | Many distinct magnitude leaves; no comparable savings; 256 cases refused early |
| Local tests | 20 pass, including 200 C fuzz cases and non-Transformer recurrent state/RNG smoke; initial wrong node-count failure preserved |
| Reproduction | 188 files and both manifests regenerate byte-identically; 34 capsule text files restore; restored C build and 20 tests pass |
| Full mission O1-O6 | OPEN; core not admitted |
| Public HF / Transformer KV / CUDA / GPU / 405B / <=8 GiB / native4BQ4 / TTFT | NOT TESTED |
| Full repository / Actions | NOT RUN |

Capsules contain exact source, preregistration, logs, per-query text records, full report and hashes. Binary arrays/program files are deterministically regenerated and hash-checked, not all embedded remotely. Logical byte/word counts are not physical traffic, peak VRAM or latency. Remote commit verification is a separate persistence axis.
