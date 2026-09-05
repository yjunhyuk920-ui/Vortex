# Validation — latest-source/native-observation continuation, 2026-09-05

[Proof, cost and source scope](docs/research/LATEST_RESEARCH_NATIVE_OBSERVATIONS.md).
[Previous matrix](docs/research/history/pre_frontier_20260905/VALIDATION_MATRIX.md)
is preserved unchanged. Prior serial-fmaf proofs remain scoped, not GPU ABI proofs.

| Claim | Evidence | Status |
|---|---|---|
| New literature supplies a full VORTEX solution | Eleven source records scoped by precision/hardware/baseline | Not established |
| Prefix source generates the exact declared normal block store | Containment/termination proof; 256 independent integer-reference comparisons | Scoped, zero mismatches |
| Prefix-box stored-value enclosure | All 65,536 completions of one registered prefix box | Passed, not all hardware inputs |
| FP32-RZ and BF16-RNE reference handling | 4,096 independent host/bit-level controls plus focused tie/guard tests | Passed in stated domain |
| Same real products suffice for native-model stored equivalence | Locally reproduced raw-operand witness gives BF16 0x4011 vs 0x4010 | False in model |
| Different private FP32 values may have identical actual stored cuts | Exact cast example plus cut-placement counterexample and induction rule | Sufficient conditional rule |
| Literal header/refinement source gives >=10x | Header read floor 9/16; observed reads 90.9332%; endpoint work counted | REJECT CORE / RETAIN AUXILIARY |
| Complete public Transformer state/RNG/ABI mapping | No actual checkpoint or complete native kernel executed | OPEN |
| 405B, total 8 GiB, same-machine p50/p95/TTFT | No target execution or sufficient full-theory bound | NOT TESTED / NOT ESTABLISHED |

Twelve focused unit tests passed. Deterministic captures are regenerated and
compared in local validation. The compact capture is a lossless encoding of
test records, not compressed model weights. No full repository tests or GPU
measurements are implied. Source guards added for unsupported numerical ranges
do not alter the preregistered input distribution or acceptance threshold.

THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false.
Repository persistence is verified independently after the commit exists.
