# Research state — 2026-09-07

The fixed mission and CTC-2026-09-05 remain unchanged. No universal cheap checkpoint-derived source, whole-state/RNG executor or sufficient 405B/8-GiB/native4BQ4 bound was constructed. This is not a hardware-only gap.

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

[Current bounded algorithm, proof, budget and reproduction](experiments/signed_orbit_20260907/REPORT.md): finite BF16 -> ordered FP32 rounded-tree signed-orbit compiler, two-slot reference and optimized C zero/nonfinite mirror rules. Exactness is independent of sparse residual/rank/small input change; small graph size is not guaranteed. This remains auxiliary E1, not a new universal computation principle.

C returns 87,040 matching coordinates over 192 registered queries. Four structured 2048 controls have small code/logical traffic; generic BF16 controls fail the leaf budget, including two 256 cases refused before full construction. Conservative integer/address accounting does not certify complete work <=10%. Logical access is not measured memory traffic or latency.

Preparation retains full W/refs and several scans; magnitude leaves, code addresses, scratch and local exception work are paid. Original CUDA/FMA ABI, Transformer/KV/RNG continuation and target resource closure remain OPEN. A small non-Transformer state/RNG smoke test is not full-state validation.

Twenty unit tests and exact regeneration of 188 files plus two manifests validate the bounded code. Initial incorrect graph-count test failure and subsequent preregistered refinement are preserved. No public checkpoint, GPU, 405B, 4B baseline, TTFT, full-repository suite or Actions was run.

[Previous state preserved byte-for-byte](docs/research/history/pre_signed_orbit_20260907/RESEARCH_STATE.md). Architecture is not promoted; historical ledgers/policies remain unchanged. Report remote handoff only after branch/commit read-back.
