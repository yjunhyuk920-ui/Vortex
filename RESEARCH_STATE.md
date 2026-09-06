# Research state — 2026-09-07

Fixed mission and CTC-2026-09-05 unchanged. [Current bounded source/proof/budget](experiments/boundary_convolution_20260907/REPORT.md).

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

Constructed streaming exact-Toeplitz BCV1 source, signed/absolute polynomial query,
and rational enclosure of original balanced FP32 tree followed by BF16 store.
No original matrix at query time; information remains in packed source and its costs.
This differs from same-input-position sign/dyadic subtree sharing, but uses known
structured/Kronecker tools and is NOT a new universal execution principle.

32 builders:16 constructed,16 refused.128 structured queries:86 exact vectors,42
unresolved;53916/54272 certified coordinates match. All native FP32 sums enclosed;
53937 are not exact real sums. Planted1024 source files are1.7583%-2.1487% of raw
weights, NOT traffic/speed. Ordinary1024 whole-query success is0/16. Generic rank
minors rule out the specific64-bit generator budget, not all representations.

Uniform construction, full native/state/RNG continuation, all-work resource upper
bounds and405B closure remain OPEN, not just unmeasured hardware. Bigint products,
packing, certificate work, failures and preparation are paid, never unit-cost/free.
15 unit tests;435 output files plus manifest reproduce. No public HF, Transformer,
GPU,405B,4B baseline,TTFT,full-repo suite or Actions was run.

[Previous state retained byte-for-byte](docs/research/history/pre_boundary_convolution_20260907/RESEARCH_STATE.md).
Architecture is not promoted. New decision/assumption/failure addendum is in the report;
old root ledgers and policies remain unchanged. Verify remote SHA before handoff claim.
