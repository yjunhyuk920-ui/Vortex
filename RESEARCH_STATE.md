# Research state — 2026-09-07

The fixed mission and CTC-2026-09-05 remain unchanged. No universal cheap checkpoint-derived native source, whole state/RNG executor or sufficient 405B/8-GiB/baseline bound was constructed. This is not a hardware-only gap.

```
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

[Actual algorithm, scope, cost and reproduction](experiments/joint_energy_20260907/REPORT.md): four-moment <=2-coordinate recovery plus exact Gram energy verification; subsequent five-moment nonnegative annihilator with no Gram/residual file. Conditional correctness is proved in the guarded integer-to-native reference domain. The source does not assume a perfect selector or silently use a dense fallback.

Energy verification pays a dense n-by-n Gram. The positive path needs R>=0 and x>=0; on strictly positive dense inputs its s<=2 success requires rank(W)<=3. Signed decomposition can destroy sparsity, and general post-rounding source generation is not supplied. Large int64 moment overflow is refused, not hidden.

27/71 registered queries return exact outputs (2,368 coordinates); the remaining 44 return no output. Deliberately favorable controls are not public-model evidence. Both ordinary dense groups return no results. Twenty-one unit tests and 63 manifest-listed deterministic artifacts validate limited code, not the fixed mission.

All O1-O6 remain OPEN at mission scope. A detailed sub-obligation ledger and full Korean proof report are preserved in the checked archive. No public checkpoint, Transformer/KV/RNG, GPU, 405B, 4B baseline, TTFT, full-repository suite or Actions was executed.

[Previous state preserved byte-for-byte](docs/research/history/pre_joint_energy_20260907/RESEARCH_STATE.md). Prior policies and all historical evidence are unchanged. Remote handoff is reported only after actual branch/commit read-back.
