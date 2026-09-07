# Research state - 2026-09-07

Fixed mission/CTC-2026-09-05 unchanged.
[Current algorithm, proof, costs and evidence](experiments/causal_cut_20260907/REPORT.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed finite-vocabulary native-prefix table, checkpoint/ABI binding and
pread-only query, with first Q/K/V/norm arrays removed from the synthetic runner.
48 IDs/2528 coordinates match;160 independent causal steps preserve logits/KV/RNG.
This discharges a bounded replacement lemma only. The same token/position has
different second-layer values after different histories. Generic weight structure
at this small cut does not imply a universal cheap history-dependent source.

Official405B scale calculation removes0.0747966466% of projection MAC, with4.4033GiB
extra table and38732015075328 construction MAC. Not a latency or VRAM measurement.
Exhaustive next branches and lazy KV also lack a whole-work10x route; stopped before
large backends. Partial evaluation is known; no claim of three new principles.
20 tests/32-file reproduction passed. No public HF/CUDA/405B/4B/TTFT/Actions/full-repo
suite. Prototype has whole small model and temporary copies, not a target memory bound.

[Previous state unchanged](docs/research/history/pre_causal_cut_20260907/RESEARCH_STATE.md).
Report contains new decision/assumption/failure addendum; old root ledgers unchanged.
Remote receipt is established by ref/PR/content read-back, not by this status file.
