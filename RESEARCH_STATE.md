# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Bounded source construction, proof, costs](experiments/symbolic_source_20260907/REPORT.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Small source expressions are compiled without Q^n response enumeration using a
finite candidate grammar and IEEE SMT miters over all finite BF16 inputs. Only
UNSAT promotes; two UNKNOWN cases retain baseline. 28678 C replays match, including
unchanged programs. State erasure, signed zero and overflow counterexamples persist.
A follow-on exact-product lemma yields constant-only guards for local FMA/same-input
rewrites. 448 vectors,1664 FP32 and1664 BF16 coordinates match independent Fraction
arithmetic. The BF16 intermediate-store witness x_bits3 gives1 vs2 in subnormal units.

These are bounded compiler/native proof auxiliaries. Weight literals stay, FMA keeps
multiply+add work, and generated code exceeds source tensor size. No >=10x general
route; F-040 universal hot-core synthesis is not reopened. SMT search is paid,
version-dependent and not an independently checked proof certificate. Full HF/causal
state/target memory/latency remain open. No public model/GPU/4BQ4/TTFT test.

22 tests,56 scientific file hashes,11-file source capsule freshly restored/replayed.
All originals/history/logs in user archive; source regenerates scientific data, not
all traces embedded remotely. [Prior state unchanged](docs/research/history/pre_symbolic_source_20260907/RESEARCH_STATE.md).
