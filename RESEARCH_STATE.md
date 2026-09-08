# Research state — 2026-09-08

Fixed mission and CTC unchanged. [Current bounded construction](experiments/output_envelope_20260908/docs/REPORT_KO.md).
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

HARDWARE_STATUS concerns the target; no timing benchmark in this round.

Constructed native row-envelope query: min/max per coordinate, sign-aware RNE
endpoints, only nonzero finite BF16 singletons broadcast; otherwise original rows
read once. Full output, no input bank/future output/previous-input similarity.

Pinned SmolLM2-135M93efa2f097d58c2a74874c7e644dbc9b0cee75a2,12unaltered matrices,
96syntheticvectors,101376BF16coordinates:0mismatch;36Fractiondots. 0of432packets
have a common actual word. Cold reads100%, interval+direct FPwork/coefficient
payload100.78%-101.04%, other costs extra. Bound tightening cannot repair these
fixed broadcast packets; no all-algorithm or reachable-activation lower bound.

Real weight provenance established, but HF activations/forward/ABI and fullKV/RNG
not established. Originalpayload17,915,904B,summary158,976B; originals retained.
Planted256x64control is separate.12tests/38numerical replays with frozen hashes.
Need a cheap exact heterogeneous effect source. Other-session correlation_source
and local-only mantissa_source are not silently imported. Correlation_source is now
committed at PR147/378fcd584330d17f9d144f64ccbc01721903ea6a on another branch; see
[frontier reconciliation](experiments/output_envelope_20260908/FRONTIER_SYNC.md).
[Prior state](docs/research/history/pre_output_envelope_20260908/RESEARCH_STATE.md).
