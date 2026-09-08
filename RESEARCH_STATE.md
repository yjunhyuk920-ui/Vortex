# Research state — 2026-09-08

Fixed mission / CTC-2026-09-05 unchanged.
[Current bounded construction](experiments/nonlinear_coordinates_20260908/REPORT.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed common nonlinear bijective state coordinates, first via explicit finite enumeration, then by checking actual reversible inverse-paired IR and its observer. Source-only Python/C transition routines preserve original states inductively. Low rank, small residual and similar consecutive inputs are not assumed, but exact constructed reversible structure IS assumed. No original Transformer was mapped into this form.

Enumeration:24attempts,8accepted,16refused,43520complete pairs. Direct IR:87040pairs for two distinct ORIGINAL observation contracts.6144causal steps/independent LCG/own-output input selection agree.12bit source328B equals original compact code.216reversible gates eliminated online for O=E; O=id still executes72decoder gates. Same affine multiply/add/mask remains. Initial encoding72gates and generator720primitive visits plus648tuple-comparison upper bound count. No timing ratio is proved by mixing these operation classes.

Finite source16424B and20488transition calls at12bits; preparation alone2561times eight original transition calls. Exponential enumeration is not admissible at scale. Direct compiler avoids this but is an IR recognizer, not a generic native nonlinear compiler.

CandidateB requires explicit input-chart switching; omitting it fails. Noncommuting affine positive controls prevent an overbroad no-go. CandidateC:one BF16 RNE output has65537FP32 predecessors and needs17fixed inverse side bits in that local output-only problem; NOT per-token loss/global Transformer noninjectivity. An explicit noninjective conjugate with floor(z/2) remains cheap.

22tests/109sciencefiles match final frozen manifest56c016845ebee52901283b15680f7460270f4a3ef9ecb0465a1bde1c4c3b728d. Earlier manifest and pre-scope-addition files are preserved in userZIP. Source/tests/report/prereg/frozenmanifest are remote; rawbinary regenerates. No full native/HF/KV/RNG/GPU/405B/8GiB/4BQ4/TTFT test.
[Prior state unchanged](docs/research/history/pre_nonlinear_coordinates_20260908/RESEARCH_STATE.md).
