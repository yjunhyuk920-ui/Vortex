# Validation matrix — 2026-09-08

[Current proof/code/replay](experiments/output_envelope_20260908/docs/REPORT_KO.md).
[Prior matrix unchanged](docs/research/history/pre_output_envelope_20260908/VALIDATION_MATRIX.md).

|Item|Evidence and scope|
|---|---|
|Constructor|Paid entire-matrix extrema scan; row-packet lower/upper BF16 arrays, originals retained|
|Runtime|Sign-aware native interval tree; singleton broadcast or direct original rows; Python/NumPy|
|Native proof|FiniteBF16/separateFP32RNE/fixed balanced/+0padding/canonicalNaN/finalBF16; no real HF ABI claim|
|Corpus|Pinned originalSmolLM2-135M12matrices,96syntheticinputs,101376BF16coordinates,0mismatch|
|Independent check|36full-length FractionRNE dot products; main candidate/reference share NumPy elementary operations|
|Tests|12passed; source/load, signedzero, nonfinite intermediates, nonpower2, read-accounting controls|
|Costs|accepted_groups=0;oracle_broadcastable_groups=0 of432;originalreads100%;coefficient/FPwork100.78%-101.04%|
|CPU observation|No candidate latency benchmark; recorded acquisition/build durations are diagnostic only|
|Reproduction|Initial39files hashed;38numerical result files regenerate excluding later source-snapshot metadata|
|Remote evidence|Source/input/output/packettrace/hash/log preserved;upstream weight and derived plan payloads ignored/reacquired|
|Whole mission|O1-O6OPEN;CORE_ADMISSION=false;THEORY_STATUS=NOT_ESTABLISHED;targetHARDWARE_STATUS=NOT_TESTED|
|Not tested|HF_FORWARD=NOT_TESTED;FULL_KV_RNG=NOT_TESTED;TARGET_405B=NOT_TESTED;realactivations/CUDA/8GiB/native4BQ4/TTFT/fullrepositorysuite|

Scientific acceptance and persistence are independent. No claim of three new qualifying principles, no universal low-coherence assumption, no full-model speed inference.
