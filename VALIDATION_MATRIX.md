# Validation matrix — 2026-09-08

[Current native proof/code/replay](experiments/native_global_transition_20260908/REPORT.md).
[Pre-native-global matrix unchanged](docs/research/history/pre_native_global_transition_20260908/VALIDATION_MATRIX.md).

|Current item|Actual evidence and scope|
|---|---|
|Original|Complete pinned SmolLM2-135M BF16 file269060552B, upstream LFS SHA verified; torch2.8.0+cpu/HF4.55.4|
|A|Guarded make_fx graph, exact pure CSE/DCE,61live roots, serialized/reloaded original tensor-name bindings|
|B|Actual byte/layout cache encoding and decoding; full original forward every step, costs charged|
|C|Actual TOP/singleton root-demand evaluation,0backward narrowing,no supplied desired answers|
|Generation|3fixed prefixes,12forward steps perarm; own native sampler;589824logits/760320KV coordinates allmatch|
|Additional roots|Full prefill196608logits plus60KVroots match; shape/stride/offset/cachefields/RNG match|
|Mask clarification|Explicitall-one mask/original auto-pad0 vs old automaticmask reference, unchanged outputs/RNG/sequences|
|Costs|A/C100%matrixMACs; A5/C4captureforwards; B12fullforwards+5114880BmincodecRW; program11775553B+originalweights|
|Scope witness|Direct nativeRoPE inverse fails144/211/263 of768keycoords at synthetic positions1/7/31; not allcharts|
|Tests/replay|14tests;3510fixedsciencefiles replay;3volatiletimefields excluded, originaltimings retained|
|OPEN|Universal O1-O6,alllegal masks/context/platforms,CUDA/405B/8GiB/native4BQ4/TTFT/fullrepositorysuite|

## Prior packet screen (unchanged scope)
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
