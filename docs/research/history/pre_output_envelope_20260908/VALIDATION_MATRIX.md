# Validation matrix — 2026-09-08

[Current proof/code/replay](experiments/native_row_frontier_20260908/REPORT.md).
[Prior matrix unchanged](docs/research/history/pre_native_row_frontier_20260908/VALIDATION_MATRIX.md).

|Item|Evidence and scope|
|---|---|
|Constructor|Exact subtree interning, paid O(m^2 N) MST, sorted BF16 edits; source replay equals original W|
|Runtime|Independent C source parser; first row then changed leaves/ancestor union; no original matrix pointer or output bank|
|Native proof|Induction for finite BF16/separate FP32RNE products/fixed balanced +0padding/canonicalNaN/finalBF16; not arbitrary CUDA ABI|
|Corpus|18matrices;360queries;80640FP32/BF16coordinates all returned,0mismatch;76692finite;378Fractionchecks|
|Scalar scope|65280finite scalar inputs at4weights; separate from full-vector corpus|
|Tests|14passed including nonpower2/signedzero/subnormal/overflow/metric/MST/parser|
|Costs|512coherent source1.48-1.72%,1%perturb5.95-6.02%,random~200%; hot heap/index/constructor charges separate|
|CPU observation|Post-hoc batch1 vs handwritten C only; favorable~20x/~7.7x warm median; random~1.6x slower; compile/load excluded|
|Reproduction|147sciencefiles; manifest e3018ea424d0e3e2b6b91f20bc3f579eabc33e3f1df6635762d2ad17081eec42; fresh capsule restore and C rebuild passed|
|Remote evidence|Checked source capsule plus report/restore; deterministic raw science regenerates. Original raw CPU samples, Korean detailed report and available logs user-ZIP-only|
|Whole mission|O1-O6OPEN;CORE_ADMISSION=false;THEORY_STATUS=NOT_ESTABLISHED;targetHARDWARE_STATUS=NOT_TESTED|
|Not tested|Publiccheckpoint/HF/fullKV/RNG/CUDA/405B/8GiB/native4BQ4/TTFT/fullrepositorysuite|

Scientific acceptance and persistence are independent. No claim of three new qualifying principles, no universal low-coherence assumption, no full-model speed inference.
