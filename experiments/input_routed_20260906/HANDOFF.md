# Input-routed source handoff

Parent commit: 147dad60b4450d7f32de6e3b24d276b075aed817 (PR #128, draft/open).
Research branch: research/input-routed-native-source-20260906.
No main write/merge, force push, external checkpoint change, or workflow dispatch.
Remote verification must be read back after committing; this file does not pre-claim it.

## Fixed mission / result
The universal exact 405B/total8GiB/native4BQ4 quantile/TTFT mission is unchanged.
THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false.
Bounded auxiliary E1 construction, not three newly admitted principles.
Full mission obligations:
- O1 OPEN: only bounded integer weight/input source, not all HF checkpoints.
- O2 OPEN: projection output only; no loading/prefill/attention/token/RNG/state engine.
- O3 OPEN: native proof for stated integer slice; no generic native/continuation theorem.
- O4 OPEN: local payload/operation/cap counters supplied; no full machine sufficient bound.
- O5 OPEN: no target 8GiB/quantile/TTFT closure or measurement.
- O6 OPEN for full theory: reproducible local code/raw observations are a bounded artifact.

## Local reproduction
Python 3, NumPy. Actual environment is in the packed measurements record.

    python src/evidence.py unpack results recorded_results
    python src/experiment.py --out regenerated_results
    python -m unittest discover -s tests -v

`results/evidence.part00..04` are binary slices of one gzip stream, not independent
compressed files. The decoder concatenates them and reconstructs exact JSON/JSONL
files, validating original SHA256 hashes. Rounding references equal to the observed
output are encoded as an explicit equality flag; unequal references have a separate
field, so the format does not silently omit disagreement.
The full frozen inputs, all 32 outcomes, every completed query/result/counter, local
measurement record and validation hashes are included. VRC1 programs are generated
from source; their original SHA256 values and byte sizes are in the evidence. The
local ZIP also contains the actual generated binaries.

15 unit tests passed; ten deterministic input/result/program files regenerated
byte-identically. Evidence packing/unpacking additionally restored all six original
JSON/JSONL records byte-identically. No full-repo tests, Actions, public checkpoint,
full state/RNG, GPU, 405B, baseline latency or TTFT run.

## Do not misread the outcome
Small-shape root overhead alone prevents the registered toy 10% gate. The pilot
therefore cannot be extrapolated as a target-scale speed limit. Cap refusal is not
a minimum-circuit lower bound. Neither tiny ratios nor a successful persistence
commit is a completed engine. Read the report before any next research.
No cap/order sweep or GPU port without a genuinely new source mechanism that
controls both native code size and demand-selected query cost.

All four changed root entrypoints preserve prior bytes and append scope/status.
Earlier local bitplane/SwiGLU/event bundles were not silently merged into this round.
