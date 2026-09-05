# VORTEX

Proof-first research for arbitrary public, unmodified Hugging Face dense
Transformers. **The 405B / single total-8-GiB GPU / native-4B-class latency mission
is not achieved, and a complete constructive theory is not established.**

## Fixed mission and current contract

Executor replacement only; no retraining, weight changes, hidden remote compute,
free precomputation or uncharged fallback. Batch one, original output/RNG and
required successor state; warm p50 <=1.2x and p95 <=1.5x native 4B Q4 on the same
machine, with the existing TTFT requirement. Every construction, storage,
CPU/RAM/SSD/GPU/KV/transfer/metadata/verification/repair cost counts.

Read [AGENTS.md](AGENTS.md), [the canonical mission](MISSION_AND_WORKING_PRINCIPLES.md)
and [the constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
Full-theory acceptance requires O1-O6, not a scoped lemma, test suite or commit.
Local validation precedes research-branch persistence and remote hash read-back;
GitHub Actions is not required unless explicitly requested.

## Latest scoped work — native ordered transfers, 2026-09-05

[Construction, proofs and cost ledger](docs/research/NATIVE_TRANSFER_CONSTRUCTION.md)
provide an exact guarded two-phase summary of ordered binary32 FMAs, a causal
suffix-synchronization reference and a direct rounding-cell inverse constructor.
These preserve the declared **serial-fmaf** ABI in their proved scopes, not an
unverified replacement for arbitrary Torch/cuBLAS reduction order.

The principal unresolved construction remains cheap exact query-dependent
information generation and complete state/cost closure. The phase constructor
still reads all terms. On 36 ordinary synthetic suffix cases none certified
early; all weights were read and FMA work was 2.75x–2.99609375x the reference.
The synchronizing positive control is not a target speedup. No core was promoted.

20 focused tests pass. Independent native/integer controls and direct-inverse
checks pass; scientific JSON/JSONL regeneration is byte-identical. No public
checkpoint forward, layer replacement, full-repository suite, GPU allocation,
TTFT or same-machine latency measurement was performed.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
```

Remote persistence status is established by the actual post-commit receipt,
not by this snapshot. Previous PR #119's five-bit minimum remains scoped E1;
its 2x3 solver UNKNOWN and earlier failures are unchanged. Separate
EXP-103A..108A branches are not merged or superseded.

## Reproduce

Python standard library and Linux libm `fmaf` with nearest-even rounding are
required; pytest is optional because the focused tests use unittest.

```bash
python experiments/native_transfer/audit.py
python experiments/native_transfer/audit_inverse.py
python -m unittest discover -s tests/native_transfer -v
python -m pytest -q tests/native_transfer
python experiments/native_transfer/evidence_codec.py --out results/native_transfer/decoded
sha256sum -c results/native_transfer/checksums.sha256
```

The two large inverse captures are losslessly integer-predictor/residual encoded
with original byte hashes in `results/native_transfer/inverse_recording.json`.
Decoding uses an exact integer IEEE reference, not native fmaf and not a new
hardware test. See the research note for provenance and literal-capture hashes.

Current [state](RESEARCH_STATE.md), [next construction](NEXT_EXPERIMENT.md),
[validation matrix](VALIDATION_MATRIX.md), and
[additive decision/assumption ledger](docs/research/NATIVE_TRANSFER_LEDGER.md).
Historical root snapshots are preserved byte-for-byte as Git blobs under
`docs/research/history/pre_native_transfer_20260905/`. Older anti-repetition and
scientific ledgers remain binding in their recorded scopes.

`vortex_runtime/` retains prior prototypes; `experiments/`, `tests/`, `results/`
and `docs/research/` contain research and evidence. No existing executable module
was changed in this round. The wider repository setup remains experiment-specific;
this isolated reference does not certify its full dependency environment.

```text
README_CURRENT=true
README_UPDATED=native construction; failed charged core screen; remaining O1-O6; reproduction and evidence
```
