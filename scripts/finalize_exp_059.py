#!/usr/bin/env python3
"""Freeze EXP-059 authority and register EXP-060."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_059"
RUN = 30840432745
ARTIFACT = 8866573958
ARTIFACT_NAME = "exp-059-candidate-30840432745"
ARTIFACT_SIZE = 68652
ZIP_SHA = "61d0c24ccacd310d7d0e7600cc926a882c74281827d524c4880c6715fad8800d"
SOURCE_HEAD = "cdae6160cd87b537e2f318c16430619736c7c9d9"
MERGE_SHA = "82979e393a87845c4c757ce5dfd3fadc4e701d92"
CONFIG_SHA = "3e318ff909597e8b9ceca9b39b2a02caacc1427ce2b34132baa6ab7456003e62"
DECISION = "REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES"
MARKER = "<!-- EXP-059-AUTHORITATIVE-FINAL -->"


def append_once(path: str, body: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def write_checksums() -> None:
    lines = []
    for path in sorted(RESULT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(RESULT).as_posix()}"
            )
    (RESULT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    summary_path = RESULT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    measured = summary["MEASURED"]
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-059 decision")
    for key in (
        "certificate_or_control_mismatches",
        "missing_exp_057_q4_checksum_count",
        "q4_checksum_mismatches_against_exp_057",
        "unregistered_dense_projection_count",
    ):
        if int(measured[key]) != 0:
            raise SystemExit(f"authority failure: {key}={measured[key]}")
    if int(measured["dense_projection_matrix_count"]) != 144:
        raise SystemExit("unexpected dense projection count")
    if int(measured["operator_certificate_row_count"]) != 612:
        raise SystemExit("unexpected operator row count")
    primary = [
        row
        for row in read_rows(RESULT / "raw/matrix_rows.jsonl")
        if row["matrix_role"] == "dense_projection"
    ]
    if len(primary) != 144:
        raise SystemExit("primary matrix row population mismatch")
    if any(float(row["selected_displacement_rank_fraction"]) != 1.0 for row in primary):
        raise SystemExit("a primary displacement rank was not full")

    provenance = {
        "workflow_run": RUN,
        "artifact_id": ARTIFACT,
        "artifact_name": ARTIFACT_NAME,
        "artifact_size_bytes": ARTIFACT_SIZE,
        "artifact_zip_sha256": ZIP_SHA,
        "source_head_sha": SOURCE_HEAD,
        "workflow_merge_sha": MERGE_SHA,
        "config_sha256": CONFIG_SHA,
        "frozen_date": "2026-08-04",
    }
    (RESULT / "raw/artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["provenance"].update(provenance)
    frozen = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(frozen, encoding="utf-8")
    (RESULT / "processed/aggregate.json").write_text(frozen, encoding="utf-8")
    write_checksums()

    append_once(
        "DECISION_LOG.md",
        f'''{MARKER}
## D-037/D-038 — Reject exact shift-displacement structure and select zero-sparsity Gate

EXP-059 authority: `results/exp_059/summary.json`; workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}`; ZIP SHA-256 `{ZIP_SHA}`. Four registered exact displacement operators were certified for every two-dimensional tensor. For all 144 dense projections, even the favorable selected displacement rank was 100% of the minimum dimension. Favorable query lower bounds were p50/p90 100%/100%, generator storage was 200%/200%, and the best real matrix still required 100% query work and 125% storage. All controls, registration, and EXP-057 Q4 checksums passed. Decision: `{DECISION}`.

D-038 closes the tested Toeplitz/Hankel/circulant-like exact route. EXP-060 measures a simpler orthogonal possibility: exact scalar zeros and all-zero blocks in the same pinned Q4 matrices, with index and byte costs fully charged.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-027 — Exact shift-displacement structure on measured real Q4 projections

All 144 pinned real-Q4 dense projections retained full displacement rank under the most favorable of zero-fill/cyclic diagonal/anti-diagonal operators. Favorable query lower bounds were 100% at p50 and p90; generator storage was 200%. Do not continue by adding more visually chosen shifts, selecting a tensor subset, or ignoring transform/boundary costs. Retain displacement certificates only as structural falsification tools.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-059 closed — Exact shift-displacement rank

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. All 144 registered real-Q4 dense projections had selected exact displacement-rank fraction 1.0. Favorable query and generator-storage lower bounds were p50/p90 1.0/1.0 and 2.0/2.0. Decision: `{DECISION}`. Q4 output preservation, constructive generators, transform kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-060 exact zero-sparsity streaming.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-035 — Real Q4 matrices may contain enough exact scalar/block zeros for sparse streaming

Status: ACTIVE FOR EXP-060 ONLY. Ordinary rank and displacement rank do not measure zero sparsity. EXP-060 must account for every stored value, index, row pointer, padded scalar in nonzero blocks, and format search. Q4 model-output preservation remains a separate unverified assumption.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-059 closure

Q4 checksum agreement PASS; registration 144/144; operator certificates 612; control/certificate mismatches 0; p50/p90 selected displacement-rank fraction 100%/100%; favorable query lower bound 100%/100%; favorable generator storage 200%/200%. Hardware and 405B remain NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Displacement-structure boundary

The runtime must not promote the tested zero-fill/cyclic diagonal/anti-diagonal generator route for matrices with full certified displacement rank. Certificates remain offline audits. The next permitted exact structural path is sparse streaming conditioned on measured Q4 zero scalars or complete zero blocks.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-059 hardware status

No transform kernel was promoted because favorable query work already equals dense work and generator storage exceeds dense storage. FFT/NTT execution, CUDA, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency remain NOT TESTED.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-059 authority

Workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_059/reproduce.sh` and verify `results/exp_059/checksums.sha256`.''',
    )

    (ROOT / "NEXT_EXPERIMENT.md").write_text(
        '''# Next Experiment

## Closed Gate — EXP-059

All 144 pinned real-Q4 dense projections had full selected shift-displacement rank. Favorable query lower bounds were 100% at p50/p90 and generator storage was 200%.

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

## EXP-060 — Pinned Real-Q4 Exact Zero-Sparsity Streaming Gate

### Mechanism

Measure whether deterministic Q4 matrices contain enough exact zero scalars or completely zero blocks to skip original multiply-adds and weight reads without approximation. Compile and account for:

```text
dense Q4 baseline
scalar CSR
row-wise nonzero-run encoding
BSR 1x4, 1x8, 4x4, 8x8, and 16x16
```

Only exact zeros may be skipped. A nonzero BSR block charges every scalar slot in that block, including internal zeros.

### Pinned population

Use the same TinyStories-1M/3M/8M revisions and all 144 named dense projections. Recompute deterministic row-symmetric Q4 and require exact checksum equality with EXP-057.

### Accounting

- dense operations: `m*n` multiply-add terms;
- CSR/run operations: exact nonzero scalar count;
- BSR operations: scalar slots in nonzero blocks;
- packed Q4 value bytes;
- column/block indexes using the minimum whole-byte width;
- CSR/BSR row pointers;
- run start/length metadata;
- alignment padding and edge blocks;
- all format compile/search work recorded separately.

Select the best format only after all formats are compiled and charged. Report operation and query-byte fractions independently.

### Controls

- highly sparse synthetic matrix must compress below 10%;
- dense-random Q4 matrix must not falsely compress;
- isolated-zero adversary must expose BSR padding waste;
- block-zero positive control must favor its registered BSR shape;
- exact reconstruction from every serialized format;
- row/column permutation changes format statistics but not reconstructed values;
- EXP-057 Q4 checksum agreement.

### Promotion Gate

```text
zero reconstruction/control/checksum mismatch
zero unregistered dense projection
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-byte fraction <=10%
p90 query-byte fraction <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY
```

Phase C observation only. Q4 model-output preservation, physical sparse kernels, actual Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
''',
        encoding="utf-8",
    )

    doc = ROOT / "docs/research/EXPERIMENT_059_REAL_Q4_DISPLACEMENT_RANK.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f'''# EXP-059 — Pinned Real-Q4 Exact Shift-Displacement Rank Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 612 operator certificates; checksum/control/certificate mismatches 0; selected displacement-rank p50/p90 100%/100%; favorable query p50/p90 100%/100%; storage p50/p90 200%/200%; peak RSS 1,581,260 KiB.

Decision:

```text
{DECISION}
```

The tested Toeplitz/Hankel/circulant-like route is rejected for this measured population. Q4 output preservation and target hardware were not tested.
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
