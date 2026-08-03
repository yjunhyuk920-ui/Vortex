#!/usr/bin/env python3
"""Freeze EXP-058 authority and register EXP-059."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_058"
RUN = 30826618962
ARTIFACT = 8861905858
ARTIFACT_NAME = "exp-058-candidate-30826618962"
ARTIFACT_SIZE = 29349
ZIP_SHA = "851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae"
SOURCE_HEAD = "8ae03de4cc34317b5536aed42b9b8c22f98c88ea"
MERGE_SHA = "3730d6ce8ca89df347079c366a91bcad4d904a85"
CONFIG_SHA = "18356731d606c819da29807a98de600c8d4d515ff16b5d06c0b90613ee431906"
DECISION = "REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES"
MARKER = "<!-- EXP-058-AUTHORITATIVE-FINAL -->"


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


def main() -> None:
    summary_path = RESULT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    measured = summary["MEASURED"]
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-058 decision")
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
    if int(measured["full_integer_rank_proven_count"]) != 144:
        raise SystemExit("not every dense projection was proven full rank")

    all_rows = [
        json.loads(line)
        for line in (RESULT / "raw/matrix_rank_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    dense_rows = [row for row in all_rows if row["matrix_role"] == "dense_projection"]
    if len(all_rows) != 153 or len(dense_rows) != 144:
        raise SystemExit("rank row population mismatch")
    if any(
        int(row["certificate"]["rank_lower_bound"])
        != int(row["certificate"]["minimum_dimension"])
        for row in dense_rows
    ):
        raise SystemExit("a primary dense projection is not full rank")

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
## D-035/D-036 — Reject exact low-rank factorization and select shift-displacement Gate

EXP-058 authority: `results/exp_058/summary.json`; workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}`; ZIP SHA-256 `{ZIP_SHA}`. All 144 registered Q4 dense projections were proven full integer/rational rank with zero certificate, control, registration, or EXP-057 checksum mismatch. Prime 251 certified 143 matrices and prime 257 certified one. Favorable conventional exact two-factor operation and storage lower bounds were p50/p90 200%/200%. Decision: `{DECISION}`.

Full rank does not rule out fast full-rank structured transforms. EXP-059 therefore tests exact zero-fill and cyclic diagonal/anti-diagonal shift-displacement rank rather than another factor search.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-026 — Conventional exact low-rank factorization of measured real Q4 projections

All 144 pinned real-Q4 dense projections had certified rank `min(rows, columns)`. Conventional exact `W=A@B` therefore has favorable operation and factor-storage lower bounds of 200% before factor bitwidth, metadata, and kernel overhead. Do not revive this using approximate SVD energy, selected matrices, or a new factor optimizer while claiming exact output preservation. Retain modular rank certificates only as falsification infrastructure.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-058 closed — Exact algebraic rank of pinned real Q4 matrices

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. All 144 registered dense projections were proven full integer/rational rank. Favorable exact two-factor operation/storage lower bounds were 2.0x at p50 and p90. Decision: `{DECISION}`. Q4 output preservation, constructive factor kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-059 exact shift-displacement rank.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-034 — Full-rank real Q4 matrices may still have low exact shift-displacement rank

Status: ACTIVE FOR EXP-059 ONLY. EXP-058 ruled out ordinary exact low rank, not Toeplitz-, Hankel-, or circulant-like full-rank structure. EXP-059 must use exact registered displacement operators and modular certificates; visual banding and approximate spectral decay are not evidence.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-058 closure

Q4 checksum agreement 144/144; full integer/rational rank 144/144; certificate/control mismatches 0; p50/p90 exact factor operation lower bound 200%/200%; p50/p90 factor-storage lower bound 200%/200%. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Exact-rank boundary

Modular rank certification is retained as an offline structural audit. Conventional exact low-rank factors are prohibited for matrices certified full rank. The next permitted algebraic route is a different full-rank structured representation, beginning with EXP-059 shift-displacement operators.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-058 hardware status

No factor kernel was promoted because the favorable structural lower bound is already 2.0x on every measured matrix. CUDA, factor bytes, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency are NOT TESTED. EXP-059 remains a CPU structural Gate unless its exact displacement-rank thresholds survive.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-058 authority

Workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); artifact ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_058/reproduce.sh` and verify `results/exp_058/checksums.sha256`.''',
    )

    (ROOT / "NEXT_EXPERIMENT.md").write_text(
        '''# Next Experiment

## Closed Gate — EXP-058

All 144 pinned real-Q4 dense projections were proven full integer/rational rank. Favorable conventional exact two-factor operation and storage lower bounds were 200% at p50 and p90.

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

## EXP-059 — Pinned Real-Q4 Exact Shift-Displacement Rank Gate

### Mechanism

Full-rank Toeplitz-, Hankel-, and circulant-like matrices can still admit fast exact transforms. For every registered Q4 dense projection `W`, certify the exact integer rank of four displacement matrices:

```text
D_zero_diag  = W - shift_zero_down_right(W)
D_zero_anti  = reverse_columns(W) - shift_zero_down_right(reverse_columns(W))
D_cycle_diag = W - shift_cycle_down_right(W)
D_cycle_anti = reverse_columns(W) - shift_cycle_down_right(reverse_columns(W))
```

Use primes 251, 257, and 263. Record every operator certificate and select the most favorable operator only after all four searches are charged.

### Pinned population

Use the unchanged TinyStories-1M/3M/8M revisions, the exact EXP-057 Q4 rule, and all 144 named dense projections. Q4 checksums must match frozen EXP-057 evidence.

### Favorable lower bounds

For displacement rank `r` and shape `m x n`:

```text
query:   r * max(m, n) frequency-domain products
storage: r * (m + n) generator scalars
```

These omit transforms, boundary terms, metadata, bitwidth expansion, and operator-search runtime, so they favor the candidate.

### Controls

- random exact Toeplitz: zero-fill diagonal displacement rank <=2;
- random exact Hankel: zero-fill anti-diagonal displacement rank <=2;
- exact circulant: cyclic diagonal displacement rank 0;
- deterministic dense-random negative control;
- transpose, column-reversal, and cyclic-shift equivalence controls;
- exact EXP-057 Q4 checksum agreement.

### Promotion Gate

```text
zero certificate/control mismatch
zero Q4 checksum mismatch
zero unregistered dense projection
p50 query lower-bound fraction <=10%
p90 query lower-bound fraction <=25%
p50 generator-storage lower-bound fraction <=10%
p90 generator-storage lower-bound fraction <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

Phase C observation only. Q4 output preservation, constructive generators, exact transform kernels, real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
''',
        encoding="utf-8",
    )

    doc = ROOT / "docs/research/EXPERIMENT_058_REAL_Q4_RANK.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f'''# EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 153 two-dimensional tensors; 144 registered dense projections; Q4 checksum mismatches 0; certificate/control mismatches 0; full integer/rational rank proven 144/144; favorable exact two-factor operation/storage p50/p90 200%/200%; peak RSS 886,572 KiB.

Decision:

```text
{DECISION}
```

Ordinary exact low-rank factorization is rejected for this measured population. Q4 output preservation and target hardware were not tested.
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
