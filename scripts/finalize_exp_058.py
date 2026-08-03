#!/usr/bin/env python3
"""Freeze EXP-058 authority and register EXP-059."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_058"

WORKFLOW_RUN = 30826618962
ARTIFACT_ID = 8861905858
ARTIFACT_NAME = "exp-058-candidate-30826618962"
ARTIFACT_SIZE = 29349
ARTIFACT_ZIP_SHA256 = "851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae"
SOURCE_HEAD = "8ae03de4cc34317b5536aed42b9b8c22f98c88ea"
WORKFLOW_MERGE = "3730d6ce8ca89df347079c366a91bcad4d904a85"
CONFIG_SHA256 = "18356731d606c819da29807a98de600c8d4d515ff16b5d06c0b90613ee431906"
DECISION = "REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES"
MARKER = "<!-- EXP-058-AUTHORITATIVE-FINAL -->"


def append_once(path: str, text: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(RESULT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(RESULT).as_posix()}")
    (RESULT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_path = RESULT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-058 decision")
    measured = summary["MEASURED"]
    required_zero = (
        "certificate_or_control_mismatches",
        "missing_exp_057_q4_checksum_count",
        "q4_checksum_mismatches_against_exp_057",
        "unregistered_dense_projection_count",
    )
    for key in required_zero:
        if int(measured[key]) != 0:
            raise SystemExit(f"EXP-058 authority failed: {key}={measured[key]}")
    if int(measured["dense_projection_matrix_count"]) != 144:
        raise SystemExit("unexpected dense projection population")
    if int(measured["full_integer_rank_proven_count"]) != 144:
        raise SystemExit("not every registered matrix was proven full rank")

    rank_rows = rows(RESULT / "raw/matrix_rank_rows.jsonl")
    if len(rank_rows) != 144:
        raise SystemExit("rank row count mismatch")
    if any(int(row["certified_rank_lower_bound"]) != int(row["minimum_dimension"]) for row in rank_rows):
        raise SystemExit("a frozen rank row is not full rank")

    provenance = {
        "workflow_run": WORKFLOW_RUN,
        "artifact_id": ARTIFACT_ID,
        "artifact_name": ARTIFACT_NAME,
        "artifact_size_bytes": ARTIFACT_SIZE,
        "artifact_zip_sha256": ARTIFACT_ZIP_SHA256,
        "source_head_sha": SOURCE_HEAD,
        "workflow_merge_sha": WORKFLOW_MERGE,
        "config_sha256": CONFIG_SHA256,
        "frozen_date": "2026-08-04",
    }
    (RESULT / "raw/artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["provenance"].update(provenance)
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(serialized, encoding="utf-8")
    (RESULT / "processed/aggregate.json").write_text(serialized, encoding="utf-8")
    write_checksums()

    append_once(
        "DECISION_LOG.md",
        f'''{MARKER}
## D-035/D-036 — Reject exact low-rank factorization and select shift-displacement Gate

D-035 records EXP-058 authority `results/exp_058/summary.json`; workflow `{WORKFLOW_RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{WORKFLOW_MERGE}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`. All 144 registered Q4 dense-projection matrices from the three pinned TinyStories checkpoints were proven full integer/rational rank by exact modular certificates. Prime 251 certified 143 matrices and prime 257 certified one; certificate/control mismatches and EXP-057 checksum mismatches were zero. The favorable conventional exact two-factor operation and storage lower bounds were p50/p90 200%/200% of direct dense form. Decision: `{DECISION}`.

D-036 closes ordinary exact two-factor low-rank decomposition for this measured real-Q4 population. Full matrix rank does not rule out full-rank structured transforms, so EXP-059 tests exact Toeplitz/Hankel/circulant-like structure through shift-displacement rank certificates rather than another low-rank factor search.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-026 — Conventional exact low-rank factorization of measured real Q4 projections

Every one of 144 pinned real-Q4 dense projections had certified rank equal to `min(rows, columns)`. A conventional exact `W=A@B` route therefore has favorable operation and factor-storage lower bounds of 200% for the measured shapes before factor bitwidth, metadata, and kernel overhead. Do not revisit using approximate SVD ranks, selected singular-value energy, or a different factor search while still claiming exact output preservation. Retain modular rank certificates as a falsification tool.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-058 closed — Exact algebraic rank of pinned real Q4 matrices

Authority: `results/exp_058/summary.json`; workflow `{WORKFLOW_RUN}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`.

MEASURED Phase C observation: all 144 registered dense projections were proven full integer/rational rank with zero certificate/control or Q4 checksum mismatch. The favorable exact two-factor operation and storage lower bounds were 2.0x at p50 and p90 for every model size. Decision: `{DECISION}`.

Q4 model-output preservation, constructive factor kernels, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED. Current frontier: EXP-059 exact shift-displacement rank on the same pinned Q4 matrices.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-034 — Real Q4 matrices may be full rank yet have low exact shift-displacement rank

Status: ACTIVE FOR EXP-059 ONLY.

EXP-058 disproved useful conventional exact low rank for the measured population but did not test Toeplitz-, Hankel-, or circulant-like full-rank structure. EXP-059 must compute exact displacement ranks under registered zero-fill and cyclic diagonal/anti-diagonal shifts. No visual banding, approximate spectral decay, or hand-selected tensor may substitute for certificates.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-058 validation closure

- same EXP-057 Q4 checksums: PASS, 0 mismatches;
- registered dense projections: PASS, 144/144;
- modular certificate/control consistency: PASS, 0 mismatches;
- full integer/rational rank proven: 144/144;
- exact two-factor operation Gate: FAIL, p50/p90 200%/200%;
- exact factor-storage Gate: FAIL, p50/p90 200%/200%;
- 405B/8 GiB/hardware: NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Rank-certificate boundary

The runtime may retain modular rank certification as an offline structural audit. It must not instantiate conventional exact low-rank factors for matrices whose certified rank is full. The next permitted algebraic route is a materially different full-rank structured representation, beginning with registered shift-displacement operators in EXP-059.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-058 hardware status

No GPU kernel was promoted because exact rank certificates already force a 2.0x favorable two-factor lower bound on the measured matrices. CUDA, physical factor bytes, PCIe, SSD, TTFT, tokens/second, power, and 8 GiB residency remain NOT TESTED. EXP-059 also remains a CPU structural Gate until its exact displacement-rank thresholds survive.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-058 authority

```text
workflow {WORKFLOW_RUN}
source head {SOURCE_HEAD}
workflow merge {WORKFLOW_MERGE}
artifact {ARTIFACT_ID} ({ARTIFACT_SIZE} bytes)
artifact ZIP SHA-256 {ARTIFACT_ZIP_SHA256}
config SHA-256 {CONFIG_SHA256}
```

Reproduce with `experiments/exp_058/reproduce.sh`; verify frozen files with `cd results/exp_058 && sha256sum -c checksums.sha256`.''',
    )

    next_text = '''# Next Experiment

## Closed Gate — EXP-058

Authority: `results/exp_058/summary.json`; workflow `30826618962`; source head `8ae03de4cc34317b5536aed42b9b8c22f98c88ea`; workflow merge `3730d6ce8ca89df347079c366a91bcad4d904a85`; artifact `8861905858`; ZIP SHA-256 `851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae`.

All 144 pinned real-Q4 dense projections were proven full integer/rational rank. The favorable conventional exact two-factor operation and storage lower bounds were 200% at p50 and p90. Decision:

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

## EXP-059 — Pinned Real-Q4 Exact Shift-Displacement Rank Gate

### Mechanism change

Full rank does not rule out Toeplitz-, Hankel-, or circulant-like matrices. Such matrices can have low rank after a fixed shift displacement even when `W` itself is full rank. For every registered Q4 dense projection, construct four exact integer displacement matrices:

```text
D_zero_diag   = W - Z_m @ W @ Z_n.T
D_zero_anti   = reverse_columns(W) - Z_m @ reverse_columns(W) @ Z_n.T
D_cycle_diag  = W - P_m @ W @ P_n.T
D_cycle_anti  = reverse_columns(W) - P_m @ reverse_columns(W) @ P_n.T
```

`Z` is a zero-fill down-shift and `P` a cyclic down-shift. Compute exact modular-rank certificates for every operator. The best registered operator may be selected only after all four certificate costs are recorded.

### Pinned evidence

Use the unchanged TinyStories-1M/3M/8M revisions and the exact deterministic Q4 checksums frozen by EXP-057. Analyze all 144 named dense projections; no tensor selection is allowed.

### Certificates

Use primes `251, 257, 263`. A modular displacement-rank certificate is a rigorous lower bound on integer/rational displacement rank. Retain pivot rows/columns, nonzero certified minor determinant, operator identity, rank lower bound, and input checksum.

### Favorable accounting

For displacement rank `r`, charge at least:

```text
query lower bound:   r * max(m, n) exact frequency-domain products
storage lower bound: r * (m + n) generator scalars
```

These omit transforms, boundary terms, scalar bitwidth expansion, metadata, and operator-search overhead, so they are favorable to the candidate. Compare with direct `m*n` scalar terms/storage and separately record all four certificate searches.

### Controls

- exact random Toeplitz matrices: zero-fill diagonal displacement rank <=2;
- exact random Hankel matrices: zero-fill anti-diagonal displacement rank <=2;
- exact circulant matrices: cyclic diagonal displacement rank 0;
- random dense matrices: high displacement rank;
- row/column permutation and checksum invariance;
- exact Q4 checksum agreement with EXP-057.

### Promotion Gate

```text
zero certificate/control mismatch
zero EXP-057 Q4 checksum mismatch
zero unregistered dense projection
p50 favorable query lower-bound fraction <=10%
p90 favorable query lower-bound fraction <=25%
p50 generator-storage lower-bound fraction <=10%
p90 generator-storage lower-bound fraction <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

### Claim boundary

Phase C observation only. Q4 output preservation, constructive displacement generators, exact transform kernels, actual Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
'''
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")

    document = f'''# EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

## Authority

- workflow `{WORKFLOW_RUN}`
- source head `{SOURCE_HEAD}`
- workflow merge `{WORKFLOW_MERGE}`
- artifact `{ARTIFACT_ID}` ({ARTIFACT_SIZE} bytes)
- artifact ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`
- config SHA-256 `{CONFIG_SHA256}`

## MEASURED

- 3 unchanged revision-pinned TinyStories checkpoints;
- 153 two-dimensional tensors and 144 registered dense projections;
- Q4 checksum mismatches against EXP-057: 0;
- certificate/control mismatches: 0;
- full integer/rational rank proven: 144/144;
- prime distribution: 251 for 143 matrices, 257 for one matrix;
- favorable exact two-factor p50/p90 operation lower bound: 200%/200%;
- favorable exact factor-storage p50/p90 lower bound: 200%/200%;
- peak RSS: 886,572 KiB.

## Decision

```text
{DECISION}
```

Ordinary exact low-rank factorization is rejected for the measured real-Q4 population. Rank certificates remain a reusable falsification primitive. Q4 output preservation and target hardware were not tested.
'''
    path = ROOT / "docs/research/EXPERIMENT_058_REAL_Q4_RANK.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
