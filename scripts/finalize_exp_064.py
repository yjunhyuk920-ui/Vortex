#!/usr/bin/env python3
"""Freeze EXP-064 authority and register EXP-065."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_064"
RUN = 30869720552
ARTIFACT = 8877450455
ARTIFACT_NAME = "exp-064-candidate-30869720552"
ARTIFACT_SIZE = 102883
ZIP_SHA = "99c634bd4fb3903d32a1ed45fada7853ea4e1d199b375c129d1d4b8da4f39cb8"
SOURCE_HEAD = "a6371c39d85dc39669b98eac6125d9c3bbf4a5dc"
MERGE_SHA = "3716584078a91ae307b11b4bf1b2662e1511e9c9"
CONFIG_SHA = "d80c0eb37968f6cfecfbfe781aef406b30b536be873b052c69734aa9add68343"
DECISION = "REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY"
MARKER = "<!-- EXP-064-AUTHORITATIVE-FINAL -->"


def append_once(path: str, body: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def checksums() -> None:
    lines: list[str] = []
    for path in sorted(RESULT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(RESULT).as_posix()}")
    (RESULT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_path = RESULT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    measured = summary["MEASURED"]
    expected = {
        "two_dimensional_tensor_count": 153,
        "dense_projection_count": 144,
        "plan_row_count": 1683,
        "checksum_mismatches": 0,
        "missing_checksums": 0,
        "reconstruction_mismatches": 0,
        "control_failures": 0,
    }
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-064 decision")
    for key, value in expected.items():
        if int(measured[key]) != value:
            raise SystemExit(f"unexpected {key}")
    matrix_rows = [
        json.loads(line)
        for line in (RESULT / "raw/matrix_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan_rows = [
        json.loads(line)
        for line in (RESULT / "raw/plan_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(matrix_rows) != 153 or len(plan_rows) != 1683:
        raise SystemExit("raw row population mismatch")
    if any(not row["checksum_match"] for row in matrix_rows):
        raise SystemExit("Q4 checksum mismatch")
    if any(int(row["reconstruction_mismatches"]) for row in matrix_rows):
        raise SystemExit("row reconstruction mismatch")

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
    checksums()

    append_once("DECISION_LOG.md", f'''{MARKER}
## D-047/D-048 — Reject exact Q4 output-row reuse and open Kronecker-rank Gate

EXP-064 authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. All 153 two-dimensional tensors and 144 dense projections matched frozen Q4 checksums; all 1,683 plans reconstructed exactly. No dense matrix contained identical or sign-related rows. The deployable selector retained dense execution for 140/144 projections and exact sparse-delta plans for four. Dense-projection p50/p90 operation and query-byte fractions were all 100%; the best single matrix reached 70.522% operations and 93.811% bytes. Decision: `{DECISION}`.

D-048 opens EXP-065: exact Kronecker-rearrangement rank certificates on the same pinned real-Q4 matrices.''')
    append_once("FAILED_APPROACHES.md", f'''{MARKER}
## F-032 — Exact output-row identity, sign reuse, and sparse-delta prototypes

No identical or sign-related dense output rows occurred. Only four projections admitted a dual-cost-beneficial sparse-delta plan, while population p50/p90 remained dense. Do not revive by ignoring per-row scales/biases, activation reads, residual indexes, or by selecting a plan that saves operations while increasing bytes. Retain the row compiler as an auxiliary exact dictionary tool.''')
    append_once("RESEARCH_STATE.md", f'''{MARKER}
## EXP-064 closed — Real-Q4 exact output-row structure

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. 153 tensors, 144 dense projections, 1,683 plans, zero checksum/reconstruction/control mismatch. Identical/sign-related dense matrices: 0/0. Selected: dense 140, sparse-delta 4. p50/p90 operations and bytes: 100%/100%. Decision: `{DECISION}`. Current frontier: EXP-065 exact Kronecker-rearrangement rank. 405B, 8 GiB, Q4 model-output preservation and hardware remain NOT TESTED.''')
    append_once("ASSUMPTION_REGISTER.md", f'''{MARKER}
## A-040 — Real Q4 matrices may have low exact Kronecker rank

Status: ACTIVE FOR EXP-065 ONLY. For every nontrivial shape factorization, rearrange the Q4 matrix so the rank equals the minimum number of Kronecker-product terms over the certified field. Rank certificates may reject a candidate but cannot promote it without an exact integer reconstruction and full operation/storage accounting.''')
    append_once("VALIDATION_MATRIX.md", f'''{MARKER}
## EXP-064 closure

153 tensors; 144 dense; 1,683 plans; zero checksum/reconstruction/control mismatch; exact identical/sign row matrices 0/0; p50/p90 operation 100%/100%; query bytes 100%/100%; projected static storage 211.31 GB. 405B execution and hardware NOT TESTED.''')
    append_once("ARCHITECTURE.md", f'''{MARKER}
## Output-row structure boundary

The core runtime must not assume row identity or sparse prototype deltas for generic dense checkpoints. The exact row compiler remains fail-closed auxiliary. EXP-065 may only promote a Kronecker path after certified low rearrangement rank and exact integer reconstruction.''')
    append_once("HARDWARE_VALIDATION_PLAN.md", f'''{MARKER}
## EXP-064 hardware status

No output-row kernel was promoted. Q4 output preservation, CUDA implementation, physical memory traffic, PCIe, SSD, TTFT, tokens/sec, 405B execution and 8 GiB residency remain NOT TESTED.''')
    append_once("REPRODUCIBILITY.md", f'''{MARKER}
## EXP-064 authority

Workflow `{RUN}`; source `{SOURCE_HEAD}`; merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_064/reproduce.sh` and verify `results/exp_064/checksums.sha256`.''')

    (ROOT / "NEXT_EXPERIMENT.md").write_text('''# Next Experiment

## Closed Gate — EXP-064

Exact identical/sign output rows were absent in all 144 dense projections. Four sparse-delta plans survived local dual-cost selection, but population p50/p90 operations and query bytes remained 100%.

```text
REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY
```

## EXP-065 — Pinned Real-Q4 Exact Kronecker-Rearrangement Rank Gate

### Mechanism

For each Q4 dense matrix `W` and every nontrivial factorization `m=m1*m2`, `n=n1*n2`, form the standard Kronecker rearrangement `R(W)` with shape `(m1*n1, m2*n2)`. A rank-`r` rearrangement is necessary for an exact sum of `r` Kronecker products:

```text
W = sum_i A_i tensor B_i
```

Each certified lower bound on `rank(R(W))` yields a lower bound on factor storage and on exact reshape-multiply execution. All row/column factor-order variants are tested. Modular determinant/rank witnesses must be independently verified.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and the frozen EXP-057 Q4 checksums. Analyze all 153 two-dimensional tensors and report promotion statistics on all 144 dense projections.

### Accounting

For every factorization and certified rank lower bound, charge at least:

- `r * (m1*n1 + m2*n2)` factor scalars;
- factor metadata and permutations;
- `r` applications of `B_i X A_i^T`, including all activation/intermediate reads and writes;
- per-row Q4 scales and biases;
- compilation/certificate work and exact reconstruction data.

A low modular rank is not sufficient for promotion: any surviving candidate must reconstruct every Q4 integer exactly over the execution representation.

### Controls

- exact rank-1 and rank-2 Kronecker sums certify correctly;
- one-nibble mutation raises the appropriate rearrangement rank;
- dense-random and forced-unique matrices have high certified rank;
- witnesses verify under at least two primes;
- reshape/order round trips are exact;
- no activation lookup table or approximation is used.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 lower-bound operation fraction <=10%
p90 lower-bound operation fraction <=25%
p50 lower-bound storage fraction <=10%
p90 lower-bound storage fraction <=25%
best dense-random adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
exact integer reconstruction for every promoted candidate
```

Failure decision:

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

### Claim boundary

Phase C weight observation and exact rank certification only. A physical Kronecker kernel, Q4 model-output preservation, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
''', encoding="utf-8")

    document = ROOT / "docs/research/EXPERIMENT_064_OUTPUT_ROW_STRUCTURE.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(f'''# EXP-064 — Pinned Real-Q4 Exact Output-Row Prototype Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 1,683 plans; zero checksum, reconstruction, or control mismatch. No dense matrix had identical or sign-related rows. Dense fallback was selected for 140 projections and an exact sparse-delta plan for four. Population p50/p90 operations and query bytes were 100%/100%. Best single projection: 70.522% operations, 93.811% bytes. Projected static storage: 211.31 GB.

Decision:

```text
{DECISION}
```

Q4 model-output preservation, physical kernels, 405B, 8 GiB and target hardware were not tested.
''', encoding="utf-8")


if __name__ == "__main__":
    main()
