#!/usr/bin/env python3
"""Freeze EXP-057 real-checkpoint authority and register EXP-058."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_057"

WORKFLOW_RUN = 30824957941
ARTIFACT_ID = 8860450501
ARTIFACT_NAME = "exp-057-candidate-30824957941"
ARTIFACT_SIZE = 197667
ARTIFACT_ZIP_SHA256 = "7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0"
SOURCE_HEAD = "cf9d7099dc11b22ce24ba6e096712d5da1bc3729"
WORKFLOW_MERGE = "0c70c5547a68ce3db4a584ac32fb0cbf9873d861"
CONFIG_SHA256 = "e99e13b3c912f1567d010c1c60fa0c8ade0b2350bd8ce6cacc49e244c4df334e"
DECISION = "REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY"
MARKER = "<!-- EXP-057-AUTHORITATIVE-FINAL -->"


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


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def write_checksums() -> None:
    lines: list[str] = []
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
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-057 decision")
    measured = summary["MEASURED"]
    if measured["reconstruction_mismatches"] != 0:
        raise SystemExit("real quantized representation reconstruction failed")
    if measured["unregistered_2d_tensor_count"] != 0:
        raise SystemExit("some real 2-D tensors were not registered")
    if measured["control_failures"] != 0:
        raise SystemExit("registered controls failed")

    representation_rows = rows(RESULT / "raw/representation_rows.jsonl")
    dense_by_representation: dict[str, list[dict]] = {}
    for representation in (
        "fp32_exact_bits",
        "q8_row_symmetric",
        "q4_row_symmetric",
    ):
        dense_by_representation[representation] = [
            row
            for row in representation_rows
            if row["representation"] == representation
            and row["matrix_role"] == "dense_projection"
        ]
    repeated_counts: dict[str, int] = {}
    maximum_repeated_coverage: dict[str, float] = {}
    for representation, items in dense_by_representation.items():
        repeated_counts[representation] = sum(
            max(
                float(row["grouping"]["identical"]["repeated_column_coverage_fraction"]),
                float(row["grouping"]["sign_canonical"]["repeated_column_coverage_fraction"]),
            )
            > 0.0
            for row in items
        )
        maximum_repeated_coverage[representation] = max(
            max(
                float(row["grouping"]["identical"]["repeated_column_coverage_fraction"]),
                float(row["grouping"]["sign_canonical"]["repeated_column_coverage_fraction"]),
            )
            for row in items
        )
    q4 = dense_by_representation["q4_row_symmetric"]
    minimum = min(q4, key=lambda row: float(row["selected_operation_fraction"]))
    residuals = [float(row["selected_residual_scalar_fraction"]) for row in q4]
    mechanisms = Counter(str(row["selected_mechanism"]) for row in q4)
    q4_quantized = [
        row
        for row in representation_rows
        if row["representation"] == "q4_row_symmetric"
    ]
    q8_quantized = [
        row
        for row in representation_rows
        if row["representation"] == "q8_row_symmetric"
    ]
    extra = {
        "dense_projection_matrices_with_exact_repeated_or_sign_columns": repeated_counts,
        "maximum_dense_projection_exact_repeated_coverage_fraction": maximum_repeated_coverage,
        "q4_dense_projection_minimum_operation_fraction": float(
            minimum["selected_operation_fraction"]
        ),
        "q4_dense_projection_minimum_operation_tensor": minimum["tensor_name"],
        "q4_dense_projection_minimum_operation_model": minimum["model_id"],
        "q4_dense_projection_median_residual_scalar_fraction": statistics.median(residuals),
        "q4_dense_projection_p90_residual_scalar_fraction": percentile(residuals, 0.9),
        "q4_selected_mechanism_distribution": dict(sorted(mechanisms.items())),
        "q8_median_mean_absolute_quantization_error": statistics.median(
            float(row["quantization"]["mean_absolute_error"])
            for row in q8_quantized
        ),
        "q4_median_mean_absolute_quantization_error": statistics.median(
            float(row["quantization"]["mean_absolute_error"])
            for row in q4_quantized
        ),
        "q8_maximum_absolute_quantization_error": max(
            float(row["quantization"]["maximum_absolute_error"])
            for row in q8_quantized
        ),
        "q4_maximum_absolute_quantization_error": max(
            float(row["quantization"]["maximum_absolute_error"])
            for row in q4_quantized
        ),
        "quantization_clipped_value_count": sum(
            int(row["quantization"]["clipped_value_count"])
            for row in q8_quantized + q4_quantized
        ),
    }
    measured.update(extra)

    provenance = {
        "workflow_run": WORKFLOW_RUN,
        "artifact_id": ARTIFACT_ID,
        "artifact_name": ARTIFACT_NAME,
        "artifact_size_bytes": ARTIFACT_SIZE,
        "artifact_zip_sha256": ARTIFACT_ZIP_SHA256,
        "source_head_sha": SOURCE_HEAD,
        "workflow_merge_sha": WORKFLOW_MERGE,
        "config_sha256": CONFIG_SHA256,
        "frozen_date": "2026-08-03",
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
## D-033/D-034 — Reject exact real-weight grouping/dictionaries and select algebraic-rank Gate

D-033 records EXP-057 authority `results/exp_057/summary.json`; workflow `{WORKFLOW_RUN}`; source head `{SOURCE_HEAD}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`. Three unchanged pinned TinyStories checkpoints exposed 327 learned tensors, including 153 analyzed 2-D tensors and 54,205,312 named 2-D scalars. Across all 144 named dense-projection matrices, exact repeated or sign-related column coverage was zero in loaded FP32, deterministic Q8, and deterministic Q4. Q4 retained p50/p90 logical operations of 82.8918%/85.8398%, p50/p90 query bytes of 329.0244%/490.6845%, and median/p90 exact residual density of 81.4087%/84.2834%. Even the best real matrix retained 70.2866% operations. Storage projection passed narrowly at 0.9300 TiB and compile amortization passed at 377 queries. Decision: `{DECISION}`.

D-034 closes exact column repetition and sparse residual dictionaries as a universal direction for the measured real checkpoints. EXP-058 tests a different exact representation: algebraic low-rank factorization. Modular-rank certificates on the same pinned Q4 matrices establish exact rank lower bounds before any factorization is promoted.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-025 — Exact column grouping/dictionaries on measured real checkpoint weights

No analyzed dense projection in three pinned TinyStories checkpoints contained even one exactly repeated or sign-related column under FP32, Q8, or Q4. Prototype residuals remained dense: Q4 median/p90 residual scalar density was 81.41%/84.28%; p50/p90 operations were 82.89%/85.84%; query bytes were 3.29x/4.91x baseline. Do not continue by increasing prototype search, quoting only the 70.29% best matrix, or treating Q4 structural results as model-output preservation. Retain the analyzers only for conditional measurement on future models.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-057 closed — Pinned real-checkpoint exact weight structure

Authority: `results/exp_057/summary.json`; workflow `{WORKFLOW_RUN}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`.

MEASURED Phase C observation: 3 pinned unchanged models, 327 tensors, 153 two-dimensional tensors, 54,205,312 named 2-D scalars, and zero unregistered matrices. All 144 dense projections had zero exact repeated/sign-related columns in FP32, Q8, and Q4. Q4 p50/p90 operations were 82.8918%/85.8398%; bytes 329.0244%/490.6845%; median residual density 81.4087%; best matrix 70.2866%. Reconstruction and controls passed; projected storage was 0.9300 TiB.

Decision: `{DECISION}`. Q4 output preservation, actual operation replacement, 405B, 8 GiB, and hardware remain NOT TESTED. Current frontier: EXP-058 pinned real-Q4 exact algebraic-rank certificates.''',
    )

    next_text = '''# Next Experiment

## Closed Gate — EXP-057

Authority: `results/exp_057/summary.json`; workflow `30824957941`; source head `cf9d7099dc11b22ce24ba6e096712d5da1bc3729`; artifact `8860450501`; ZIP SHA-256 `7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0`.

All 144 real dense projections had zero exact repeated/sign-related columns in FP32, Q8, and Q4. Q4 p50/p90 operations were 82.8918%/85.8398% and query bytes 329.0244%/490.6845%. Decision:

```text
REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY
```

## EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

### Mechanism change

Test whether deterministic row-symmetric Q4 projection matrices admit an exact low-rank factorization:

```text
W = A @ B
W x = A @ (B x)
```

No approximation or truncated SVD is allowed. Instead of assuming rank, compute exact modular-rank certificates. A full-rank minor modulo any registered prime proves the integer/rational rank is full and rules out a lower exact factorization rank.

### Pinned evidence

Use the same unchanged TinyStories-1M/3M/8M revisions and the exact EXP-057 Q4 rule. Analyze every named dense-projection matrix. Embeddings/output heads are reported separately.

### Registered primes

```text
251, 257, 263
```

Stop after the first full-rank certificate; test all primes only when a matrix remains deficient. Record pivot rows/columns, certificate prime, rank lower bound, minimum dimension, and checksums.

### Fully accounted lower bounds

For certified rank `r`, any conventional exact two-factor path must perform at least:

```text
r*n + m*r scalar multiply/add terms
```

and store at least `r*(m+n)` factor scalars before metadata. Compare this with the direct `m*n` matrix path. Calculate the maximum rank that could meet 10% and 25% operation budgets, and determine whether the certified lower bound already exceeds them.

### Controls

- known exact low-rank products with registered ranks;
- full-rank identity and random integer controls;
- row/column permutation rank invariance;
- duplicate-row rank-deficient control;
- deterministic Q4 checksum agreement with EXP-057 rules.

### Promotion Gate

```text
zero certificate/control mismatch
zero unregistered dense projections
real-matrix p50 exact-factor operation lower bound <=10%
real-matrix p90 exact-factor operation lower bound <=25%
real-matrix p50 factor-storage lower bound <=10%
real-matrix p90 factor-storage lower bound <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

### Claim boundary

Phase C observation only. Q4 output preservation, factor-kernel execution, actual Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
'''
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")

    document = f'''# EXP-057 — Pinned Real-Checkpoint Weight-Structure Extraction Gate

## Authority

- workflow `{WORKFLOW_RUN}`
- source head `{SOURCE_HEAD}`
- workflow merge `{WORKFLOW_MERGE}`
- artifact `{ARTIFACT_ID}` ({ARTIFACT_SIZE} bytes)
- artifact ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`
- config SHA-256 `{CONFIG_SHA256}`

## MEASURED

- 3 unchanged revision-pinned TinyStories checkpoints;
- 327 learned tensors, 153 analyzed 2-D tensors, 54,205,312 named 2-D scalars;
- 144 dense-projection matrices in the primary Q4 Gate;
- zero unregistered 2-D tensors and zero reconstruction/control failures;
- exact repeated/sign-related dense matrices: 0 in FP32, Q8, and Q4;
- Q4 p50/p90 operations: 82.8918%/85.8398%;
- Q4 p50/p90 query bytes: 329.0244%/490.6845%;
- Q4 median/p90 exact residual density: 81.4087%/84.2834%;
- best real dense matrix: 70.2866% operations;
- maximum projected logical 405B-Q4 storage: 0.9300 TiB;
- maximum compile amortization: 377 queries.

## Decision

```text
{DECISION}
```

The measured models do not contain the exact repeated or sparse-residual column structure required by EXP-055/056. Q8/Q4 model-output preservation and operation replacement were not tested.
'''
    path = ROOT / "docs/research/EXPERIMENT_057_REAL_WEIGHT_STRUCTURE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
