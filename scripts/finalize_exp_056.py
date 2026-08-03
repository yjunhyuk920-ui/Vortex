#!/usr/bin/env python3
"""Freeze EXP-056 authority and register the real-weight extraction Gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_056"

WORKFLOW_RUN = 30823042599
ARTIFACT_ID = 8859665874
ARTIFACT_NAME = "exp-056-candidate-30823042599"
ARTIFACT_SIZE = 245256
ARTIFACT_ZIP_SHA256 = "9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8"
SOURCE_HEAD = "73655fc216340d9bd1d452d779951c28ac1b3d3b"
WORKFLOW_MERGE = "df19bf0dee5e7f42a10378d5bca70d5513697982"
CONFIG_SHA256 = "f75819b0cc6a741fac464d0e2adec2cb9b83612e7a17eed4b95a0cec5c03f151"
DECISION = "REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY"
MARKER = "<!-- EXP-056-AUTHORITATIVE-FINAL -->"


def append_once(path: str, text: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


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
    if not summary_path.exists():
        raise SystemExit("EXP-056 summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-056 decision")
    measured = summary["MEASURED"]
    if any(
        measured[key] != 0
        for key in (
            "score_mismatches",
            "top1_mismatches",
            "packed_mismatches",
            "truth_table_representations",
        )
    ):
        raise SystemExit("EXP-056 exactness/representation contract failed")

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
    (RESULT / "raw").mkdir(parents=True, exist_ok=True)
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
## D-031/D-032 — Reject prototype-residual dictionaries as universal core and select real-weight extraction

D-031 records EXP-056 authority `results/exp_056/summary.json`; workflow `{WORKFLOW_RUN}`; source head `{SOURCE_HEAD}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`. 56 cases and 448 exact plans produced 1,161,216 scalar validations with zero score, top-1, or packed mismatch and no runtime table. Repeated columns reached 7.8125% logical work at n=64, sparse prototype perturbations reached 10.9375%, and sign clusters reached 15.625%. The universal Gate failed: p50/p90 operations 62.5%/131.25%, p50/p90 bytes 62.115%/169.643%, dense/unique p50 123.4375%, and 24 cases did not amortize. Projected logical 405B-Q4 storage peaked at 0.6791 TiB and passed only its isolated Gate. Decision: `{DECISION}`.

D-032 stops synthetic dictionary elaboration until real pinned checkpoint matrices are measured. EXP-057 extracts exact FP bit patterns and deterministic Q8/Q4 weight columns from unchanged pinned TinyStories checkpoints and applies the retained EXP-055/056 analyzers with full per-matrix accounting.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-024 — Exact prototype plus sparse-residual dictionaries as universal core

The compiler exactly reconstructed all registered columns, and favorable repeated/sparsely perturbed controls improved with width. General dense and unique columns retained too many residuals: p50/p90 logical work was 62.5%/131.25%, query bytes 62.115%/169.643%, dense/unique p50 123.4375%, and 24 cases never beat baseline. Do not continue by adding synthetic prototype counts, hiding residual activation/index costs, or presenting favorable repeated matrices as arbitrary-model evidence. Further use requires measured real-checkpoint structure.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-056 closed — Exact prototype plus sparse-residual dictionaries

Authority: `results/exp_056/summary.json`; workflow `{WORKFLOW_RUN}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`.

MEASURED E1: 56 cases, 448 plans, 1,161,216 scalar validations, zero exact mismatches, zero runtime tables. Repeated n=64 reached 7.8125%, exact sparse prototype perturbations 10.9375%, and sign clusters 15.625%. General p50/p90 work was 62.5%/131.25%; bytes 62.115%/169.643%; dense/unique p50 123.4375%; 24 cases did not amortize. Projected logical storage maximum 0.6791 TiB passed only storage.

Decision: `{DECISION}`. Current frontier is EXP-057 pinned real-checkpoint weight-structure extraction. 405B, 8 GiB, actual operation replacement, and target hardware remain NOT TESTED.''',
    )

    next_text = '''# Next Experiment

## Closed Gate — EXP-056

Authority: `results/exp_056/summary.json`; workflow `30823042599`; source head `73655fc216340d9bd1d452d779951c28ac1b3d3b`; artifact `8859665874`; ZIP SHA-256 `9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8`.

Exact prototype-residual plans were correct, but p50/p90 logical work was 62.5%/131.25%, p50/p90 bytes 62.115%/169.643%, and dense/unique p50 123.4375%. Decision:

```text
REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY
```

## EXP-057 — Pinned Real-Checkpoint Weight-Structure Extraction Gate

### Why this changes the evidence class

EXP-055 and EXP-056 found exact savings only when weight columns truly repeat or differ by very sparse exact residuals. Continuing with invented matrices would not answer whether public Transformer weights contain that structure. EXP-057 therefore moves from synthetic construction to Phase C observation on unchanged pinned checkpoints.

### Pinned models

```text
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

### Matrix scope

Enumerate every 2-D learned weight tensor used by Transformer linear/embedding projections. Record model, module path, shape, dtype, parameter count, and checksum. Biases and 1-D normalization weights are excluded from column-structure claims but remain in the manifest.

### Representations

1. exact stored floating-point bit patterns;
2. deterministic symmetric per-output-row Q8;
3. deterministic symmetric per-output-row Q4.

Quantization is an execution representation only; checkpoints remain unchanged. Scale, zero handling, clipping, packing, and dequantization error are recorded separately. No quality or output-preservation claim is made by this structural Gate.

### Analyses

For each matrix and representation:

- exact identical and sign-canonical column groups;
- exact group coverage and largest group;
- EXP-055 logical operation/byte fraction;
- EXP-056 frequency/greedy prototype counts 1/2/4/8;
- exact residual scalar/column density;
- best fully accounted logical operation/byte/storage fraction;
- compile search and amortization;
- layer/type/model-size trends.

### Controls

- shuffled-column order control, which must preserve structure counts;
- element-permuted adversary, which should destroy column structure;
- synthetic repeated and sparse-residual positive controls;
- exact reconstruction checks for every compiled plan;
- checksum-pinned model and tensor manifests.

### Early Gate

Promotion to an actual small-model operation-replacement kernel requires all of:

```text
zero reconstruction mismatch
zero unregistered tensors
real-matrix p50 operations <=10%, p90 <=25%
real-matrix p50 bytes <=10%, p90 <=25%
no model-size degradation beyond 25%
projected storage <=1 TiB
compile amortization <=1,000,000 queries
```

Failure decision:

```text
REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY
```

### Claim boundary

Phase C observation at most. It does not execute a replacement Transformer operation and does not test 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, or tokens/sec.
'''
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_text, encoding="utf-8")

    document = f'''# EXP-056 — Exact Prototype Plus Sparse-Residual Dictionary Gate

## Authority

- workflow `{WORKFLOW_RUN}`
- source head `{SOURCE_HEAD}`
- workflow merge `{WORKFLOW_MERGE}`
- artifact `{ARTIFACT_ID}` ({ARTIFACT_SIZE} bytes)
- artifact ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`
- config SHA-256 `{CONFIG_SHA256}`

## MEASURED

- 56 cases, 448 plans, 7 families;
- 1,161,216 scalar validations; 702,464 exhaustive and 458,752 sampled;
- score/top-1/packed mismatches: 0/0/0;
- runtime truth tables: 0;
- p50/p90 operations: 62.5%/131.25%;
- p50/p90 bytes: 62.115%/169.643%;
- dense/unique p50: 123.4375%;
- repeated n=64: 7.8125%; sparse prototype residual n=64: 10.9375%; sign clusters n=64: 15.625%;
- maximum projected logical 405B-Q4 storage: 0.6791 TiB;
- 24 cases had no positive amortization.

## Decision

```text
{DECISION}
```

Exact prototype dictionaries remain auxiliary. The next evidence must measure real pinned checkpoint matrices before more synthetic representation work.
'''
    path = ROOT / "docs/research/EXPERIMENT_056_PROTOTYPE_SPARSE_RESIDUAL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
