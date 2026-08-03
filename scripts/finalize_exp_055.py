#!/usr/bin/env python3
"""Freeze EXP-055 authority and advance the research frontier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_055"

WORKFLOW_RUN = 30820909775
ARTIFACT_ID = 8858805996
ARTIFACT_NAME = "exp-055-candidate-30820909775"
ARTIFACT_SIZE = 53989
ARTIFACT_ZIP_SHA256 = "983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8"
SOURCE_HEAD = "c15b1bb94496ad629bf8911d30d47a7cbe792595"
WORKFLOW_MERGE = "58e83895bbc626391cb9ac70397cea14b70c84a4"
CONFIG_SHA256 = "688e176e57f1a2cabebc55d2907bc6d6198b4536015839f32001d9bf36222ff5"
DECISION = "REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY"
MARKER = "<!-- EXP-055-AUTHORITATIVE-FINAL -->"


def append_once(path: str, text: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(RESULT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(RESULT).as_posix()}")
    (RESULT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_path = RESULT / "summary.json"
    if not summary_path.exists():
        raise SystemExit("EXP-055 summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-055 decision")
    measured = summary["MEASURED"]
    if any(measured[key] != 0 for key in ("score_mismatches", "top1_mismatches", "packed_mismatches", "truth_table_representations")):
        raise SystemExit("EXP-055 exactness/representation contract failed")

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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULT / "processed/aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksums()

    append_once(
        "DECISION_LOG.md",
        f'''{MARKER}
## D-029/D-030 — Reject exact column grouping as universal core and select EXP-056

D-029 records EXP-055 authority `results/exp_055/summary.json`; workflow `{WORKFLOW_RUN}`; source head `{SOURCE_HEAD}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`. Across 48 cases and 96 compiled plans, 248,832 scalar validations plus packed controls produced zero score, top-1, or packed mismatches and no truth-table representation. Ideal repeated/sign-related columns improved monotonically to 7.8125%/9.375% logical operations at n=64. However global p50/p90 operation fractions were 62.5%/250%, p50/p90 query-byte fractions were 63.64%/200%, dense/unique p50 was 250%, and 21 cases had no positive compile amortization. Projected logical 405B-Q4 storage peaked at 0.7597 TiB and passed its isolated Gate. Decision: `{DECISION}`.

D-030 retains exact grouping as an auxiliary optimization only when real weight extraction proves repetition. EXP-056 tests automatically derived exact prototype-plus-sparse-residual dictionaries, charging prototype, membership, residual, compile, and query costs.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-023 — Exact identical/sign-related column aggregation as universal core

Exact grouping preserved every registered decision and ideal repeated/sign-related controls fell below 10% logical work at n=64. General dense and forced-unique columns did not share that structure: global p50/p90 logical work was 62.5%/250%, query bytes were 63.64%/200%, and dense/unique p50 was 250%. Do not continue by reporting only repeated synthetic columns, hiding membership/popcount/vector-add work, or assuming real Transformer columns repeat without extraction evidence. Classification: auxiliary exact optimization conditioned on measured repetition.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-055 closed — Exact column-signature popcount aggregation

Authority: `results/exp_055/summary.json`; workflow `{WORKFLOW_RUN}`; artifact `{ARTIFACT_ID}`; ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`.

MEASURED E1: 48 cases, 96 plans, 248,832 scalar validations, zero exact mismatches, zero runtime tables. Repeated/sign-related n=64 controls reached 7.8125%/9.375% logical work, proving a real exact compression fragment under strong repetition. The universal Gate failed: p50/p90 operations 62.5%/250%, p50/p90 bytes 63.64%/200%, dense/unique p50 250%, and 21 non-amortizing cases. Projected logical storage maximum 0.7597 TiB passed only the storage Gate.

Decision: `{DECISION}`. Real Transformer extraction, operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.

Current frontier: EXP-056 exact prototype-plus-sparse-residual dictionary Gate.''',
    )

    next_experiment = '''# Next Experiment

## Closed Gate — EXP-055

Authority: `results/exp_055/summary.json`; workflow `30820909775`; source head `c15b1bb94496ad629bf8911d30d47a7cbe792595`; artifact `8858805996`; ZIP SHA-256 `983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8`.

Exact identical/sign-related grouping was correct and ideal structured cases improved below 10% at n=64, but general p50/p90 operations were 62.5%/250%, query bytes 63.64%/200%, and dense/unique p50 250%. Decision:

```text
REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY
```

## EXP-056 — Exact Prototype Plus Sparse-Residual Dictionary Gate

### Mechanism change

Generalize exact repetition without changing the model. Automatically compile each exact weight column as:

```text
column_i = prototype[group_i] + exact_sparse_residual_i
score = bias + sum_g popcount(active members_g) * prototype_g
              + sum_active_i residual_i
```

Prototype selection is deterministic and weight-derived. Every nonzero residual scalar, index, membership mask, prototype read, popcount, multiply/add, compile search, and fallback is charged. No approximation, training, target adapter, or runtime state table is allowed.

### Conditions

```text
G0 independent signed modular top-1 reference
G1 deterministic exact prototype construction
G2 exact sparse residual reconstruction
G3 scalar and packed evaluator
G4 repeated/sign-related/sparse/low-rank controls
G5 dense-random and forced-unique adversaries
G6 exhaustive small-domain and deterministic larger-domain validation
```

### Registered search

Test prototype counts 1/2/4/8 and deterministic medoid/most-frequent candidates. Select only by fully accounted operation then byte cost; charge every attempted compilation. Residuals remain exact signed integers.

### Early rejection Gate

```text
exact mismatch >0
runtime state table used
p50 operations >10% or p90 >25%
p50 bytes >10% or p90 >25%
dense-random/unique p50 >25%
projected storage >1 TiB
compile amortization >1,000,000 queries
savings degrade with input/classes
```

Failure decision:

```text
REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY
```

### Evidence boundary

Phase A/B, E1. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.

### Next exact action

1. implement deterministic exact prototype and sparse-residual compiler;
2. add independent reconstruction/evaluation validators;
3. execute the registered structured and adversarial matrix;
4. freeze all accounting, binaries, checksums, and decision;
5. promote to real checkpoint extraction only if the universal synthetic Gate survives.
'''
    (ROOT / "NEXT_EXPERIMENT.md").write_text(next_experiment, encoding="utf-8")

    experiment_doc = f'''# EXP-055 — Exact Column-Signature Popcount Aggregation Gate

## Question

Can identical or exact-negated multi-class weight columns be grouped into activation popcounts while preserving signed modular top-1 decisions and meeting the universal runtime budget?

## Authority

- workflow `{WORKFLOW_RUN}`
- source head `{SOURCE_HEAD}`
- workflow merge `{WORKFLOW_MERGE}`
- artifact `{ARTIFACT_ID}` (`{ARTIFACT_SIZE}` bytes)
- artifact ZIP SHA-256 `{ARTIFACT_ZIP_SHA256}`
- config SHA-256 `{CONFIG_SHA256}`

## MEASURED

- 48 cases, 96 plans, 6 families;
- 248,832 scalar validations; 150,528 exhaustive and 98,304 deterministic sampled;
- score/top-1/packed mismatches: 0/0/0;
- runtime truth-table representations: 0;
- p50/p90 operations: 62.5%/250%;
- p50/p90 query bytes: 63.64%/200%;
- dense/forced-unique p50 operations: 250%;
- repeated n=64: 7.8125%; sign-related n=64: 9.375%;
- maximum projected logical 405B-Q4 storage: 0.7597 TiB;
- 21 cases had infinite compile amortization because runtime work did not beat baseline.

## Decision

```text
{DECISION}
```

Exact repetition is a genuine auxiliary optimization, but it is not a universal core for arbitrary dense weights. Real Transformer extraction and hardware execution were not tested.
'''
    doc = ROOT / "docs/research/EXPERIMENT_055_COLUMN_SIGNATURE_POPCOUNT.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(experiment_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
