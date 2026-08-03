#!/usr/bin/env python3
"""Freeze EXP-060 authority and register EXP-061."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_060"
RUN = 30841671707
ARTIFACT = 8867145590
ARTIFACT_NAME = "exp-060-candidate-30841671707"
ARTIFACT_SIZE = 58039
ZIP_SHA = "5e5255dbedd779b734876faa027cd2bf5e4a1b00ece7f28cbf35f428fb9a0b05"
SOURCE_HEAD = "bf89d087343a4790202126c34562ca0344ebe452"
MERGE_SHA = "5f2af394180beaf3e5b5b8c7386d2becdf7eb8e7"
CONFIG_SHA = "82254fd1177bcce6b788199ed92bbc122d97f04783f0bc02d056c090ba043a29"
DECISION = "REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY"
MARKER = "<!-- EXP-060-AUTHORITATIVE-FINAL -->"


def append_once(path: str, body: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        raise SystemExit("unexpected EXP-060 decision")
    for key in (
        "reconstruction_or_control_mismatches",
        "missing_exp_057_q4_checksum_count",
        "q4_checksum_mismatches_against_exp_057",
        "unregistered_dense_projection_count",
    ):
        if int(measured[key]) != 0:
            raise SystemExit(f"authority failure: {key}={measured[key]}")
    if int(measured["dense_projection_matrix_count"]) != 144:
        raise SystemExit("unexpected dense projection count")
    if int(measured["format_row_count"]) != 1224:
        raise SystemExit("unexpected format row count")
    primary = [
        row
        for row in read_rows(RESULT / "raw/matrix_rows.jsonl")
        if row["matrix_role"] == "dense_projection"
    ]
    if len(primary) != 144:
        raise SystemExit("primary matrix row population mismatch")
    if any(row["selected_format"] != "row_runs" for row in primary):
        raise SystemExit("unexpected selected sparse format population")

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
## D-039/D-040 — Reject exact Q4 zero-sparsity streaming and select activation-sparsity Gate

EXP-060 authority: `results/exp_060/summary.json`; workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}`; ZIP SHA-256 `{ZIP_SHA}`. Across 144 pinned real-Q4 dense projections, exact zero-scalar fraction was p50 17.7612%, p90 20.3674%, and maximum 30.1041%. Favorable row-run selection left p50/p90 operation fractions 82.2205%/85.0586%, while indexes and run metadata raised query bytes to 150.9277%/200.8606%. The best matrix still required 69.896% operations and 190.118% bytes. Reconstruction, controls, registration, and EXP-057 checksums passed. Decision: `{DECISION}`.

D-040 closes static exact-zero weight streaming for this measured Q4 population. EXP-061 moves to runtime state and measures exact zeros at inputs to every dense projection during causal prefill and decode.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-028 — Static exact-zero sparse streaming on measured real Q4 projections

Real Q4 weights contained only 17.76% median exact zeros. Skipping them left 82.22% median work and required 150.93% median query bytes after exact run metadata. Even the best matrix remained at 69.90% work and 190.12% bytes. Do not revisit using more CSR/BSR block sizes, zero clustering, or index compression: scalar nonzero density itself is already above the 25% Gate. Retain exact sparse formats only as conditional auxiliaries.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-060 closed — Exact Q4 zero-sparsity streaming

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. The 144 real-Q4 dense projections had p50/p90 exact zero fractions 17.76%/20.37%. Exact sparse execution retained p50/p90 82.22%/85.06% operations and 150.93%/200.86% query bytes. Decision: `{DECISION}`. Q4 output preservation, physical sparse kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-061 causal exact activation sparsity.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-036 — Causal dense-projection inputs may contain useful exact zeros

Status: ACTIVE FOR EXP-061 ONLY. Static weight zeros failed, but runtime activations could skip complete weight columns. EXP-061 must measure exact IEEE zero at every registered dense-projection input, separate prefill from warm decode, exclude causal-mask zeros already handled by standard attention, preserve held-out prompt families, and charge activation-index metadata. Near-zero thresholds are approximation and are forbidden.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-060 closure

Q4 checksum agreement PASS; dense registration 144/144; formats 1224; reconstruction/control mismatches 0; exact zero fraction p50/p90 17.76%/20.37%; operation fraction 82.22%/85.06%; query-byte fraction 150.93%/200.86%. Physical sparse kernels, 405B, 8 GiB, and target hardware remain NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Static-zero sparsity boundary

The runtime must not promote static CSR/run/BSR streaming for the measured Q4 population. Exact sparse formats remain conditional auxiliaries. The next permitted sparsity route is causal activation-column skipping based only on exact runtime zeros and fail-closed dense fallback.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-060 hardware status

No sparse GPU kernel was promoted because logical work remained above 69% even for the best matrix and metadata exceeded dense Q4 bytes. CUDA sparse kernels, PCIe, SSD, TTFT, tokens/sec, power, and 8 GiB residency remain NOT TESTED.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-060 authority

Workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_060/reproduce.sh` and verify `results/exp_060/checksums.sha256`.''',
    )

    (ROOT / "NEXT_EXPERIMENT.md").write_text(
        '''# Next Experiment

## Closed Gate — EXP-060

Pinned real-Q4 dense projections contained p50 17.76% exact zeros. Exact row-run streaming retained p50/p90 82.22%/85.06% operations and required 150.93%/200.86% query bytes.

```text
REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY
```

## EXP-061 — Pinned Causal Exact Activation-Sparsity Gate

### Mechanism

For every causal forward pass, an exact-zero input coordinate to a dense projection allows the corresponding weight column to be skipped for every output row. Measure exact IEEE zeros at the input of every registered `torch.nn.Linear`/equivalent learned 2-D projection during:

```text
prompt prefill
first decode token
decode tokens 2..64
```

Causal-attention mask zeros and padding positions are excluded; they are already standard structural sparsity. Only actual projection-input scalar values equal to positive or negative zero count.

### Pinned models and prompts

Use unchanged TinyStories-1M/3M/8M revisions and the pinned GPT-Neo tokenizer from EXP-050. Use the six held-out families: English narrative, Korean, code, mathematics, structured JSON, and identifier boundary. Generate 64 greedy tokens with KV cache for each model/prompt pair.

### Registration

- enumerate every learned 2-D projection module before execution;
- record module name, weight shape/checksum, input feature width, calls, tokens, and phase;
- fail on unhooked or shape-mismatched dense projections;
- deduplicate tied modules only by object identity while preserving named aliases;
- do not count embeddings or causal attention masks as projection-input sparsity.

### Accounting

For input width `n`, output width `m`, and `z` exact-zero input coordinates:

```text
dense operations = m*n
sparse operations = m*(n-z)
weight bytes = Q4 columns for n-z coordinates
activation metadata = nonzero-coordinate indexes + vector row pointer
```

Report operation and query-byte fractions per call, weighted by original dense scalar terms. Charge scanning every activation coordinate to discover zeros as a separate runtime operation count. Selection by prompt, model, module, or token is forbidden; aggregate the full registered population.

### Controls

- injected all-zero vector: zero operation fraction and exact dense fallback equivalence;
- ReLU negative control input: registered exact zeros detected;
- GELU random input: no false zero creation;
- positive-zero and negative-zero counted identically;
- column-skipped mathematical reference equals dense reference for exact-zero coordinates;
- hook registration and call accounting are deterministic;
- greedy committed tokens match an unhooked reference run exactly.

### Promotion Gate

```text
zero output-token mismatch
zero hook/registration/control mismatch
p50 warm-decode operation fraction <=10%
p90 warm-decode operation fraction <=25%
p50 warm-decode query-byte fraction <=10%
p90 warm-decode query-byte fraction <=25%
all six prompt families represented
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY
```

### Claim boundary

Phase C observation only. Actual sparse projection kernels, 405B activation statistics, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
''',
        encoding="utf-8",
    )

    doc = ROOT / "docs/research/EXPERIMENT_060_REAL_Q4_ZERO_SPARSITY.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f'''# EXP-060 — Pinned Real-Q4 Exact Zero-Sparsity Streaming Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 1224 format rows; checksum/reconstruction/control mismatches 0; exact zero fraction p50/p90 17.76%/20.37%; operation fraction 82.22%/85.06%; query-byte fraction 150.93%/200.86%; best matrix 69.90% operations and 190.12% bytes; peak RSS 1,169,112 KiB.

Decision:

```text
{DECISION}
```

Static exact-zero sparse streaming is rejected for this measured population. Q4 output preservation, physical kernels, and target hardware were not tested.
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
