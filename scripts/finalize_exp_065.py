#!/usr/bin/env python3
"""Freeze EXP-065 authority and register EXP-066."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_065"
RUN = 30870558294
ARTIFACT = 8878551394
ARTIFACT_NAME = "exp-065-candidate-30870558294"
ARTIFACT_SIZE = 244495
ZIP_SHA = "cf5bfcc53bda4117430c0856b6989704e79bb34fb52c9a4f81869bf20233155d"
SOURCE_HEAD = "22fd41697979f0e5aeb570880714a47958270d7f"
MERGE_SHA = "2e512e91b5bfcd5e30a19ef163a6438221a134dc"
CONFIG_SHA = "6dd637104c6edfdaaf424d22790e1f521dc9fa59f9a10f59552a6dfeaec18666"
DECISION = "REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY"
MARKER = "<!-- EXP-065-AUTHORITATIVE-FINAL -->"


def append_once(path: str, body: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def checksums() -> None:
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
    measured = summary["MEASURED"]
    expected = {
        "two_dimensional_tensor_count": 153,
        "dense_projection_count": 144,
        "plan_row_count": 6108,
        "selected_certificate_row_count": 306,
        "checksum_mismatches": 0,
        "missing_checksums": 0,
        "witness_mismatches": 0,
        "control_failures": 0,
        "selected_full_rearrangement_rank_count": 144,
    }
    if summary.get("authoritative_decision") != DECISION:
        raise SystemExit("unexpected EXP-065 decision")
    for key, value in expected.items():
        if int(measured[key]) != value:
            raise SystemExit(f"unexpected {key}")
    if measured.get("selected_rank_distribution") != {"4": 144}:
        raise SystemExit("unexpected selected rank distribution")

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
    certificate_rows = [
        json.loads(line)
        for line in (RESULT / "raw/certificate_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(matrix_rows) != 153 or len(plan_rows) != 6108 or len(certificate_rows) != 306:
        raise SystemExit("raw evidence population mismatch")
    if any(not row["checksum_match"] for row in matrix_rows):
        raise SystemExit("Q4 checksum mismatch")
    if any(not row["verified"] for row in certificate_rows):
        raise SystemExit("unverified selected certificate")

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
## D-049/D-050 — Reject exact Kronecker sums and open Tensor-Train/MPO bond-rank Gate

EXP-065 authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. All 153 two-dimensional tensors and 144 dense projections matched frozen Q4 checksums. Across 6,108 ordered factorization plans, selected two-prime certificates had zero witness mismatch. Every dense projection selected a full-rank 4-row rearrangement. Favorable lower-bound p50/p90 operation fractions were 203.891%/215.385%; storage fractions 100.234%/101.042%. Decision: `{DECISION}`.

D-050 opens EXP-066: exact Tensor-Train/Matrix-Product-Operator unfolding ranks, which strictly generalize one-cut Kronecker structure.''')
    append_once("FAILED_APPROACHES.md", f'''{MARKER}
## F-033 — Exact low Kronecker-rearrangement rank

Every selected real-Q4 dense rearrangement was full rank at its four-row cut. Even favorable 4-bit-factor accounting required at least 200.877% of dense operations and slightly more static storage. Do not revive by reporting query bytes alone, using one prime without witness verification, or treating a low field rank as an exact integer factor reconstruction. Retain the certifier as auxiliary.''')
    append_once("RESEARCH_STATE.md", f'''{MARKER}
## EXP-065 closed — Real-Q4 exact Kronecker rank

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. 153 tensors, 144 dense projections, 6,108 factorization plans, 306 selected certificates, zero checksum/witness/control mismatch. All 144 selected rearrangements were full rank 4. p50/p90 operation lower bounds: 203.891%/215.385%; storage: 100.234%/101.042%. Decision: `{DECISION}`. Current frontier: EXP-066 exact TT/MPO bond ranks. 405B, 8 GiB, exact factors, Q4 outputs and hardware remain NOT TESTED.''')
    append_once("ASSUMPTION_REGISTER.md", f'''{MARKER}
## A-041 — Multi-cut TT/MPO ranks may remain low despite failed one-cut Kronecker rank

Status: ACTIVE FOR EXP-066 ONLY. Pair row and column radix modes, interleave them into MPO physical dimensions, and certify every prefix/suffix unfolding rank. Exact TT/MPO storage is lower-bounded by the certified bond ranks. Mode order search, rank metadata, scales, biases, contractions and intermediates must be charged. Approximate tensor decomposition is forbidden.''')
    append_once("VALIDATION_MATRIX.md", f'''{MARKER}
## EXP-065 closure

153 tensors; 144 dense; 6,108 plans; 306 selected two-prime certificates; zero checksum/witness/control mismatch; all selected ranks 4/full; p50/p90 operation 203.891%/215.385%; storage 100.234%/101.042%; projected storage 202.66 GB. 405B execution and hardware NOT TESTED.''')
    append_once("ARCHITECTURE.md", f'''{MARKER}
## Kronecker structure boundary

The core runtime must not assume a short exact sum of Kronecker products for generic Q4 dense weights. The modular certifier remains auxiliary. EXP-066 may only advance TT/MPO candidates after all bond-rank witnesses and favorable full accounting pass.''')
    append_once("HARDWARE_VALIDATION_PLAN.md", f'''{MARKER}
## EXP-065 hardware status

No Kronecker kernel or exact factor reconstruction was promoted. Q4 output preservation, CUDA, physical traffic, PCIe, SSD, TTFT, tokens/sec, 405B execution and 8 GiB residency remain NOT TESTED.''')
    append_once("REPRODUCIBILITY.md", f'''{MARKER}
## EXP-065 authority

Workflow `{RUN}`; source `{SOURCE_HEAD}`; merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_065/reproduce.sh` and verify `results/exp_065/checksums.sha256`.''')

    (ROOT / "NEXT_EXPERIMENT.md").write_text('''# Next Experiment

## Closed Gate — EXP-065

All 144 dense projections selected a full-rank four-row Kronecker rearrangement. Favorable lower-bound operations exceeded dense execution by more than 2x and static storage did not shrink.

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

## EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

### Mechanism

Factor matrix dimensions into ordered radix sequences:

```text
m = product_k m_k
n = product_k n_k
```

Pad the shorter sequence only with unit modes, pair `(m_k,n_k)`, and reshape the Q4 matrix into an interleaved Matrix-Product-Operator tensor with physical mode sizes `d_k=m_k*n_k`. For every cut `k`, certify the exact rank of the prefix/suffix unfolding:

```text
R_k = rank(unfold(W, product_{i<=k} d_i, product_{i>k} d_i))
```

These are necessary TT/MPO bond ranks. With `R_0=R_L=1`, exact core storage is lower-bounded by:

```text
sum_k R_{k-1} * m_k * n_k * R_k
```

All admissible radix schedules and deterministic mode-order variants defined before execution must be evaluated. Every selected cut receives independently verified witnesses under at least two primes.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and frozen EXP-057 Q4 checksums. Analyze all 153 two-dimensional tensors and report promotion statistics over all 144 dense projections.

### Accounting

Charge the bond-rank storage lower bound, mode-order metadata, per-row scales and biases, input reads, MPO contractions, every intermediate tensor read/write, output reductions, compilation and certificate work. Use favorable 4-bit core storage so rejection remains conservative.

### Controls

- exact rank-1 and low-bond MPO tensors certify correctly;
- a one-nibble mutation raises at least one bond rank;
- dense-random and forced-unique tensors produce high bond ranks;
- interleaved reshape/order round trips are exact;
- every selected bond witness verifies under two primes;
- no approximation, training, activation table or changed quantization.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 lower-bound operation fraction <=10%
p90 lower-bound operation fraction <=25%
p50 lower-bound storage fraction <=10%
p90 lower-bound storage fraction <=25%
dense-random adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
exact integer MPO reconstruction before operation-replacement promotion
```

Failure decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

### Claim boundary

Phase C weight observation and exact unfolding-rank certification only. Exact MPO cores, Q4 model-output preservation, a physical MPO kernel, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
''', encoding="utf-8")

    document = ROOT / "docs/research/EXPERIMENT_065_KRONECKER_RANK.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(f'''# EXP-065 — Pinned Real-Q4 Exact Kronecker-Rearrangement Rank Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 153 tensors; 144 dense projections; 6,108 factorization plans; 306 selected two-prime certificates; zero checksum, witness or control mismatch. All 144 selected rearrangements were full rank 4. Favorable lower-bound p50/p90 operations were 203.891%/215.385%; storage 100.234%/101.042%; projected storage 202.66 GB.

Decision:

```text
{DECISION}
```

Exact integer factors, Q4 output preservation, physical kernels, 405B, 8 GiB and target hardware were not tested.
''', encoding="utf-8")


if __name__ == "__main__":
    main()
