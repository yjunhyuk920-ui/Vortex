#!/usr/bin/env python3
"""Freeze EXP-062 authority and register EXP-063."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_062"
RUN = 30844873182
ARTIFACT = 8868287407
ARTIFACT_NAME = "exp-062-candidate-30844873182"
ARTIFACT_SIZE = 523940
ZIP_SHA = "497816dcca7e6b8c40e9222ed8511efa266fe2358aab847a93795d7c04637390"
SOURCE_HEAD = "c38baa187e41760ef07676326c6a14f08635acc3"
MERGE_SHA = "891868c186eb22869925ad20cba43ef32d371589"
CONFIG_SHA = "c987fc4ab548d08036e7db534b473aa13addc50398cc3492c22222b0fb21d98f"
DECISION = "REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY"
MARKER = "<!-- EXP-062-AUTHORITATIVE-FINAL -->"


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
        raise SystemExit("unexpected EXP-062 decision")
    if int(measured["token_registration_or_control_mismatches"]) != 0:
        raise SystemExit("token/registration/control mismatch")
    if int(measured["case_count"]) != 18:
        raise SystemExit("unexpected case population")
    if int(measured["forward_row_count"]) != 1152:
        raise SystemExit("unexpected forward population")
    if int(measured["attention_row_count"]) != 9216:
        raise SystemExit("unexpected attention population")
    attention_rows = read_rows(RESULT / "raw/attention_rows.jsonl")
    if len(attention_rows) != 9216:
        raise SystemExit("attention row count mismatch")
    if any(float(row["maximum_row_sum_error"]) > 1e-5 for row in attention_rows):
        raise SystemExit("attention normalization violation")

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
## D-043/D-044 — Reject exact non-mask attention-zero skipping and select KV equivalence Gate

EXP-062 authority: `results/exp_062/summary.json`; workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}`; ZIP SHA-256 `{ZIP_SHA}`. Across 18 causal cases, 1,152 observed forwards and 9,216 attention rows matched reference generation with zero registration/control mismatch. Causal and local-window mask zeros were excluded. Warm decode contained 2,564 exact non-mask zeros among 8,404,224 eligible probabilities: aggregate 0.030508%, weighted p50 0%, p90 0.075301%, and maximum single-row 7.1629%. QK, softmax, probability scan, metadata, unchanged Linear work and bytes were charged, yielding whole-model p50/p90 operation fractions 100.0484%/100.1541% and byte fractions 100.0930%/100.3031%. Decision: `{DECISION}`.

D-044 closes exact post-softmax zero skipping for this measured population. EXP-063 tests a separate exact reuse condition: bit-identical cached Key and Key-Value vectors at causally eligible warm-decode positions, which could reuse QK scores and identical contribution products without approximation.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-030 — Exact non-mask attention-probability zero skipping

After excluding causal and local-window structural masks, warm-decode exact-zero probability density was only 0.0305% in aggregate. Whole-model work and bytes exceeded dense execution after QK, softmax, discovery, metadata, and unchanged Linear costs. Do not revive by counting structural mask zeros, reporting the 7.16% maximum row alone, or using near-zero thresholds while claiming exactness. Retain the probability validator/accounting as an auxiliary.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-062 closed — Exact non-mask attention probability sparsity

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. Warm decode had 2,564 exact eligible zeros among 8,404,224 probabilities. Whole-model p50/p90 work was 100.048%/100.154% and bytes 100.093%/100.303%. Decision: `{DECISION}`. Physical kernels, 405B statistics, 405B, 8 GiB and target hardware remain NOT TESTED. Current frontier: EXP-063 exact cached KV equivalence.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-038 — Cached Keys or Key-Value pairs may repeat exactly across positions

Status: ACTIVE FOR EXP-063 ONLY. Identical Key vectors permit one QK score to be copied for every duplicate position. Identical Key-Value pairs additionally permit one probability-times-Value product to be reused when the copied scores produce identical probabilities. EXP-063 must compare exact tensor bit patterns, exclude structurally ineligible local positions, charge cache scanning/hashing/group metadata/copies/additions, and preserve all reference tokens. Approximate vector similarity is forbidden.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-062 closure

Cases 18; forwards 1,152; attention rows 9,216; token/registration/control mismatches 0; warm eligible probabilities 8,404,224; exact non-mask zeros 2,564; whole-model p50/p90 operations 100.048%/100.154%; bytes 100.093%/100.303%. 405B and hardware remain NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Attention-probability zero boundary

The runtime must not scan post-softmax probabilities for exact zeros on the measured architecture. Structural mask zeros remain a standard attention optimization and are not VORTEX evidence. EXP-063 may inspect cached K/V bit equivalence but must fail closed to ordinary attention when no exact group exists.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-062 hardware status

No attention-sparse kernel was promoted because whole-model logical work and bytes exceeded baseline. CUDA attention kernels, physical cache traffic, PCIe, SSD, TTFT, tokens/sec, power, 405B attention statistics, and 8 GiB residency remain NOT TESTED.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-062 authority

Workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_062/reproduce.sh` and verify `results/exp_062/checksums.sha256`.''',
    )

    (ROOT / "NEXT_EXPERIMENT.md").write_text(
        '''# Next Experiment

## Closed Gate — EXP-062

Warm decode contained only 0.0305% exact non-mask zero probabilities in aggregate. Fully accounted whole-model work and bytes exceeded dense execution.

```text
REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY
```

## EXP-063 — Pinned Causal Exact Cached-KV Equivalence Reuse Gate

### Mechanism

During each warm-decode attention step, inspect causally eligible cached vectors per layer and head:

```text
exact K groups   = bit-identical Key vectors across eligible positions
exact KV groups  = bit-identical (Key, Value) vector pairs
```

For an exact K group, compute the query-key score once and copy it to all members. For an exact KV group, copied scores imply bit-identical softmax probabilities; compute the probability-times-Value vector once and reuse the product while retaining source-order output additions. No approximate similarity, clustering, quantization, or reordered reduction is allowed.

### Pinned population

Use unchanged TinyStories-1M/3M/8M revisions, the pinned tokenizer, all six held-out prompt families, and 64-token KV-cached greedy generation. Compare a standard reference against an observation path that returns `past_key_values`; all 1,152 tokens must match.

### Eligibility

- global layers: all causal cache positions;
- local layers: only the registered local window;
- prefill is recorded separately but promotion is based on warm decode;
- tied or repeated tensor storage is not a vector duplicate unless position vectors have identical dtype, shape, and bit pattern.

### Accounting

For each query/head with eligible length `L`, unique Key count `U_K`, unique KV count `U_KV`, and head width `d`:

```text
dense QK multiplications       = d * L
candidate QK multiplications   = d * U_K
score copies                    = L - U_K
dense Value multiplications    = d * L
candidate Value multiplications= d * U_KV
Value additions                = d * L  (unchanged, source order)
cache equivalence scan/hash     = all K and V scalar bits
mapping/index metadata          = fully charged
softmax                         = unchanged
all Linear/MLP work             = unchanged
```

Report K-only, KV-pair, attention-only, and whole-model fractions. A favorable selector may choose no grouping when metadata exceeds savings, but all scan/hash cost remains charged.

### Controls

- injected duplicate K vectors: QK reuse detected;
- injected duplicate KV pairs: QK and product reuse detected;
- one-bit K or V difference prevents the corresponding group;
- positive and negative floating zero are distinct bit patterns for grouping;
- NaN payloads are rejected;
- group construction invariant to stable position enumeration;
- reference and observation tokens identical.

### Promotion Gate

```text
zero token/registration/control mismatch
all six prompt families represented
p50 whole-model warm operation fraction <=10%
p90 whole-model warm operation fraction <=25%
p50 whole-model warm query-byte fraction <=10%
p90 whole-model warm query-byte fraction <=25%
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY
```

### Claim boundary

Phase C observation only. Physical grouped-attention kernels, 405B KV equivalence statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
''',
        encoding="utf-8",
    )

    doc = ROOT / "docs/research/EXPERIMENT_062_ATTENTION_PROBABILITY_SPARSITY.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f'''# EXP-062 — Pinned Causal Exact Non-Mask Attention-Probability Sparsity Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 3 models; 18 cases; 1,152 forwards; 9,216 attention rows; mismatches 0; warm eligible probabilities 8,404,224; exact non-mask zeros 2,564; p50/p90 exact-zero fractions 0%/0.0753%; whole-model work 100.048%/100.154%; bytes 100.093%/100.303%; peak RSS 742,548 KiB.

Decision:

```text
{DECISION}
```

Exact non-mask attention-zero skipping is rejected for this measured population. Physical kernels, 405B, 8 GiB, and target hardware were not tested.
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
