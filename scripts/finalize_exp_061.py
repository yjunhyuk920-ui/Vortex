#!/usr/bin/env python3
"""Freeze EXP-061 authority and register EXP-062."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_061"
RUN = 30843404056
ARTIFACT = 8867731496
ARTIFACT_NAME = "exp-061-candidate-30843404056"
ARTIFACT_SIZE = 662994
ZIP_SHA = "a01d31b012badd7d06087df576279b852db07813a0c7fb50d65c3a7283e9ca65"
SOURCE_HEAD = "15097a9b0323aa992679214173aaac0e7a98821c"
MERGE_SHA = "44c3d6691d78714dc975e46e19bb8fdfe97a22cf"
CONFIG_SHA = "b5635e3cd57dae39bc66c7939ef75ea7c79d6dab2a22d634c1441f0a9d930e82"
DECISION = "REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY"
MARKER = "<!-- EXP-061-AUTHORITATIVE-FINAL -->"


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
        raise SystemExit("unexpected EXP-061 decision")
    if int(measured["output_token_mismatches"]) != 0:
        raise SystemExit("hooked generation changed output tokens")
    if int(measured["hook_registration_or_control_mismatches"]) != 0:
        raise SystemExit("hook registration/control mismatch")
    if int(measured["case_count"]) != 18:
        raise SystemExit("unexpected case population")
    if int(measured["activation_call_count"]) != 56448:
        raise SystemExit("unexpected activation call population")
    if int(measured["warm_decode_call_count"]) != 54684:
        raise SystemExit("unexpected warm-decode call population")
    if float(measured["maximum_warm_decode_exact_zero_fraction"]) != 0.0:
        raise SystemExit("unexpected nonzero activation sparsity result")
    calls = read_rows(RESULT / "raw/call_rows.jsonl")
    if len(calls) != 56448:
        raise SystemExit("call row count mismatch")
    if any(int(row["exact_zero_count"]) != 0 for row in calls):
        raise SystemExit("a frozen projection input contained an exact zero")

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
## D-041/D-042 — Reject causal exact activation-zero skipping and select attention-probability Gate

EXP-061 authority: `results/exp_061/summary.json`; workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}`; ZIP SHA-256 `{ZIP_SHA}`. Across 18 pinned model/prompt cases, 1,152 hooked generation tokens matched unhooked references exactly. 147 unique projection registrations produced 56,448 calls, including 54,684 warm-decode calls and 12,165,888 warm-decode input scalars. Exact positive/negative-zero count was zero in prefill, first decode, and warm decode. A full zero scan raised weighted p50/p90 operation fractions to 100.00199%/100.390625% and query-byte fractions to 100.00404%/101.56555%. Decision: `{DECISION}`.

D-042 closes exact zero-coordinate skipping at registered dense-projection inputs for this causal population. EXP-062 measures a different runtime structure: exact non-mask zero probabilities after attention softmax and the fully accounted effect on Value accumulation and total Transformer work.''',
    )
    append_once(
        "FAILED_APPROACHES.md",
        f'''{MARKER}
## F-029 — Causal exact-zero activation-column skipping

No exact projection-input zero occurred in 56,448 calls or 17,529,344 observed input scalars across prefill and decode. Warm-decode work and query bytes slightly exceeded dense execution after mandatory zero discovery and metadata. Do not revisit with more zero scanners, module selectors, or near-zero thresholds: near-zero is approximate and exact zero population was empty. Retain the hook/accounting machinery only for architectures with explicit exact-zero activations.''',
    )
    append_once(
        "RESEARCH_STATE.md",
        f'''{MARKER}
## EXP-061 closed — Causal exact activation sparsity

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. Hooked and unhooked generation matched for all 1,152 tokens. Exact zero count was 0 over 56,448 projection calls; warm-decode p50/p90 fully accounted work was 100.002%/100.391% and bytes 100.004%/101.566%. Decision: `{DECISION}`. Physical sparse kernels, 405B activation statistics, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-062 exact non-mask attention-probability sparsity.''',
    )
    append_once(
        "ASSUMPTION_REGISTER.md",
        f'''{MARKER}
## A-037 — Non-mask causal attention probabilities may underflow to exact zero

Status: ACTIVE FOR EXP-062 ONLY. EXP-061 found no exact zeros at dense inputs, but softmax can theoretically underflow for sufficiently negative unmasked scores. EXP-062 must exclude causal-mask and padding entries, use exact returned probabilities, compare hooked/attention-output-enabled tokens with reference generation, and charge QK, softmax, Value accumulation, probability scanning, indexes, and the unchanged non-attention model work.''',
    )
    append_once(
        "VALIDATION_MATRIX.md",
        f'''{MARKER}
## EXP-061 closure

Reference/hooked output tokens 1,152/1,152 exact; projection registrations 147; calls 56,448; warm calls 54,684; hook/control mismatches 0; exact-zero count 0; warm p50/p90 operation fraction 100.002%/100.391%; query bytes 100.004%/101.566%. 405B and hardware remain NOT TESTED.''',
    )
    append_once(
        "ARCHITECTURE.md",
        f'''{MARKER}
## Activation-zero boundary

The runtime must not scan ordinary dense-projection inputs for exact zeros on the measured architecture because the observed population is empty and scanning adds work. Observation hooks remain auxiliary for architectures with explicit hard-zero nonlinearities. EXP-062 is restricted to post-softmax attention probabilities, excluding mask zeros.''',
    )
    append_once(
        "HARDWARE_VALIDATION_PLAN.md",
        f'''{MARKER}
## EXP-061 hardware status

No activation-sparse kernel was promoted because exact-zero density was zero and logical accounting exceeded dense execution. CUDA sparse projection kernels, PCIe, SSD, TTFT, tokens/sec, power, 405B activation statistics, and 8 GiB residency remain NOT TESTED.''',
    )
    append_once(
        "REPRODUCIBILITY.md",
        f'''{MARKER}
## EXP-061 authority

Workflow `{RUN}`; source head `{SOURCE_HEAD}`; workflow merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_061/reproduce.sh` and verify `results/exp_061/checksums.sha256`.''',
    )

    (ROOT / "NEXT_EXPERIMENT.md").write_text(
        '''# Next Experiment

## Closed Gate — EXP-061

No exact positive or negative zero was observed in 56,448 projection calls. Mandatory zero discovery made warm-decode logical work and bytes exceed dense execution.

```text
REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY
```

## EXP-062 — Pinned Causal Exact Non-Mask Attention-Probability Sparsity Gate

### Mechanism

Request exact attention probabilities during causal generation and measure entries equal to positive or negative zero only after excluding positions that are zero solely because of causal or padding masks. An exact zero probability permits skipping the corresponding Value-vector multiply/add for that query/head without changing output.

### Pinned population

Use unchanged TinyStories-1M/3M/8M revisions, the pinned GPT-Neo tokenizer, six held-out prompt families, prompt prefill, first decode, and 64-token KV-cached generation. Run a standard reference and an `output_attentions=True` observation path; all committed tokens must match.

### Registration

- enumerate every attention layer and head count;
- record query/key/value lengths, unmasked entry population, exact non-mask zero count, phase, token, model, prompt family, and layer;
- fail on missing attention tensors, shape mismatch, NaN, negative probability, or row-sum violation beyond the pinned numerical tolerance;
- causal-mask and padding zeros are excluded from both numerator and eligible population.

### Accounting

For each head/query with key length `L`, charge:

```text
QK score terms               = head_dim * L
softmax terms                = L
Value dense terms            = head_dim * L
Value sparse terms           = head_dim * nonzero_probability_count
probability zero scan        = L
nonzero-key indexes/pointers = exact metadata bytes
```

Total Transformer accounting must also include all unchanged dense-projection and MLP terms from the registered architecture. Report both attention-only and whole-model operation/query-byte fractions. Skipping QK or softmax is forbidden because zero status is known only afterward.

### Controls

- explicit masked logits: mask zeros excluded;
- extreme unmasked logits that underflow: exact zeros detected;
- moderate logits: no false zeros;
- positive/negative zero equivalence;
- dense versus zero-skipped Value accumulation equality in fixed scalar order;
- probability rows finite, nonnegative, and normalized;
- reference and observation generation tokens identical.

### Promotion Gate

```text
zero token/registration/control mismatch
all six families represented
p50 whole-model warm-decode operation fraction <=10%
p90 whole-model warm-decode operation fraction <=25%
p50 whole-model query-byte fraction <=10%
p90 whole-model query-byte fraction <=25%
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY
```

### Claim boundary

Phase C observation only. Physical attention-sparse kernels, 405B attention statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
''',
        encoding="utf-8",
    )

    doc = ROOT / "docs/research/EXPERIMENT_061_CAUSAL_ACTIVATION_SPARSITY.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f'''# EXP-061 — Pinned Causal Exact Activation-Sparsity Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 3 models; 18 cases; 1,152 generated tokens; 147 registered projections; 56,448 calls; output/hook/control mismatches 0; exact activation zeros 0; warm p50/p90 fully accounted work 100.002%/100.391%; warm query bytes 100.004%/101.566%; peak RSS 761,248 KiB.

Decision:

```text
{DECISION}
```

Exact activation-zero skipping is rejected for this measured population. Physical kernels, 405B, 8 GiB, and target hardware were not tested.
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
