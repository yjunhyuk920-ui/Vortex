#!/usr/bin/env python3
"""Freeze EXP-063 authority and register EXP-064."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exp_063"
RUN = 30846082964
ARTIFACT = 8868770832
ARTIFACT_NAME = "exp-063-candidate-30846082964"
ARTIFACT_SIZE = 2371412
ZIP_SHA = "b900a7019d8527d6f67d0eb412bb2fb7a0331188d84cd74444ca10762a105a14"
SOURCE_HEAD = "979bde3a23b76270f740740fbf511c7f90900a7c"
MERGE_SHA = "488fa0e3785885bbcea25681aae55bb361fa0f84"
CONFIG_SHA = "69ebb4868b3707bbdf42d07a9f7f75458c147eb9c43dfe7e92e93843a5ffc32b"
DECISION = "REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY"
MARKER = "<!-- EXP-063-AUTHORITATIVE-FINAL -->"


def append_once(path: str, body: str) -> None:
    target = ROOT / path
    current = target.read_text(encoding="utf-8")
    if MARKER not in current:
        target.write_text(current.rstrip() + "\n\n" + body.strip() + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def checksums() -> None:
    lines=[]
    for path in sorted(RESULT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(RESULT).as_posix()}")
    (RESULT/"checksums.sha256").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    sp=RESULT/"summary.json"
    summary=json.loads(sp.read_text(encoding="utf-8")); m=summary["MEASURED"]
    if summary.get("authoritative_decision") != DECISION: raise SystemExit("unexpected decision")
    expected={"case_count":18,"forward_row_count":1152,"group_row_count":147456,"registration_row_count":171,"token_registration_or_control_mismatches":0}
    for key,value in expected.items():
        if int(m[key]) != value: raise SystemExit(f"unexpected {key}")
    gr=rows(RESULT/"raw/group_rows.jsonl")
    if len(gr)!=147456: raise SystemExit("group row count mismatch")
    if any(int(r["duplicate_key_count"]) or int(r["duplicate_kv_count"]) for r in gr):
        raise SystemExit("authority unexpectedly contains duplicates")
    provenance={"workflow_run":RUN,"artifact_id":ARTIFACT,"artifact_name":ARTIFACT_NAME,"artifact_size_bytes":ARTIFACT_SIZE,"artifact_zip_sha256":ZIP_SHA,"source_head_sha":SOURCE_HEAD,"workflow_merge_sha":MERGE_SHA,"config_sha256":CONFIG_SHA,"frozen_date":"2026-08-04"}
    (RESULT/"raw/artifact_provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary["provenance"].update(provenance)
    frozen=json.dumps(summary,indent=2,sort_keys=True)+"\n"
    sp.write_text(frozen,encoding="utf-8"); (RESULT/"processed/aggregate.json").write_text(frozen,encoding="utf-8"); checksums()

    append_once("DECISION_LOG.md",f'''{MARKER}
## D-045/D-046 — Reject exact cached-KV equivalence reuse and select real-Q4 output-row Gate

EXP-063 authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. Across 18 causal cases, 1,152 forwards and 147,456 layer/head rows, exact Key duplicates and exact Key-Value duplicates were both zero. Token, registration and control mismatches were zero. Fully accounted warm whole-model p50/p90 operation fractions were 100.0211%/100.0273%; query-byte fractions were 106.2629%/119.4005%. Decision: `{DECISION}`.

D-046 opens EXP-064: inspect exact identical, sign-related, and prototype-plus-sparse-delta output rows in the pinned real Q4 dense matrices. This is the row-space dual not covered by EXP-057's column grouping.''')
    append_once("FAILED_APPROACHES.md",f'''{MARKER}
## F-031 — Exact cached Key/Key-Value equivalence reuse

No exact K or KV duplicate occurred in 147,456 measured layer/head rows. Hashing and metadata increased bytes. Do not revive by using approximate similarity while claiming exactness, by counting repeated token IDs instead of vector bit patterns, or by omitting local-window eligibility and metadata. Retain the validator as auxiliary.''')
    append_once("RESEARCH_STATE.md",f'''{MARKER}
## EXP-063 closed — Exact cached-KV equivalence

Authority: workflow `{RUN}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`. Exact K/KV duplicate counts were zero across 147,456 group rows. Warm p50/p90 work was 100.021%/100.027%; bytes 106.263%/119.401%. Decision: `{DECISION}`. Current frontier: EXP-064 pinned real-Q4 output-row equivalence and sparse-delta structure. 405B, 8 GiB, physical kernels and target hardware remain NOT TESTED.''')
    append_once("ASSUMPTION_REGISTER.md",f'''{MARKER}
## A-039 — Real Q4 output rows may share exact prototypes

Status: ACTIVE FOR EXP-064 ONLY. Identical or sign-related weight rows can share a dot product; rows near an exact prototype may reuse the prototype dot plus an exact sparse residual. Bias additions, row mappings, prototype storage, residual indexes, activation reads and output copies/sign operations must all be charged. Approximate residuals are forbidden.''')
    append_once("VALIDATION_MATRIX.md",f'''{MARKER}
## EXP-063 closure

18 cases; 1,152 forwards; 147,456 group rows; exact K duplicates 0; exact KV duplicates 0; mismatches 0; warm p50/p90 operations 100.021%/100.027%; bytes 106.263%/119.401%. 405B and hardware NOT TESTED.''')
    append_once("ARCHITECTURE.md",f'''{MARKER}
## Cached-KV equivalence boundary

The runtime must not maintain exact K/KV grouping on the measured architecture. EXP-064 may compile static output-row prototypes from Q4 weights, but must fail closed to dense row evaluation whenever exact accounting is not favorable.''')
    append_once("HARDWARE_VALIDATION_PLAN.md",f'''{MARKER}
## EXP-063 hardware status

No grouped-attention kernel was promoted. CUDA kernels, physical KV traffic, PCIe, SSD, TTFT, tokens/sec, 405B statistics and 8 GiB residency remain NOT TESTED.''')
    append_once("REPRODUCIBILITY.md",f'''{MARKER}
## EXP-063 authority

Workflow `{RUN}`; source `{SOURCE_HEAD}`; merge `{MERGE_SHA}`; artifact `{ARTIFACT}` ({ARTIFACT_SIZE} bytes); ZIP SHA-256 `{ZIP_SHA}`; config SHA-256 `{CONFIG_SHA}`. Reproduce with `experiments/exp_063/reproduce.sh` and verify `results/exp_063/checksums.sha256`.''')

    (ROOT/"NEXT_EXPERIMENT.md").write_text('''# Next Experiment

## Closed Gate — EXP-063

No exact cached Key or Key-Value duplicate occurred in 147,456 measured layer/head rows. Fully accounted work and bytes exceeded dense execution.

```text
REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY
```

## EXP-064 — Pinned Real-Q4 Exact Output-Row Prototype and Sparse-Delta Gate

### Mechanism

For every registered Q4 dense projection, treat each output row as a linear form. Compile three exact candidates:

1. identical-row groups: one dot product plus output copies;
2. sign-canonical groups: one dot product plus exact sign operations;
3. prototype rows plus exact sparse residuals: `w_r = p_g + delta_r`, so `w_r*x = p_g*x + delta_r*x`.

Bias is never folded away: every output bias addition remains charged. No approximation, thresholding, retraining, or changed quantization is allowed.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and the same FP32-to-Q4 extraction/checksum contract as EXP-057..060. Analyze all 153 two-dimensional tensors and promote only the 144 registered dense projections.

### Accounting

Charge prototype and residual weight bytes, row/prototype mappings, residual column indexes and values, activation reads, prototype dot products, sparse residual multiply-adds, output copies/signs, and all bias additions. Compare against packed Q4 dense row evaluation. Report identical/sign group counts, residual density, operation and query-byte fractions, compilation cost, reconstruction checksum, and 405B storage projection.

### Controls

- exact identical and sign-related rows compress;
- sparse-delta rows reconstruct exactly;
- one changed Q4 nibble prevents false identity;
- dense-random and forced-unique adversaries do not compress;
- selected representation round-trips every scalar;
- no runtime lookup table over activations.

### Promotion Gate

```text
zero checksum/reconstruction/control mismatch
all 144 dense projections represented
p50 operation fraction <=10%
p90 operation fraction <=25%
p50 query-byte fraction <=10%
p90 query-byte fraction <=25%
best dense/unique adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY
```

### Claim boundary

Phase C weight observation only. Q4 output preservation under a physical kernel, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
''',encoding="utf-8")

    doc=ROOT/"docs/research/EXPERIMENT_063_CACHED_KV_EQUIVALENCE.md"; doc.parent.mkdir(parents=True,exist_ok=True)
    doc.write_text(f'''# EXP-063 — Pinned Causal Exact Cached-KV Equivalence Reuse Gate

Authority: workflow `{RUN}`, source `{SOURCE_HEAD}`, merge `{MERGE_SHA}`, artifact `{ARTIFACT}`, ZIP SHA-256 `{ZIP_SHA}`.

MEASURED: 3 models; 18 cases; 1,152 forwards; 147,456 group rows; exact K duplicates 0; exact KV duplicates 0; mismatches 0; warm p50/p90 work 100.021%/100.027%; bytes 106.263%/119.401%; peak RSS 1,087,700 KiB.

Decision:

```text
{DECISION}
```

Exact cached-KV equivalence reuse is rejected for this population. Physical kernels, 405B, 8 GiB and target hardware were not tested.
''',encoding="utf-8")

if __name__ == "__main__": main()
