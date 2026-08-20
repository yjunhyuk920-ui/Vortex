from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- EXP087A_FP32_CORRECTION_RESULT -->"
INVALID_RESULT_COMMIT = "4a23f49cd6e5354e8616eb930814e11db95cd933"


def append_once(path: Path, block: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    path.write_text(
        text + separator + MARKER + "\n" + block.rstrip() + "\n",
        encoding="utf-8",
    )


def fields(result: dict[str, Any]) -> dict[str, Any]:
    measured = result["MEASURED"]
    queries = measured["queries"]
    oracle = queries["evaluation_oracle"]["report"]
    router = queries["evaluation_router"]["report"]
    build = queries["build_oracle"]["report"]
    resources = result["DERIVED"]["target_resources"]
    return {
        "source": result.get("source_sha"),
        "decision": result.get("authoritative_decision"),
        "core": result.get("deterministic_core_sha256"),
        "build_program_exact": measured.get("build_program_exact"),
        "build_vectors": f"{build.get('exact_vectors')} / {build.get('count')}",
        "oracle_vectors": f"{oracle.get('exact_vectors')} / {oracle.get('count')}",
        "oracle_fraction": oracle.get("vector_exact_fraction"),
        "oracle_row_p50": oracle.get("row_exact_fraction_p50"),
        "oracle_mismatch_p50": oracle.get("mismatched_rows_p50"),
        "router_vectors": f"{router.get('exact_vectors')} / {router.get('count')}",
        "sidecar_gib": resources.get("sidecar_gib"),
        "compiled_fraction": resources.get("compiled_mlp_operation_fraction"),
        "whole_fraction": resources.get("whole_model_operation_fraction_if_only_mlp_replaced"),
        "service_tokens": result["DERIVED"].get(
            "minimum_service_tokens_to_amortize_build_transitions"
        ),
        "gates": result.get("gates"),
    }


def latest_markdown(result: dict[str, Any], result_path: Path) -> str:
    d = fields(result)
    return f"""# EXP-087A corrected latest result

## Provenance correction

- superseded control-failure result commit: `{INVALID_RESULT_COMMIT}`
- corrected source SHA: `{d['source']}`
- raw corrected result: `{result_path.relative_to(ROOT)}`
- deterministic core: `{d['core']}`
- evidence: `E1`

The earlier build rejection used FP64 predicate labels with an FP32 deployed sidecar. The corrected compiler selects and fits predicates with the exact charged FP32 runtime representation. Checkpoint, states, language capacity, resource equations, and thresholds are unchanged.

## Corrected public-checkpoint result

```text
decision                              {d['decision']}
build program exact                   {d['build_program_exact']}
build oracle vectors exact            {d['build_vectors']}
evaluation oracle vectors exact       {d['oracle_vectors']}
evaluation oracle vector fraction     {d['oracle_fraction']}
evaluation oracle row-exact p50       {d['oracle_row_p50']}
evaluation oracle mismatched rows p50 {d['oracle_mismatch_p50']}
evaluation router vectors exact       {d['router_vectors']}
```

## Frozen favorable ledger

```text
sidecar GiB                           {d['sidecar_gib']}
compiled MLP operation fraction       {d['compiled_fraction']}
whole-model fraction if MLP-only      {d['whole_fraction']}
minimum build-amortization tokens     {d['service_tokens']}
gates                                 {json.dumps(d['gates'], sort_keys=True)}
```

This result is scoped to the official small public checkpoint and one complete MLP boundary. Complete Transformer successor state, physical sparse/bitwise lowering, TARGET-W, 8-GiB VRAM, and 4B-class p50/p95 remain `NOT TESTED`.
"""


def record(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = fields(result)
    latest = latest_markdown(result, result_path)
    (ROOT / "docs/research/EXP_087A_FP32_CORRECTED_LATEST_RESULT.md").write_text(
        latest, encoding="utf-8"
    )
    (ROOT / "docs/research/EXP_087A_LATEST_RESULT.md").write_text(
        latest, encoding="utf-8"
    )

    summary = f"""## EXP-087A corrected finite-word residual Gate

- superseded result: `{INVALID_RESULT_COMMIT}` (`FP64 compile / FP32 query predicate mismatch`)
- corrected source: `{d['source']}`
- corrected decision: `{d['decision']}`
- build programs exact: `{d['build_program_exact']}`
- target-seeing evaluation oracle: `{d['oracle_vectors']}` complete BF16 vectors
- evaluation row-exact p50: `{d['oracle_row_p50']}`
- projected sidecar: `{d['sidecar_gib']} GiB`
- favorable compiled-MLP operation fraction: `{d['compiled_fraction']}`
- raw evidence: `{result_path.relative_to(ROOT)}`
- deterministic core: `{d['core']}`

The correction changed only predicate numerical consistency. All frozen scientific capacity and resource parameters remained fixed. The corrected result is authoritative for this mechanism fingerprint.
"""
    append_once(ROOT / "RESEARCH_STATE.md", summary)
    append_once(ROOT / "DECISION_LOG.md", summary)
    append_once(ROOT / "VALIDATION_MATRIX.md", summary)

    assumption = f"""## EXP-087A predicate-semantics correction

- The first hosted result's build failure is superseded because compiler labels and deployed predicate tensors used different precisions.
- The corrected run uses the exact FP32 mean/basis representation charged by the sidecar equation for selection, fitting, and replay.
- Corrected integrity Gate: `{d['gates'].get('integrity')}`.
- The corrected mechanism verdict is `{d['decision']}`; TARGET-W and hardware remain unverified.
"""
    append_once(ROOT / "ASSUMPTION_REGISTER.md", assumption)

    architecture = f"""## EXP-087A corrected architecture status

The quadratic residual library remains an offline compiler candidate, not a production executor. Corrected promotion status is `{d['decision']}`. No complete-layer or successor-state component enters the architecture unless a later Gate explicitly promotes it.
"""
    append_once(ROOT / "ARCHITECTURE.md", architecture)

    reproduction = f"""## EXP-087A corrected reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_087a
python experiments/exp_087a/run_fp32_corrected_experiment.py \\
  --config experiments/exp_087a/config.json \\
  --output-dir results/exp_087a_fp32/<source-sha>
```

Verify the corrected result directory's `checksums.sha256`. The original result is retained only as a provenance record for the fixed predicate-boundary defect.
"""
    append_once(ROOT / "REPRODUCIBILITY.md", reproduction)

    if str(d["decision"]).startswith("REJECT_"):
        failure = summary + """
Stop rule: do not sweep context width, polynomial degree, program count, clustering, prompt/layer selection, state-keyed routing, or tolerance. The corrected target-seeing oracle is the favorable upper bound for this frozen finite-context degree-two language. Reopening requires a non-polynomial, recurrent, or otherwise materially new exact information source with a complete resource equation.
"""
        append_once(ROOT / "FAILED_APPROACHES_RECENT.md", failure)
        append_once(ROOT / "FAILED_APPROACHES.md", failure)
        next_action = """## Post-EXP-087A corrected handoff

The next core candidate may not be another finite-context polynomial capacity sweep. It must introduce a materially different exact information source or state dependency and first pass an E0 target-scale resource equation. Priority is a checkpoint-static non-polynomial superinstruction or a causal recurrent transition program that produces exact successor state without dense fallback.
"""
    else:
        next_action = """## Post-EXP-087A corrected handoff

The corrected Gate promoted the component. The next mandatory step is an actual complete-MLP replacement followed by a complete Transformer-layer and 128-consecutive-transition successor-state Gate with existing-ISA byte and instruction accounting.
"""
    append_once(ROOT / "NEXT_EXPERIMENT.md", next_action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    record(args.result.resolve())
