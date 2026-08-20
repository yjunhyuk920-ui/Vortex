from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREREG_MARKER = "<!-- EXP087A_PREREGISTERED -->"
RESULT_MARKER = "<!-- EXP087A_RESULT -->"


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        return
    separator = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    path.write_text(text + separator + marker + "\n" + block.rstrip() + "\n", encoding="utf-8")


def preregister() -> None:
    append_once(
        ROOT / "NEXT_EXPERIMENT.md",
        PREREG_MARKER,
        """## Active EXP-087A — Cross-Weight Quadratic BF16 Residual Generator Gate

The active cheapest decisive Gate tests a checkpoint-static, non-affine finite-word language rather than another page, rank, or precision selector. Four programs receive only twelve current-input predicate bits and generate all 16 residual bits of every complete SwiGLU BF16 output word with a degree-two GF(2) program.

Frozen public evidence is `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, official eager BF16 `LlamaForCausalLM`, layer-0 complete MLP, 96 build states and 96 unseen causal evaluation states across six families. The favorable oracle may select among the four already compiled programs but may not use a state key, prefix table, dense MLP call, or fallback.

Promotion requires 100% full-vector BF16 exactness on every held-out state, at least four held-out states per used program, projected sidecar no greater than 4 GiB, and favorable compiled-MLP work no greater than `1.185185185%`. Oracle failure permanently closes width/degree/program-count sweeps around this finite-context quadratic residual language.""",
    )
    append_once(
        ROOT / "ASSUMPTION_REGISTER.md",
        PREREG_MARKER,
        """## EXP-087A frozen assumptions

- `A-087A-1`: a complete SwiGLU BF16 word residual is a reusable degree-two Boolean function of twelve checkpoint-static input predicates on unseen causal states. **UNVERIFIED**.
- `A-087A-2`: four non-state-keyed programs can cover every held-out state without dense fallback. **UNVERIFIED**.
- `A-087A-3`: the target-shape packed residual sidecar fits below 4 GiB and its favorable packed-word query work is below the MLP target fraction. **DERIVED, NOT PHYSICAL**.
- `A-087A-4`: failure of the target-seeing oracle rejects this exact finite-context quadratic fingerprint but not every non-polynomial or recurrent transition compiler. **FIXED SCOPE**.""",
    )
    append_once(
        ROOT / "VALIDATION_MATRIX.md",
        PREREG_MARKER,
        """## EXP-087A preregistered validation

| Requirement | Frozen check | Status before run |
|---|---|---|
| Official checkpoint | SmolLM2-135M pinned revision through `LlamaForCausalLM` | NOT TESTED |
| Finite-word integrity | independent synthetic GF(2) quadratic replay | NOT TESTED |
| Build language | four assigned populations 100% BF16 vector exact | NOT TESTED |
| Unseen oracle | 96/96 held-out vectors exact; >=4 states/program | NOT TESTED |
| Causal router | nearest-centroid held-out vectors exact | NOT TESTED |
| Sidecar | projected <=4 GiB | DERIVED BY RUNNER |
| Query work | favorable packed compiled-MLP fraction <=1.185185185% | DERIVED BY RUNNER |
| Complete layer/state | actual replacement and successor state | NOT TESTED |
| 405B/8 GiB/latency | Phase-D evidence | NOT TESTED |""",
    )


def format_result(result: dict[str, Any]) -> str:
    measured = result.get("MEASURED", {})
    derived = result.get("DERIVED", {})
    queries = measured.get("queries", {})
    resources = derived.get("target_resources", {})
    gates = result.get("gates", {})
    oracle = queries.get("evaluation_oracle", {}).get("report", {})
    router = queries.get("evaluation_router", {}).get("report", {})
    return f"""# EXP-087A latest result

```text
source SHA             {result.get('source_sha')}
decision               {result.get('authoritative_decision')}
evidence                {result.get('evidence_level')}
deterministic core      {result.get('deterministic_core_sha256')}
```

## Actual public-checkpoint result

```text
build states                         {measured.get('build_state_count')}
evaluation states                    {measured.get('evaluation_state_count')}
build program exact                  {measured.get('build_program_exact')}
evaluation oracle vectors exact      {oracle.get('exact_vectors')} / {oracle.get('count')}
evaluation oracle vector fraction    {oracle.get('vector_exact_fraction')}
evaluation oracle row-exact p50      {oracle.get('row_exact_fraction_p50')}
evaluation oracle mismatched rows p50 {oracle.get('mismatched_rows_p50')}
evaluation router vectors exact      {router.get('exact_vectors')} / {router.get('count')}
evaluation router vector fraction    {router.get('vector_exact_fraction')}
```

## Target-shape favorable ledger

```text
feature count                         {resources.get('feature_count')}
sidecar bytes                         {resources.get('sidecar_bytes')}
sidecar GiB                           {resources.get('sidecar_gib')}
compiled MLP operation fraction       {resources.get('compiled_mlp_operation_fraction')}
whole-model fraction if MLP-only      {resources.get('whole_model_operation_fraction_if_only_mlp_replaced')}
minimum build-amortization tokens     {derived.get('minimum_service_tokens_to_amortize_build_transitions')}
```

## Gates

```text
{json.dumps(gates, sort_keys=True)}
```

This is a small-real-checkpoint favorable-oracle result. It is not a complete MLP replacement, successor-state result, CUDA benchmark, 405B execution, 8-GiB allocation, or 4B-class latency measurement.
"""


def record_result(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    latest = ROOT / "docs/research/EXP_087A_LATEST_RESULT.md"
    latest.write_text(format_result(result), encoding="utf-8")
    decision = result.get("authoritative_decision", "UNKNOWN")
    measured = result.get("MEASURED", {})
    oracle = measured.get("queries", {}).get("evaluation_oracle", {}).get("report", {})
    resources = result.get("DERIVED", {}).get("target_resources", {})
    result_summary = f"""## EXP-087A — quadratic BF16 residual generator

```text
decision                            {decision}
evaluation oracle vector exact      {oracle.get('exact_vectors')} / {oracle.get('count')}
evaluation oracle row-exact p50     {oracle.get('row_exact_fraction_p50')}
projected sidecar GiB               {resources.get('sidecar_gib')}
compiled MLP operation fraction     {resources.get('compiled_mlp_operation_fraction')}
```

The compiler used four checkpoint-static programs and only twelve runtime input predicate bits per program. It generated complete 16-bit BF16 XOR residual words for the composed SwiGLU output and used no state key, dense query call, row/page repair, or fallback. Authority: `{result_path.relative_to(ROOT)}`; deterministic core `{result.get('deterministic_core_sha256')}`.
"""
    append_once(ROOT / "RESEARCH_STATE.md", RESULT_MARKER, result_summary)
    append_once(ROOT / "DECISION_LOG.md", RESULT_MARKER, result_summary)
    append_once(ROOT / "VALIDATION_MATRIX.md", RESULT_MARKER, result_summary)
    if str(decision).startswith("REJECT_"):
        append_once(
            ROOT / "FAILED_APPROACHES_RECENT.md",
            RESULT_MARKER,
            result_summary
            + "\nStop rule: do not sweep context width, polynomial degree, program count, clustering, prompts, layers, precision, or state-keyed routing around this finite-context quadratic residual fingerprint. Reopening requires a non-polynomial or recurrent exact information source with a new resource equation.",
        )
    append_once(
        ROOT / "NEXT_EXPERIMENT.md",
        RESULT_MARKER,
        result_summary
        + "\nThe next Gate must follow the recorded decision. An oracle rejection authorizes no nearby polynomial-capacity rescue; the next candidate must change the exact information source or state dependency.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preregister", "result"), required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister()
    else:
        if args.result is None:
            parser.error("--result is required in result mode")
        record_result(args.result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
