from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREREG_MARKER = "<!-- EXP085A_PREREGISTERED -->"


def append_once(path: Path, marker: str, text: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return False
    separator = "" if not current or current.endswith("\n") else "\n"
    path.write_text(current + separator + "\n" + marker + "\n" + text.rstrip() + "\n", encoding="utf-8")
    return True


def preregister() -> list[str]:
    changed: list[str] = []
    entries = {
        "NEXT_EXPERIMENT.md": """## Active EXP-085A — Joint Exact SwiGLU Compiler Gate

The active cheapest decisive Gate compiles one actual public BF16 SwiGLU MLP as a sum of fused nonlinear macro-pages crossing gate, SiLU, multiplication, and down projection. It tests an impossible favorable oracle selector and a sound metadata selector on 24 real causal activations from `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

The frozen pass requires zero false accept/fallback, p50/p95 whole-model-equivalent fractions at or below `1.185185%/1.481481%`, and a projected complete metadata ledger within 8 GiB. The contract is `docs/research/EXPERIMENT_085A_JOINT_EXACT_SWIGLU_COMPILER_GATE.md`. No backend or complete-layer work is authorized before this Gate survives.""",
        "RESEARCH_STATE.md": """## EXP-085A preregistered constructive frontier

EXP-085A changes the operation boundary from independent matrices or a linear session basis to a complete fused SwiGLU macrofunction. It is an exact fail-closed compiler reference, not an approximate channel selector. Official DEV-W execution and the scientific decision remain `NOT RUN` until hosted evidence is committed.""",
        "ASSUMPTION_REGISTER.md": """## EXP-085A frozen assumptions

- `A-085A-1`: additive intermediate macro-pages can expose enough exact BF16 rounding closure before nearly all pages are read. **UNVERIFIED**.
- `A-085A-2`: an output-row/page bound table is small enough for the complete 405B 8-GiB ledger. **DERIVED BY THE GATE, NOT YET RUN**.
- `A-085A-3`: output-row splitting preserves gate/up BF16 rows and the down FP32 accumulation envelope contains the official result. **TO BE CONTROL-TESTED**.
- `A-085A-4`: oracle failure is sufficient to reject this page-separable fingerprint, but not every globally coupled nonlinear exact compiler. **FIXED SCOPE**.""",
        "VALIDATION_MATRIX.md": """## EXP-085A validation registration

| Gate | Evidence required | Status before run |
|---|---|---|
| official public checkpoint | pinned `LlamaForCausalLM.from_pretrained` | NOT RUN |
| real causal activations | 6 families x 4 positions | NOT RUN |
| exactness | zero row-split mismatch, bound violation, false accept, candidate mismatch | NOT RUN |
| favorable oracle | p50/p95 fractions and fallback | NOT RUN |
| sound selector | p50/p95 fractions, fallback, metadata | NOT RUN |
| complete layer / 128 steps / hardware | separate later gate | NOT AUTHORIZED |""",
    }
    for name, text in entries.items():
        path = ROOT / name
        if append_once(path, PREREG_MARKER, text):
            changed.append(name)
    return changed


def result_text(result: dict[str, Any]) -> dict[str, str]:
    verdict = result.get("authoritative_decision", "MISSING")
    source = result.get("source_sha", "UNKNOWN")
    marker = f"<!-- EXP085A_RESULT:{source} -->"
    if verdict == "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION":
        summary = f"""## EXP-085A hosted execution result

```text
source   {source}
verdict  {verdict}
error    {result.get('error', 'unknown')}
```

No scientific mechanism decision follows. The committed harness remains the executable next gate."""
        return {"marker": marker, "summary": summary, "next": summary, "decision": summary, "assumption": summary, "validation": summary, "failed": ""}

    oracle = result["aggregates"]["oracle"]
    sound = result["aggregates"]["sound"]
    projected = result["projected"]
    gates = result["gates"]
    summary = f"""## EXP-085A authoritative result

```text
source SHA                         {source}
deterministic core                 {result['deterministic_core_sha256']}
causal activations                 {result['measured_activation_count']}
oracle pages p50/p95               {oracle['pages_p50']} / {oracle['pages_p95']}
oracle weight fraction p50/p95     {oracle['whole_model_fraction_p50']:.9%} / {oracle['whole_model_fraction_p95']:.9%}
oracle operation p50/p95           {oracle['whole_model_operation_fraction_p50']:.9%} / {oracle['whole_model_operation_fraction_p95']:.9%}
oracle fallback rate               {oracle['fallback_rate']:.9%}
sound pages p50/p95                {sound['pages_p50']} / {sound['pages_p95']}
sound weight fraction p50/p95      {sound['whole_model_fraction_p50']:.9%} / {sound['whole_model_fraction_p95']:.9%}
sound operation p50/p95            {sound['whole_model_operation_fraction_p50']:.9%} / {sound['whole_model_operation_fraction_p95']:.9%}
sound fallback rate                {sound['fallback_rate']:.9%}
projected hot metadata             {projected['target_hot_metadata_gib']:.6f} GiB
integrity/oracle/sound/metadata     {gates['integrity_passed']} / {gates['oracle_passed']} / {gates['sound_passed']} / {gates['metadata_passed']}
verdict                            {verdict}
```

Structurally valid conditions were established. Large-model performance remains unverified."""
    if verdict.startswith("REJECT_"):
        next_gate = """## Next highest-information constructive gate after EXP-085A

The additive macro-page execution dependency is closed under its frozen scope. The next admissible compiler must be **nonseparable across pages**: one stored/query primitive must produce exact shared information about multiple distinct nonlinear neuron groups at lower total cost than evaluating those groups separately. Another page size, page order, rank, bound, or selector is prohibited.

Before implementation, freeze a finite constructor equation for cross-page nonlinear common-subexpression synthesis and a cheap real-checkpoint tile Gate that charges code, cold probes, query arithmetic, metadata, exact BF16 semantics, and fallback."""
        failed = f"""## EXP-085A — page-separable fused SwiGLU exact refinement

Decision: `{verdict}`.

The complete MLP was represented as an exact sum of fused gate/SiLU/up/down macro-pages. Under the frozen population and ledgers, it did not jointly satisfy oracle page depth, deployable exact closure, and 8-GiB metadata requirements. Do not reopen with page-width, prompt, layer, selector, tolerance, or margin sweeps. Reopening requires cross-page nonlinear information sharing."""
    else:
        next_gate = """## Next highest-information constructive gate after EXP-085A

Implement the surviving compiler at a complete real Transformer-layer boundary, preserve exact successor state for at least 128 transitions, and charge artifact, metadata, cold bytes, instructions, and latency. No TARGET-W promotion follows from the MLP Gate alone."""
        failed = ""
    return {
        "marker": marker,
        "summary": summary,
        "next": summary + "\n\n" + next_gate,
        "decision": summary,
        "assumption": summary,
        "validation": summary,
        "failed": failed,
    }


def record_result(result_path: Path) -> list[str]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    texts = result_text(result)
    marker = texts.pop("marker")
    changed: list[str] = []
    mapping = {
        "RESEARCH_STATE.md": texts["summary"],
        "NEXT_EXPERIMENT.md": texts["next"],
        "DECISION_LOG.md": texts["decision"],
        "ASSUMPTION_REGISTER.md": texts["assumption"],
        "VALIDATION_MATRIX.md": texts["validation"],
        "docs/research/EXPERIMENT_085A_JOINT_EXACT_SWIGLU_COMPILER_GATE.md": texts["summary"],
    }
    if texts["failed"]:
        mapping["FAILED_APPROACHES_RECENT.md"] = texts["failed"]
    for name, text in mapping.items():
        if append_once(ROOT / name, marker, text):
            changed.append(name)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preregister", "result"), required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        changed = preregister()
    else:
        if args.result is None:
            parser.error("--result is required in result mode")
        changed = record_result(args.result)
    print(json.dumps({"changed": changed, "mode": args.mode}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
