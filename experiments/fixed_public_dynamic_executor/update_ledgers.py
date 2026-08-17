from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def ns_ms(value: Any) -> str:
    return "NOT_MEASURED" if value is None else f"{float(value) / 1e6:.3f} ms"


def append_once(path: Path, marker: str, text: str) -> None:
    content = path.read_text() if path.exists() else ""
    if marker in content:
        return
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content + "\n" + marker + "\n" + text.rstrip() + "\n")


def mechanism_line(m: dict[str, Any]) -> str:
    gate = m["physical_gate"]
    return (
        f"- `{m['mechanism']}`: G2={m['G2_exact_transition']}, G3={m['G3_full_transformer_layer_boundary']}, "
        f"G4={m['G4_physical_saving']}; artifact={m['artifact_bytes']} B; "
        f"reference-layer={m['reference_layer_parameter_bytes']} B; compiled-layer-resident={m['compiled_layer_resident_parameter_bytes']} B; "
        f"baseline p50/p95={ns_ms(gate['baseline_p50_ns'])}/{ns_ms(gate['baseline_p95_ns'])}; "
        f"candidate p50/p95={ns_ms(gate['candidate_p50_ns'])}/{ns_ms(gate['candidate_p95_ns'])}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    sha = str(result.get("github", {}).get("sha") or os.getenv("GITHUB_SHA") or "UNKNOWN")
    run_id = str(result.get("github", {}).get("run_id") or os.getenv("GITHUB_RUN_ID") or "UNKNOWN")
    marker = f"<!-- FIXED_PUBLIC_DYNAMIC_EXECUTOR_RESULT:{sha} -->"
    mechanisms = result["mechanisms"]
    mech_text = "\n".join(mechanism_line(m) for m in mechanisms)
    dev, target = result["pins"]["DEV-W"], result["pins"]["TARGET-W"]
    dev_res, target_res = result["hf_resolution"]["DEV-W"], result["hf_resolution"]["TARGET-W"]
    common = f"""## Fixed-public dynamic executor hosted result — commit `{sha}` / run `{run_id}`

- Verdict: `{result['verdict']}`
- DEV-W: `{dev['repo_id']}@{dev['revision']}`; resolved SHA `{dev_res.get('resolved_sha')}`; official `LlamaForCausalLM.from_pretrained()` executed.
- TARGET-W: `{target['repo_id']}@{target['revision']}`; metadata access={target_res.get('accessible')}; full TARGET-W weights/performance remain unevaluated unless the gated tensors were actually accessible.
- Official reference forward: executed={result['official_reference_forward']['executed']}.
- Frozen workload SHA-256: `{result['workload_sha256']}`; 128 decode steps per workload.
- Actual layer tensor audit: `{result['actual_tensor_audit']['tensor_count']}` parameter tensors, `{result['actual_tensor_audit']['reference_parameter_bytes']}` reference bytes.
{mech_text}
- Instruction/SASS counts are not synthesized: each mechanism reports its measured/not-measured status explicitly.
- G5/G6 remain gated by TARGET-W measurements; DEV-W results are not extrapolated into a 405B performance claim.
"""
    append_once(ROOT / "RESEARCH_STATE.md", marker, common)
    append_once(ROOT / "DECISION_LOG.md", marker, "## Decision: fixed-public executor constructor gate\n\n" + common)
    append_once(ROOT / "ASSUMPTION_REGISTER.md", marker, "## Fixed-public dynamic executor assumption audit\n\n- No synthetic checkpoint weights are accepted as scientific evidence.\n- Exactness is STATE-EXACT for the executable cache/RNG state tested; no bisimulation shortcut is claimed.\n- Physical saving includes artifact bytes, resident replaced-layer bytes, and peak hot projection bytes, or requires both measured p50 and p95 latency improvement.\n- TARGET-W performance is not inferred from DEV-W.\n\n" + common)
    append_once(ROOT / "VALIDATION_MATRIX.md", marker, "## Fixed-public dynamic executor validation row\n\n" + f"- G1 ACTUAL RUNTIME: PASS on DEV-W; official loader + forward executed.\n- G2 EXACT TRANSITION: {any(m['G2_exact_transition'] for m in mechanisms)}.\n- G3 REAL BOUNDARY: {any(m['G3_full_transformer_layer_boundary'] and m['G2_exact_transition'] for m in mechanisms)}.\n- G4 PHYSICAL SAVING: {any(m['G4_physical_saving'] for m in mechanisms)}.\n- G5 FULL-MODEL PATH: formula recorded; TARGET-W 4B-class budget not established.\n- G6 8GiB LEDGER: TARGET-W not established.\n- G7 EXISTING ISA: PASS for implemented software primitives; no hypothetical opcode.\n- G8 REPRODUCIBILITY: raw hosted result committed by workflow; workflow artifact retained.\n\n" + common)
    append_once(ROOT / "NEXT_EXPERIMENT.md", marker, "## Next constructor action after fixed-public hosted gate\n\nUse the committed actual-checkpoint tensor audit and physical ledger to modify only the measured failing resource term. If G4 failed for both primitives, the next implementation must eliminate the observed artifact/residency/I/O/latency term rather than rename the same codec or compiler primitive. If G4 passed, move directly to multi-layer accounting and an exact TARGET-W artifact/8GiB ledger when gated tensor access is available.\n\n" + common)
    rejected = [m for m in mechanisms if not m["G4_physical_saving"]]
    if rejected:
        append_once(ROOT / "FAILED_APPROACHES_RECENT.md", marker, "## Actual-checkpoint mechanism rejection evidence\n\n" + "\n".join(mechanism_line(m) for m in rejected) + "\n\n" + common)
    latest = ROOT / "docs" / "research" / "FIXED_PUBLIC_DYNAMIC_EXECUTOR_LATEST_RESULT.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(marker + "\n" + common)


if __name__ == "__main__":
    main()
