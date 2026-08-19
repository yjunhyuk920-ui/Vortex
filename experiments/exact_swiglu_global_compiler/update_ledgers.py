from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _format_ns(value: Any) -> str:
    return "NOT_MEASURED" if value is None else f"{int(value) / 1_000_000:.3f} ms"


def _format_pct(value: Any) -> str:
    return "NOT_MEASURED" if value is None else f"{100.0 * float(value):.9f}%"


def _append_once(path: Path, marker: str, section: str) -> None:
    text = path.read_text() if path.exists() else ""
    if marker in text:
        return
    separator = "" if not text or text.endswith("\n\n") else "\n"
    path.write_text(text + separator + section.rstrip() + "\n")


def _write_latest(result: dict[str, Any]) -> str:
    compiler = result["compiler"]
    gate = compiler["physical_gate"]
    audit = compiler["compiler_audit"]
    fiber = audit["q4_nonlinear_fiber_tree_screen"]
    github = result["github"]
    section = f"""# Exact Whole-SwiGLU Global Compiler Latest Result

Source SHA: `{github.get('sha')}`
Workflow run: `{github.get('run_id')}`
Verdict: `{result['verdict']}`
Highest gate: `{result['highest_gate']}`

## Measured public-checkpoint result

```text
DEV-W                                  {result['pins']['DEV-W']['repo_id']}@{result['pins']['DEV-W']['revision']}
compiler semantic unit                 complete down(SiLU(gate(x))*up(x))
selected joint gate/up lowering        {compiler['selected_joint_affine_mode']}
completed transitions                  {compiler['completed_transitions']} / {compiler['expected_transitions']}
G2 exact transition                    {compiler['G2_exact_transition']}
G3 complete Transformer layer          {compiler['G3_full_transformer_layer_boundary']}
G4 continuous decode                   {compiler['G4_continuous_decode']}
G5 physical saving                     {compiler['G5_physical_saving']}
reference layer bytes                  {compiler['reference_layer_parameter_bytes']}
compiled artifact bytes                {compiler['artifact_bytes']}
compiled layer resident bytes          {compiler['compiled_layer_resident_parameter_bytes']}
peak decoded weight bytes              {compiler['peak_projection_hot_bytes']}
baseline p50 / p95                     {_format_ns(gate.get('baseline_p50_ns'))} / {_format_ns(gate.get('baseline_p95_ns'))}
candidate p50 / p95                    {_format_ns(gate.get('candidate_p50_ns'))} / {_format_ns(gate.get('candidate_p95_ns'))}
```

All reported transition and successor-state results are from the actual official SmolLM2 checkpoint path. A compiler-substrate pass is not a target-scale core rewrite.

## Global nonlinear audit

```text
exact duplicate gate rows              {audit['gate_rows']['duplicate_rows']}
exact duplicate up rows                {audit['up_rows']['duplicate_rows']}
exact duplicate gate/up functions      {audit['gate_up_functions']['duplicate_rows']}
exact duplicate down columns           {audit['down_columns']['duplicate_rows']}
exact duplicate complete fibers        {audit['complete_nonlinear_fibers']['duplicate_rows']}
identity-CSE optimistic fraction        {_format_pct(compiler['identity_cse_optimistic_coefficient_fraction'])}
Q4 fiber-tree screen status             {fiber.get('status')}
Q4 fiber-tree certified fraction        {_format_pct(compiler['q4_fiber_tree_certified_fraction'])}
registered target-equivalent fraction   {_format_pct(compiler['target_equivalent_fraction'])}
implemented native target rewrite       {compiler['implemented_target_scale_native_rewrite']}
```

The Q4 fiber-tree screen is an exact integer-surrogate screen only. It is not native BF16 or physical-performance evidence.

## Scientific interpretation

The repository now contains a proof-carrying compiler whose semantic object is the complete nonlinear SwiGLU function, an automatic actual-checkpoint structural audit, frozen-ABI lowering selection, integrity-checked artifacts, and consecutive state validation.

The initial lowering deliberately retains existing affine kernels. It abolishes the matrix boundary in the compiler IR, but it does **not** yet abolish dense matrix work in the executable lowering. Consequently this result establishes compiler infrastructure and closes only the implemented duplicate/CSE/Q4-fiber-tree rewrite rules when they miss the fixed target fraction.

The next admissible core work is one materially nonlinear, automatically generated rewrite inside this IR with a complete native-order operation, traffic, state, verification, and miss equation. Another tile-size, compression-codec, rank, or matrix-local basis sweep is not that rewrite.

## NOT TESTED

```text
TARGET-W weights or full-model compiler
405B execution
8 GiB target allocation
CUDA, PCIe, SSD, power, or thermal behavior
same-machine 4B-class p50/p95
E6 / E7
```
"""
    path = ROOT / "docs/research/EXACT_SWIGLU_GLOBAL_COMPILER_LATEST_RESULT.md"
    path.write_text(section)
    return section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    result = json.loads(result_path.read_text())
    source_sha = str(result["github"].get("sha") or "UNKNOWN")
    marker = f"<!-- EXACT_SWIGLU_GLOBAL_COMPILER_RESULT:{source_sha} -->"
    compiler = result["compiler"]
    audit = compiler["compiler_audit"]
    fiber = audit["q4_nonlinear_fiber_tree_screen"]
    latest = _write_latest(result)

    common = f"""{marker}
## Exact Whole-SwiGLU global compiler — source `{source_sha}`

- Verdict: `{result['verdict']}`; highest gate `{result['highest_gate']}`.
- Official DEV-W complete-layer compiler: G2={compiler['G2_exact_transition']}, G3={compiler['G3_full_transformer_layer_boundary']}, G4={compiler['G4_continuous_decode']}, G5={compiler['G5_physical_saving']}.
- Consecutive transitions: `{compiler['completed_transitions']}/{compiler['expected_transitions']}` with token and successor-state comparison.
- Selected native ABI mode: `{compiler['selected_joint_affine_mode']}`.
- Exact duplicate complete nonlinear fibers: `{audit['complete_nonlinear_fibers']['duplicate_rows']}`.
- Identity-CSE optimistic coefficient fraction: `{_format_pct(compiler['identity_cse_optimistic_coefficient_fraction'])}`.
- Q4 joint-fiber tree favorable lower-bound fraction: `{_format_pct(compiler['q4_fiber_tree_certified_fraction'])}` versus target `{_format_pct(compiler['target_equivalent_fraction'])}`; native BF16 semantics remain `NOT_ESTABLISHED` for that screen.
- This establishes the global nonlinear proof-carrying compiler substrate, not a target-scale core rewrite. TARGET-W, 8 GiB, CUDA, and 4B-class latency remain `NOT TESTED`.
"""

    _append_once(ROOT / "RESEARCH_STATE.md", marker, common)
    _append_once(ROOT / "DECISION_LOG.md", marker, common)
    _append_once(ROOT / "VALIDATION_MATRIX.md", marker, common)

    next_section = f"""{marker}
## Next constructive Gate after the exact Whole-SwiGLU compiler

The complete nonlinear compiler substrate is now executable. Do not expand a broad 405B executor from the substrate alone.

The one highest-information next Gate is:

```text
derive one automatically generated proof-carrying nonlinear rewrite over the complete SwiGLU IR
freeze its native-order correctness proof and strongest counterexample
charge artifact, selector, operations, traffic, hot/cold state, verification, misses, and fallback
require the favorable target-scale operation and traffic equations to be <= 1.185185185%
then run it on the already pinned DEV-W workload
```

The rewrite must change the executable dependency. Joint payload layout, tile-size changes, another lossless codec, matrix-local rank/basis, or another Q4 Hamming tree is auxiliary or closed by the current result.
"""
    _append_once(ROOT / "NEXT_EXPERIMENT.md", marker, next_section)

    assumption = f"""{marker}
## Whole-SwiGLU compiler assumptions

- The compiler semantic unit is the complete `down(SiLU(gate(x))*up(x))` function.
- A stacked gate/up affine call is eligible only on the pinned BF16 ABI after frozen bitwise probes and remains bounded by held-out consecutive-transition evidence; it is not an all-input theorem.
- The proof-preserving fallback lowering jointly materializes gate/up payloads but executes two reference-shaped affine calls.
- The Q4 fiber-tree screen is an integer-surrogate structural Gate and cannot certify native BF16 equality or speed.
- No target-scale nonlinear rewrite is assumed to exist merely because the IR exists.
"""
    _append_once(ROOT / "ASSUMPTION_REGISTER.md", marker, assumption)

    architecture = f"""{marker}
## Exact Whole-SwiGLU compiler layer

The constructive runtime now has a compiler layer above matrix-local storage:

```text
checkpoint MLP tensors
 -> exact whole-SwiGLU semantic IR
 -> checkpoint structural audit and proof records
 -> frozen native-ABI lowering selection
 -> integrity-checked cold artifact
 -> complete Transformer-layer replacement
 -> exact token and successor-state replay
```

The first lowering still calls existing affine kernels. Future core work must add a proof-carrying nonlinear rewrite inside this IR rather than add another executor-specific matrix wrapper.
"""
    _append_once(ROOT / "ARCHITECTURE.md", marker, architecture)

    reproduction = f"""{marker}
## Reproduce exact Whole-SwiGLU global compiler Gate

```bash
python -m pytest -q tests/test_exact_swiglu_global_compiler.py
python experiments/exact_swiglu_global_compiler/run_real_gate.py \\
  --workload experiments/fixed_public_dynamic_executor/workload.json \\
  --output-dir results/exact_swiglu_global_compiler/<source-sha>
python experiments/exact_swiglu_global_compiler/update_ledgers.py \\
  --result results/exact_swiglu_global_compiler/<source-sha>/result.json
```

Use the pinned fixed-public requirements and official checkpoint revision. Verify uploaded artifact and committed raw-result hashes before promoting any Gate.
"""
    _append_once(ROOT / "REPRODUCIBILITY.md", marker, reproduction)

    handoff = f"""{marker}
## Exact Whole-SwiGLU compiler handoff

The user-selected research order is now reflected in the repository: the compiler semantic boundary is the complete nonlinear MLP. The actual DEV-W result is `{result['verdict']}`. Read `docs/research/EXACT_SWIGLU_GLOBAL_COMPILER_GATE.md`, the raw result, and `docs/research/EXACT_SWIGLU_GLOBAL_COMPILER_LATEST_RESULT.md` before adding rewrite rules.

Do not present compiler infrastructure, a stacked-call optimization, or a Q4 surrogate screen as the missing 405B solution. The next core Gate must contain a materially nonlinear exact rewrite with a complete target-scale equation.
"""
    _append_once(ROOT / "docs/research/VORTEX_RESEARCH_HANDOFF.md", marker, handoff)

    if (
        fiber.get("status") == "EXECUTED"
        and compiler["q4_fiber_tree_certified_fraction"] is not None
        and float(compiler["q4_fiber_tree_certified_fraction"])
        > float(compiler["target_equivalent_fraction"])
    ):
        failure = f"""{marker}
## Joint nonlinear-fiber Q4 Hamming-tree screen

Decision:

```text
REJECT_Q4_COMPLETE_SWIGLU_FIBER_HAMMING_TREE_AS_CORE
RETAIN_WHOLE_SWIGLU_COMPILER_SUBSTRATE
```

The actual DEV-W fibers were formed from each gate row, up row, and matching down column. Even the collision-free 32-coefficient block nearest-neighbor lower bound was `{_format_pct(compiler['q4_fiber_tree_certified_fraction'])}`, above the complete target fraction `{_format_pct(compiler['target_equivalent_fraction'])}` before tree metadata, deltas, traversal, native scaling, or physical execution. Do not rescue this tree by tile, ordering, or nearby quantizer sweeps. The result does not reject different nonlinear compiler rules.
"""
        _append_once(ROOT / "FAILED_APPROACHES_RECENT.md", marker, failure)

    print(json.dumps({
        "updated": True,
        "marker": marker,
        "latest_result_bytes": len(latest.encode()),
    }, indent=2))


if __name__ == "__main__":
    main()
