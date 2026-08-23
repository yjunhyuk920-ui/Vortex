from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-100A:START -->"
END = "<!-- EXP-100A:END -->"

MANDATORY_PATHS = (
    "RESEARCH_STATE.md",
    "NEXT_EXPERIMENT.md",
    "DECISION_LOG.md",
    "FAILED_APPROACHES.md",
    "FAILED_APPROACHES_RECENT.md",
    "ASSUMPTION_REGISTER.md",
    "VALIDATION_MATRIX.md",
    "ARCHITECTURE.md",
    "HARDWARE_VALIDATION_PLAN.md",
    "REPRODUCIBILITY.md",
    "docs/research/VORTEX_RESEARCH_HANDOFF.md",
)


def percent(value: float | None, digits: int = 9) -> str:
    if value is None:
        return "none"
    return f"{100.0 * float(value):.{digits}f}%"


def gib(value: float | int | None) -> str:
    if value is None:
        return "none"
    return f"{float(value) / 2**30:.9f} GiB"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-100A marker block in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def plan_summary(row: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    if row is None:
        return {
            "block": None,
            "arithmetic": None,
            "io": None,
            "workspace": None,
            "static": None,
            "ten_x": False,
            "p50_arithmetic": False,
            "p50_io": False,
            "workspace_pass": False,
        }
    plan = row[key]
    return {
        "block": int(row["block_length"]),
        "arithmetic": float(plan["arithmetic_ratio"]),
        "io": float(plan["io_fraction"]),
        "workspace": int(plan["maximum_workspace_bytes"]),
        "static": float(plan["total_compiled_static_bytes"]),
        "ten_x": bool(plan["ten_x_arithmetic_pass"]),
        "p50_arithmetic": bool(plan["p50_arithmetic_pass"]),
        "p50_io": bool(plan["p50_io_pass"]),
        "workspace_pass": bool(plan["workspace_pass"]),
    }


def details(result: Mapping[str, Any]) -> dict[str, Any]:
    direct = plan_summary(result.get("best_direct_row"), "direct")
    oracle = plan_summary(
        result.get("best_free_transform_oracle_row"), "free_transform_oracle"
    )
    catalog = result["catalog_summary"]
    return {
        "decision": str(result["authoritative_decision"]),
        "core": str(result["deterministic_core_sha256"]),
        "integrity": list(result["integrity_failures"]),
        "eligible": int(catalog["eligible_factorization_count"]),
        "catalog_keys": int(catalog["catalog_key_count"]),
        "orientations": int(catalog["pareto_orientation_count"]),
        "controls": int(catalog["synthetic_control_count"]),
        "control_mismatches": int(catalog["synthetic_control_mismatch_count"]),
        "direct_pass_blocks": list(result["direct_ten_x_pass_blocks"]),
        "oracle_pass_blocks": list(result["free_transform_ten_x_pass_blocks"]),
        "final_pass_blocks": list(result["direct_final_p50_pass_blocks"]),
        "direct": direct,
        "oracle": oracle,
    }


def next_gate(decision: str) -> tuple[str, str]:
    if decision == (
        "PROMOTE_CATALOGUED_RECTANGULAR_FMM_TO_FINITE_WORD_KERNEL_AND_CAUSAL_BLOCK_GATE"
    ):
        return (
            "EXP-101A — finite-word transformed-weight closure and causal-block Gate",
            "Freeze the winning complete-population plan before any new search. Prove an exact bounded-word representation for every transformed checkpoint form, implement one existing-ISA projection kernel, charge packing/decompression/repair selection, and combine it with a causally available block. Promotion requires exact official projection words and successor state over at least 128 consecutive DEV-W transitions; a perfect future block or perfect repair selector receives no runtime credit.",
        )
    if decision == "RETAIN_RECTANGULAR_RANK_HEADROOM_REQUIRE_EXPLICIT_TRANSFORM_CIRCUIT":
        return (
            "EXP-101A — exact factor-transform circuit elimination Gate",
            "Do not build a matrix kernel yet. Freeze the best rank-oracle sequence and derive a checkpoint-static exact transform circuit whose fully charged additions, scales, writes, compiled bytes, repair-row bytes, and workspace reduce the complete-population arithmetic below 10%. Reject the path immediately if even a perfect causal block cannot close that explicit-transform Gate.",
        )
    if decision == "REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE":
        return (
            "EXP-101A — new-information exact dense-arithmetic replacement Gate",
            "Do not sweep AlphaTensor catalog entries, recursion depth, beam width, block length, coefficient cap, padding, transformed word width, or compression grant. Generate three new principles that change the information source or verification unit. The first admissible Gate must present an explicit existing-ISA path to at least 10× complete-population arithmetic reduction and charge exact successor state, checkpoint bytes, metadata, and fallback.",
        )
    return (
        "EXP-100A control repair",
        "Repair only the recorded identity, tensor-reconstruction, registered-input, or deterministic-control failure. Do not interpret an invalid run scientifically and do not change the frozen search or thresholds while repairing infrastructure.",
    )


def latest_markdown(
    result: Mapping[str, Any], result_path: Path, source_commit: str
) -> str:
    d = details(result)
    direct = d["direct"]
    oracle = d["oracle"]
    gate_title, gate_text = next_gate(d["decision"])
    return f"""# EXP-100A latest result — Explicit Rectangular FMM Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- AlphaTensor catalog commit: `{result['catalog_identity']['commit']}`
- AlphaTensor catalog Git blob: `{result['catalog_identity']['git_blob_sha1']}`
- deterministic core: `{d['core']}`

## Authoritative decision

`{d['decision']}`

## Integrity and catalog

- catalog keys inspected: `{d['catalog_keys']}`
- eligible exact integral small-coefficient factorizations: `{d['eligible']}`
- retained tensor orientations: `{d['orientations']}`
- deterministic integer controls: `{d['controls']}`
- control mismatches: `{d['control_mismatches']}`
- integrity failures: `{d['integrity']}`

## Best explicit staged-transform plan

- block length: `{direct['block']}`
- arithmetic ratio: `{percent(direct['arithmetic'], 12)}`
- cold-byte fraction/token: `{percent(direct['io'], 12)}`
- favorable FMM workspace: `{gib(direct['workspace'])}`
- compiled static representation: `{gib(direct['static'])}`
- first 10x arithmetic Gate: `{direct['ten_x']}`
- final p50-equivalent arithmetic Gate: `{direct['p50_arithmetic']}`
- p50 cold-byte Gate: `{direct['p50_io']}`
- favorable workspace screen: `{direct['workspace_pass']}`
- all direct 10x joint-pass blocks: `{d['direct_pass_blocks']}`
- all direct final-p50 pass blocks: `{d['final_pass_blocks']}`

## Best free-transform rank oracle

- block length: `{oracle['block']}`
- arithmetic ratio: `{percent(oracle['arithmetic'], 12)}`
- cold-byte fraction/token: `{percent(oracle['io'], 12)}`
- favorable workspace: `{gib(oracle['workspace'])}`
- first 10x arithmetic Gate: `{oracle['ten_x']}`
- all oracle 10x joint-pass blocks: `{d['oracle_pass_blocks']}`

The oracle grants every factor transform, packing operation, and transform scratch
for free. It is algebraic headroom, not an executor result.

## Next gate

### {gate_title}

{gate_text}

## Claim boundary

The run verifies the pinned public factorization catalog and derives a bounded
405B-shape resource ledger. A causal future block, finite-word transformed-weight
closure, sound native-repair selector, existing-ISA packed kernel, complete
successor state, TARGET-W execution, physical 8-GiB allocation, and same-machine
4B p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = details(result)
    gate_title, gate_text = next_gate(d["decision"])

    for relative in MANDATORY_PATHS:
        if not (ROOT / relative).exists():
            raise FileNotFoundError(f"mandatory ledger missing: {relative}")

    latest_path = ROOT / "docs/research/EXP_100A_LATEST_RESULT.md"
    latest_path.write_text(
        latest_markdown(result, result_path, source_commit), encoding="utf-8"
    )

    direct = d["direct"]
    oracle = d["oracle"]
    common = f"""## EXP-100A — explicit rectangular FMM Gate

- decision: `{d['decision']}`
- source commit: `{source_commit}`
- evidence: `{result_path.as_posix()}`
- deterministic core: `{d['core']}`
- eligible exact catalog factorizations: `{d['eligible']}`
- retained orientations: `{d['orientations']}`
- deterministic control mismatches: `{d['control_mismatches']}`
- best explicit arithmetic / I/O: `{percent(direct['arithmetic'], 12)} / {percent(direct['io'], 12)}` at `K={direct['block']}`
- best free-transform oracle arithmetic / I/O: `{percent(oracle['arithmetic'], 12)} / {percent(oracle['io'], 12)}` at `K={oracle['block']}`
- direct 10x joint-pass blocks: `{d['direct_pass_blocks']}`
- direct final-p50 arithmetic-pass blocks: `{d['final_pass_blocks']}`
- favorable FMM workspace at best direct plan: `{gib(direct['workspace'])}`

The direct arm charges all catalogued leaf operations, factor additions/scales,
form writes, transformed-checkpoint stream expansion, role-p95 native-order
repair work, and original-row side-stream bytes. The rank oracle grants factor
transforms and packing free. Both arms grant a perfect future block with `N/A=1`;
therefore neither is a causal executor or target-hardware measurement.
"""

    replace_block(ROOT / "RESEARCH_STATE.md", common)

    next_body = f"""## Active next gate after EXP-100A

### {gate_title}

Current decision: `{d['decision']}`.

{gate_text}

The frozen target remains arbitrary public unmodified Dense 405B, exact output
and successor-state contract, one GPU with peak VRAM at most 8 GiB, and
same-machine native-4B-Q4 p50/p95 limits. Do not describe a structural FMM ratio
as measured latency.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_body)

    decision_body = f"""## EXP-100A decision

`{d['decision']}`

The pinned AlphaTensor standard-arithmetic catalog was checked by Git blob,
complete integer tensor reconstruction, and deterministic matrix-product
controls. The bounded mixed-recursion search produced best explicit arithmetic
ratio `{percent(direct['arithmetic'], 12)}` and best free-transform oracle ratio
`{percent(oracle['arithmetic'], 12)}`. Cold bytes and favorable FMM workspace
were jointly screened. This decision is scoped to the frozen catalog, integral
coefficient cap, bounded search, and perfect-future-block grants.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_body)

    if d["decision"] == "REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE":
        failure = (
            "The pinned catalogued small-coefficient mixed-recursion language failed the first 10x complete-population Gate even under a perfect future block, perfect native-repair selector, favorable 1.5x compression, and bounded 8-GiB FMM-workspace grant. The frozen catalog/search family is closed; nearby recursion or beam sweeps are prohibited."
        )
    elif d["decision"] == "INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE":
        failure = (
            "No scientific failure is registered because the run was invalid. Repair only the recorded integrity fault and rerun the unchanged contract."
        )
    else:
        failure = (
            "EXP-100A did not close the rectangular-FMM family. Its promotion is conditional and does not credit a causal block, finite-word transformed representation, repair selector, or physical kernel."
        )
    failure_body = f"""## EXP-100A — catalogued explicit rectangular FMM

{failure}

Frozen fingerprint: pinned AlphaTensor real-arithmetic catalog; integral
coefficients with absolute value at most two; six factor-space orientations;
depth at most 12; registered 405B projection population; explicit transform,
write, cold-byte, workspace, and role-p95 repair ledger; perfect `N/A=1` future
block grant.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failure_body)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failure_body)

    assumptions = f"""## EXP-100A assumptions and provenance

- `MEASURED`: catalog bytes/hash; exact integer tensor reconstruction; deterministic integer controls; committed EXP-080A/099A input identities.
- `DERIVED`: mixed split/rank products; leaf and factor-transform operations; transformed-stream expansion; repair-row side stream; favorable FMM workspace; joint I/O dynamic program.
- `PROJECTED`: application of those counts to the registered 405B projection inventory; 1.5x lossless ratio for transformed forms; role-p95 repair transfer to TARGET-W shapes.
- `UNVERIFIED`: causal block; exact finite-word transformed words; repair selector; transposition/packing kernel; TARGET-W behavior; complete 8-GiB runtime; wall-clock latency.
- Bounded beam search is not an exhaustive tensor-rank or algorithm search.
- A static-transformed plan must retain/read original BF16 repair rows; those bytes are charged.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumptions)

    validation = f"""## EXP-100A validation row

| Boundary | Contract | Result | Evidence |
|---|---|---|---|
| Pinned AlphaTensor catalog | blob identity + exact integer tensor reconstruction | `{d['control_mismatches']} mismatches`, integrity `{d['integrity']}` | E1 |
| Registered 405B projection ledger | explicit ops + bytes + favorable FMM scratch | `{d['decision']}` | E0/E1 derived/projected |
| Causal block and exact successor state | at least 128 official DEV-W transitions | `NOT TESTED` | — |
| Existing-ISA finite-word kernel | exact words and fully charged bytes/instructions | `NOT TESTED` | — |
| TARGET-W / physical 8 GiB / 4B-class p50/p95 | complete final contract | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation)

    architecture = f"""## EXP-100A architecture status

The mixed rectangular tensor sequence and cut-depth transformed-weight plan are
research descriptors, not production opcodes. Architecture promotion is
conditional on `{d['decision']}` and the active next Gate `{gate_title}`. Until
finite-word closure, a causal block, exact successor state, and existing-ISA
lowering pass, the production executor architecture remains unchanged.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture)

    hardware = f"""## EXP-100A hardware status

No target GPU or 405B checkpoint was executed. Best direct favorable FMM scratch
was `{gib(direct['workspace'])}` and is a projected lower-bound workspace, not a
complete VRAM measurement. KV, allocator reserve, decompression buffers,
concurrent streams, packing, repair selection, PCIe/SSD/HBM throughput, TTFT,
and p50/p95 are `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware)

    reproduction = f"""## EXP-100A reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_100a
python experiments/exp_100a/run_experiment.py \\
  --config experiments/exp_100a/config.json \\
  --output-dir results/exp_100a/<source-commit>
python experiments/exp_100a/update_ledgers.py \\
  --result results/exp_100a/<source-commit>/result.json \\
  --source-commit <source-commit>
```

Catalog commit: `{result['catalog_identity']['commit']}`; Git blob:
`{result['catalog_identity']['git_blob_sha1']}`. Verify the result directory's
`checksums.sha256` before using summaries.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduction)

    handoff = f"""## EXP-100A current frontier

EXP-099A's native-order repair signal was followed by a pinned explicit
rectangular-FMM resource Gate. Current decision: `{d['decision']}`. Best direct
arithmetic ratio is `{percent(direct['arithmetic'], 12)}` and best free-transform
oracle ratio is `{percent(oracle['arithmetic'], 12)}`. The immediate constructive
frontier is `{gate_title}`.

Do not reinterpret this as a causal executor, 405B run, 8-GiB physical result,
or measured latency. The perfect future block and repair selector are oracle
grants and must be removed at the next applicable Gate.
"""
    replace_block(ROOT / "docs/research/VORTEX_RESEARCH_HANDOFF.md", handoff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result.resolve(), args.source_commit)
