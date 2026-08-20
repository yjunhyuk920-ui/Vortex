from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-088B:START -->"
END = "<!-- EXP-088B:END -->"


def fmt_fraction(value: float) -> str:
    return f"{100.0 * float(value):.9f}%"


def replace_block(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"mandatory ledger missing: {path}")
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed EXP-088B marker block in {path}")
        prefix, remainder = text.split(START, 1)
        _old, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def decision_details(result: dict[str, Any]) -> dict[str, Any]:
    selected = result["shared_build_selection"]["selected"]
    projection = result["optimistic_405b_projection"]
    gates = result["gates"]
    controls = result["controls"]
    return {
        "decision": result["authoritative_decision"],
        "mask": int(selected["kept_mask"]),
        "program_sha256": selected["program_sha256"],
        "retained_fraction": float(selected["retained_fraction"]),
        "retained_bytes": int(selected["retained_linear_bytes"]),
        "total_bytes": int(selected["total_linear_bytes"]),
        "shared_count": int(
            result["shared_build_selection"]["shared_exact_program_count"]
        ),
        "holdout_exact": bool(
            gates["selected_program_exact_on_all_unseen_holdout_states"]
        ),
        "gate_passed": bool(gates["oracle_program_sharing_gate_passed"]),
        "integrity": bool(gates["integrity_passed"]),
        "projected_hot_bytes": int(projection["ideal_hot_bytes_per_token"]),
        "projected_hot_pass": bool(projection["hot_byte_target_passed"]),
        "empty_holdout_exact_count": int(
            controls["empty_mask_exact_holdout_state_count"]
        ),
    }


def latest_result_markdown(
    result: dict[str, Any], result_path: Path, source_commit: str
) -> str:
    d = decision_details(result)
    holdout_rows = result["holdout_rows"]
    selected_row = next(
        row for row in holdout_rows if row["kept_mask"] == d["mask"]
    )
    holdout_total = len(result["state_traces"]) - len(
        result["shared_build_selection"]["build_state_ids"]
    )
    return f"""# EXP-088B latest result — oracle cross-layer program-sharing Gate

## Identity

- source commit: `{source_commit}`
- raw result: `{result_path.as_posix()}`
- public checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['revision']}`
- selected complete layer window: `{result['checkpoint']['layer_indices']}`
- mechanism fingerprint: `{result['mechanism_fingerprint']}`

## Authoritative decision

`{d['decision']}`

## Measured DEV-W result

- integrity controls: `{d['integrity']}`
- build states: `{len(result['shared_build_selection']['build_state_ids'])}` distinct states
- untouched holdout states: `{selected_row['exact_holdout_state_count']} / {holdout_total}` exact under the selected program
- exhaustive program language: `{result['oracle_contract']['search_space_size']}` masks
- exact programs shared by all build states: `{d['shared_count']}`
- selected mask: `{d['mask']:#04x}`
- selected program SHA-256: `{d['program_sha256']}`
- selected retained linear bytes: `{d['retained_bytes']:,} / {d['total_bytes']:,}`
- retained fraction: `{fmt_fraction(d['retained_fraction'])}`
- empty-program exact holdout states: `{d['empty_holdout_exact_count']}`
- oracle program-sharing Gate: `{d['gate_passed']}`

## Optimistic target projection

The projection grants that every 405B checkpoint linear page obtains the same retained fraction, while program metadata, non-linear state traffic, addressing, materialization, and sparse-kernel overhead are free.

- projected hot bytes/token: `{d['projected_hot_bytes']:,}`
- registered hot-byte limit: `100,000,000`
- projected hot-byte Gate: `{d['projected_hot_pass']}`

This is `PROJECTED`, not a 405B or target-GPU measurement.

## Claim boundary

This experiment is a non-deployable oracle upper-bound over one frozen program language. The official DEV-W graph executed two adjacent complete Transformer layers, and exactness covered the pair output hidden state plus both layers' complete post-step K/V caches. The oracle evaluator intentionally used dense official operations to decide whether a sparse static program is worth implementing; its wall time is not an execution speed claim. TARGET-W, an actual sparse kernel, 128-step continuation, 8-GiB VRAM, and 4B-class p50/p95 remain `NOT TESTED`.
"""


def update(result_path: Path, source_commit: str) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    d = decision_details(result)
    latest = latest_result_markdown(result, result_path, source_commit)
    latest_path = ROOT / "docs/research/EXP_088B_LATEST_RESULT.md"
    latest_path.write_text(latest, encoding="utf-8")

    if d["gate_passed"]:
        next_action = (
            "Freeze the selected program before any new prompts, implement a complete-layer selector-free executor, "
            "and run at least 128 consecutive exact decode transitions with full successor-state comparison."
        )
        failure_line = (
            "The page-mask language was not rejected in EXP-088B; no failed-family entry is registered."
        )
    else:
        next_action = (
            "Do not sweep page-family count, tile size, layer pair, prompt set, or deletion order. The next admissible "
            "decompiler primitive must replace or fuse checkpoint pages with an exact static computation, rather than "
            "only retain/zero them, and must repeat the same build-only selection plus untouched-holdout state Gate."
        )
        failure_line = (
            "EXP-088B rejects the exhaustive eight-family cross-layer page-mask microprogram as a 405B core under its "
            "frozen scope. Reopening requires a richer exact program instruction that changes the information source, "
            "not a finer mask sweep."
        )

    state_block = f"""## EXP-088B — checkpoint-static cross-layer program-sharing Gate

- decision: `{d['decision']}`
- evidence: E2/E3 small-real-checkpoint oracle falsification; TARGET-W remains `NOT TESTED`
- selected static mask: `{d['mask']:#04x}` (`{fmt_fraction(d['retained_fraction'])}` of selected-window linear bytes retained)
- selected program exact on every untouched holdout state: `{d['holdout_exact']}`
- oracle program-sharing Gate: `{d['gate_passed']}`
- optimistic projected 405B hot bytes/token: `{d['projected_hot_bytes']:,}` (`PROJECTED`)
- raw evidence: `{result_path.as_posix()}`
- source commit: `{source_commit}`

The program was selected only from three build states after exhaustive enumeration of all `2^8` masks. One unchanged mask was then evaluated on three distinct unseen states. Exactness covered final pair hidden bytes and complete new K/V state for both adjacent official decoder layers. No state ID, state hash, output literal, or future generated token is stored in the program.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state_block)

    next_block = f"""## EXP-088B handoff

Current decision: `{d['decision']}`.

{next_action}

Do not call the dense oracle evaluator an executor. A promotion requires an existing-ISA runtime whose measured online bytes, MACs, state traffic, and exact 128-step behavior are charged.
"""
    replace_block(ROOT / "NEXT_EXPERIMENT.md", next_block)

    decision_block = f"""## EXP-088B decision

`{d['decision']}`

The frozen oracle searched all 256 checkpoint-static masks over eight cross-layer page families and chose the minimum-byte program in the intersection of the build-state exact sets. The selected mask retained `{fmt_fraction(d['retained_fraction'])}` and its exactness on untouched holdout states was `{d['holdout_exact']}`. The registered promotion threshold was 25%, and the final 405B hot-byte target remained independently enforced through the optimistic projection.
"""
    replace_block(ROOT / "DECISION_LOG.md", decision_block)

    failed_block = f"""## EXP-088B — cross-layer page-mask program sharing

{failure_line}

Frozen fingerprint: two adjacent complete DEV-W layers; all linear weights tiled `64x64`; eight deterministic page families spanning both layers; all 256 masks; final pair hidden plus both layers' complete K/V; three build and three untouched holdout states; no selector and no runtime fallback credited.
"""
    replace_block(ROOT / "FAILED_APPROACHES.md", failed_block)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failed_block)

    assumption_block = f"""## EXP-088B assumptions and grants

- `MEASURED`: exact DEV-W transition equality and exhaustive mask outcomes.
- `GRANTED`: dense oracle search, zero-cost page addressing, zero-cost program metadata, and ideal sparse execution of retained tiles.
- `PROJECTED`: `{d['projected_hot_bytes']:,}` hot bytes/token for 405B under uniform generalization of the measured retained fraction.
- `UNVERIFIED`: a sparse kernel can reproduce the dense-zero semantic evaluator bitwise; TARGET-W shares the same program structure; target hardware latency/VRAM.
- The oracle may inspect build reference successor states offline. Holdout targets are not used to choose the program.
"""
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", assumption_block)

    validation_block = f"""## EXP-088B validation row

| Boundary | Exact contract | Result | Evidence level |
|---|---|---|---|
| Two adjacent official DEV-W decoder layers | pair hidden + complete K/V for both layers, byte equality | `{d['decision']}` | E2 |
| Same selected program on three untouched states | no selector, no output/state literal | `{d['holdout_exact']}` | E3 oracle Gate |
| Actual sparse existing-ISA executor | full online resource trace | `NOT TESTED` | — |
| 405B / 8-GiB / 4B-class p50/p95 | full target contract | `NOT TESTED` | — |
"""
    replace_block(ROOT / "VALIDATION_MATRIX.md", validation_block)

    architecture_block = f"""## EXP-088B architecture status

The page-mask object is an offline oracle program descriptor, not an accepted runtime component. It has no causal selector: one checkpoint-static mask is frozen from build states and applied unchanged to holdout states. Architecture promotion is `{d['gate_passed']}`. Until a full-layer existing-ISA executor passes 128 exact steps, the production architecture remains unchanged.
"""
    replace_block(ROOT / "ARCHITECTURE.md", architecture_block)

    hardware_block = f"""## EXP-088B hardware status

No target GPU run was performed. Oracle evaluator timing on GitHub CPU is not a speed record. The only target-scale number is the explicitly optimistic `PROJECTED` hot-byte value `{d['projected_hot_bytes']:,}` bytes/token; target VRAM, SSD, PCIe, HBM, TTFT, and p50/p95 remain `NOT TESTED`.
"""
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", hardware_block)

    reproduce_block = f"""## EXP-088B reproduction

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
pytest -q tests/exp_088b
python experiments/exp_088b/run_experiment.py \\
  --config experiments/exp_088b/config.json \\
  --output-dir results/exp_088b/<source-commit>
```

Checkpoint: `{result['checkpoint']['model_id']}@{result['checkpoint']['revision']}`. Source commit: `{source_commit}`. Verify `checksums.sha256` before reading derived summaries.
"""
    replace_block(ROOT / "REPRODUCIBILITY.md", reproduce_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
