"""Throwaway interactive shell for EXP-080A.

Question: after granting a perfect future activation block, does exact
rectangular arithmetic fit the final VORTEX traffic and compute fractions?
The pure state logic lives in :mod:`vortex_runtime.hyperblock_fmm`; this shell
only exposes it for manual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from vortex_runtime.hyperblock_fmm import (
    TensorFamily,
    evaluate_block,
    target_equivalent_fraction,
)


ROOT = Path(__file__).resolve().parents[1]
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def load_state() -> tuple[dict, tuple[TensorFamily, ...]]:
    config = json.loads(
        (ROOT / "experiments/exp_080a/config.json").read_text(encoding="utf-8")
    )
    excluded = set(config["candidate"]["exclude_tensor_families"])
    families = []
    shape_path = ROOT / config["registered_inputs"]["target_shape_path"]
    for line in shape_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["tensor"] not in excluded:
            families.append(
                TensorFamily(
                    name=row["tensor"],
                    rows=int(row["rows"]),
                    columns=int(row["columns"]),
                    count=int(row["count"]),
                )
            )
    return config, tuple(families)


def evaluate(config: dict, families: tuple[TensorFamily, ...], state: dict):
    contract = config["target_contract"]
    candidate = config["candidate"]
    allowed = target_equivalent_fraction(
        baseline_billions=float(contract["baseline_billions"]),
        target_billions=float(contract["target_billions"]),
        latency_multiple=float(contract["p50_latency_multiple"]),
    )
    return evaluate_block(
        families,
        block_length=state["block_length"],
        target_sweeps=state["target_sweeps"],
        proposal_billions=state["proposal_billions"],
        target_billions=float(contract["target_billions"]),
        allowed_fraction=allowed,
        peak_vram_bytes=int(contract["peak_vram_bytes"]),
        omega=float(candidate["theoretical_omega"]),
        q4_weight_bytes_per_scalar=float(candidate["q4_weight_bytes_per_scalar"]),
        activation_bytes_per_scalar=int(candidate["activation_bytes_per_scalar"]),
        scratch_bytes_per_scalar=int(candidate["scratch_bytes_per_scalar"]),
        scratch_square_tile_count=int(candidate["scratch_square_tile_count"]),
    )[0]


def _percent(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def render(config: dict, families: tuple[TensorFamily, ...], state: dict) -> str:
    row = evaluate(config, families, state)
    mode = state["display_mode"]
    displayed_fraction = (
        row.constructive_total_fraction
        if mode == "constructive_strassen"
        else row.omega_oracle_total_fraction
    )
    displayed_pass = (
        row.constructive_joint_pass
        if mode == "constructive_strassen"
        else row.omega_oracle_joint_pass
    )
    contract = config["target_contract"]
    lines = [
        f"{BOLD}EXP-080A Hyperblock arithmetic prototype{RESET}",
        f"{DIM}Perfect future activations are a non-deployable oracle.{RESET}",
        "",
        f"{BOLD}Current state{RESET}",
        f"block_length:                 {state['block_length']}",
        f"target_sweeps:                {state['target_sweeps']}",
        f"proposal_billions/token:      {state['proposal_billions']}",
        f"display_mode:                 {mode}",
        f"registered_tensor_families:   {len(families)}",
        f"p50_allowed_fraction:         {_percent(float(contract['p50_target_fraction']))}",
        "",
        f"{BOLD}Derived result{RESET}",
        f"traffic_fraction:             {_percent(row.traffic_fraction)}",
        f"constructive_arithmetic:      {_percent(row.constructive_total_fraction)}",
        f"unit-constant omega oracle:   {_percent(row.omega_oracle_total_fraction)}",
        f"displayed_fraction:           {_percent(displayed_fraction)}",
        f"constructive speedup:         {1.0 / row.constructive_ratio:.3f}x",
        f"required packing after cost:  {row.required_constructive_packing_factor:.3f}x",
        f"favorable workspace:          {row.favorable_workspace_bytes / 2**30:.3f} GiB",
        f"traffic pass:                 {row.constructive_traffic_pass}",
        f"constructive arithmetic pass: {row.constructive_arithmetic_pass}",
        f"workspace pass:               {row.workspace_pass}",
        f"displayed joint pass:         {displayed_pass}",
        "",
        f"{BOLD}Controls{RESET}",
        f"{BOLD}[n]{RESET} next K  {BOLD}[p]{RESET} previous K  "
        f"{BOLD}[s]{RESET} add sweep  {BOLD}[z]{RESET} reset sweep",
        f"{BOLD}[d]{RESET} toggle 0B/4B proposal  {BOLD}[a]{RESET} toggle arithmetic  "
        f"{BOLD}[q]{RESET} quit",
    ]
    return "\n".join(lines)


def batch_table(config: dict, families: tuple[TensorFamily, ...]) -> None:
    print(
        "K\ttraffic%\tstrassen%\tomega%\tstrassen_speedup\tworkspace_GiB\tjoint"
    )
    for block_length in config["candidate"]["block_lengths"]:
        state = {
            "block_length": int(block_length),
            "target_sweeps": float(config["candidate"]["authoritative_target_sweeps"]),
            "proposal_billions": float(
                config["candidate"]["authoritative_proposal_billions"]
            ),
            "display_mode": "constructive_strassen",
        }
        row = evaluate(config, families, state)
        print(
            f"{block_length}\t{100*row.traffic_fraction:.6f}\t"
            f"{100*row.constructive_total_fraction:.6f}\t"
            f"{100*row.omega_oracle_total_fraction:.6f}\t"
            f"{1/row.constructive_ratio:.3f}\t"
            f"{row.favorable_workspace_bytes/2**30:.3f}\t"
            f"{row.constructive_joint_pass}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    arguments = parser.parse_args()
    config, families = load_state()
    if arguments.batch or not sys.stdin.isatty():
        batch_table(config, families)
        return
    block_lengths = [int(value) for value in config["candidate"]["block_lengths"]]
    index = 0
    state = {
        "block_length": block_lengths[index],
        "target_sweeps": 1.0,
        "proposal_billions": 0.0,
        "display_mode": "constructive_strassen",
    }
    while True:
        print("\x1b[2J\x1b[H", end="")
        print(render(config, families, state))
        command = input("\n> ").strip()
        if command == "q":
            return
        if command == "n":
            index = min(index + 1, len(block_lengths) - 1)
        elif command == "p":
            index = max(index - 1, 0)
        elif command == "s":
            state["target_sweeps"] += 1.0
        elif command == "z":
            state["target_sweeps"] = 1.0
        elif command == "d":
            state["proposal_billions"] = (
                0.0 if state["proposal_billions"] else 4.0
            )
        elif command == "a":
            state["display_mode"] = (
                "unit_constant_omega_oracle"
                if state["display_mode"] == "constructive_strassen"
                else "constructive_strassen"
            )
        state["block_length"] = block_lengths[index]


if __name__ == "__main__":
    main()
