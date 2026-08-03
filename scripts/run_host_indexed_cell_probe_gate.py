from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.host_indexed_cell_probe import (
    PointerTableConfig,
    balanced_resident_cache,
    benchmark_packed_pointer_chase,
    build_disjoint_pointer_table,
    cache_frontier_point,
    evaluate_adversarial_pair,
    sampled_resident_cache,
    target_cell_probe_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Experiment 043 host-indexed exact-decision "
            "cell-probe Gate."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/host_indexed_cell_probe_gate.json"
        ),
    )
    parser.add_argument("--benchmark-cells", type=int, default=262_144)
    parser.add_argument("--benchmark-steps", type=int, default=200_000)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PointerTableConfig(chains=64, steps=256)
    table = build_disjoint_pointer_table(config)

    capacities = [
        0,
        64,
        640,
        4096,
        8192,
        config.cells,
    ]
    balanced_frontier = []
    for capacity in capacities:
        cache = balanced_resident_cache(config, capacity)
        balanced_frontier.append(
            cache_frontier_point(
                table,
                cache_capacity_records=capacity,
                resident_records=cache,
            ).to_dict()
        )

    sampled_frontier = []
    sampled_capacity = 4097
    for seed in range(8):
        cache = sampled_resident_cache(
            config,
            sampled_capacity,
            seed=seed,
        )
        point = cache_frontier_point(
            table,
            cache_capacity_records=sampled_capacity,
            resident_records=cache,
        )
        item = point.to_dict()
        item["seed"] = seed
        sampled_frontier.append(item)

    adversaries = [
        evaluate_adversarial_pair(
            table,
            chain=17,
            step=step,
        ).to_dict()
        for step in (0, 127, 255)
    ]

    timing = benchmark_packed_pointer_chase(
        cells=args.benchmark_cells,
        steps_per_repeat=args.benchmark_steps,
        repeats=args.benchmark_repeats,
        seed=43,
    )
    target = target_cell_probe_projection()

    balanced_pass = all(
        item["theorem_pass"] for item in balanced_frontier
    )
    sampled_pass = all(
        item["theorem_pass"] for item in sampled_frontier
    )
    adversary_pass = all(item["passes"] for item in adversaries)
    certificate_passes = bool(
        balanced_pass
        and sampled_pass
        and adversary_pass
        and target.worst_chain_host_miss_lower_bound == 249
        and timing.logical_probes_per_step == 1
    )

    payload = {
        "experiment": "043_host_indexed_cell_probe_gate",
        "evidence_level": (
            "E1 explicit-pointer cell-probe theorem and "
            "nonrepresentative host prototype"
        ),
        "micro_table": {
            "chains": config.chains,
            "steps": config.steps,
            "cells": config.cells,
            "address_bits": config.address_bits,
            "record_bits": config.record_bits,
        },
        "balanced_cache_frontier": balanced_frontier,
        "sampled_cache_frontier": sampled_frontier,
        "indistinguishable_record_adversaries": adversaries,
        "host_prototype": timing.to_dict(),
        "target_projection": target.to_dict(),
        "scope_separation": {
            "explicit_pointer_serial_dependency_proven": certificate_passes,
            "raw_complete_record_cache_lower_bound_proven": certificate_passes,
            "one_logical_record_probe_per_token_is_sufficient_for_model": True,
            "arbitrary_compressed_cache_lower_bound_proven": False,
            "minimum_physical_transaction_bytes_proven": False,
            "target_host_or_pcie_latency_proven": False,
            "ci_timing_is_target_representative": False,
            "host_indexed_escape_closed": False,
            "fixed_runtime_target_fully_contradicted": False,
            "real_405b_execution_performed": False,
        },
        "conclusion": {
            "certificate_passes": certificate_passes,
            "target_worst_chain_serial_host_misses": (
                target.worst_chain_host_miss_lower_bound
            ),
            "target_chain_tokens": target.chain_steps,
            "target_host_miss_fraction": (
                target.worst_chain_host_miss_fraction
            ),
            "target_logical_host_bytes_per_token": (
                target.logical_host_bytes_per_token_lower_bound
            ),
            "explicit_pointer_host_storage_gib": (
                target.explicit_pointer_table_gib
            ),
            "resident_cache_gib": target.resident_cache_gib,
            "cell_probe_interpretation": (
                "near-one serial host miss per token is unavoidable for "
                "the explicit raw-record pointer representation on a "
                "worst-case prompt, but only about 4.86 logical bytes/token "
                "are forced; probe count alone does not violate the "
                "4B-class latency target"
            ),
            "fixed_target_status": (
                "unsolved: the host-indexed path remains algorithmically "
                "possible and requires real construction plus CPU/GPU "
                "latency and locality measurements"
            ),
        },
        "next_obligation": (
            "build a charged host-indexed exact-decision VM prototype with "
            "mmap or pinned host memory and GPU-facing asynchronous lookup, "
            "or derive a stronger family requiring many information probes "
            "per generated token rather than one small dependent record"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
