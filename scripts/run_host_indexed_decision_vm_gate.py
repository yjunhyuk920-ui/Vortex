from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.host_indexed_cell_probe import (
    PointerTableConfig,
    build_disjoint_pointer_table,
)
from vortex_runtime.host_indexed_decision_vm import (
    FORMAT_ALIGNED64,
    FORMAT_COMPACT40,
    benchmark_format,
    target_decision_vm_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and benchmark the Experiment 044 mmap-backed "
            "host-indexed exact-decision VM."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/host_indexed_decision_vm_gate.json"
        ),
    )
    parser.add_argument("--chains", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--address-samples", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.chains, args.steps, args.address_samples) <= 0:
        raise ValueError("benchmark dimensions must be positive")

    config = PointerTableConfig(
        chains=args.chains,
        steps=args.steps,
    )
    table = build_disjoint_pointer_table(config)
    target = target_decision_vm_projection()

    with tempfile.TemporaryDirectory(
        prefix="vortex-decision-vm-"
    ) as temporary_directory:
        directory = Path(temporary_directory)
        compact = benchmark_format(
            table,
            directory / "decision-compact40.vtx",
            flags=FORMAT_COMPACT40,
            address_samples=args.address_samples,
            seed=44,
        )
        aligned = benchmark_format(
            table,
            directory / "decision-aligned64.vtx",
            flags=FORMAT_ALIGNED64,
            address_samples=args.address_samples,
            seed=44,
        )

    functional_pass = bool(
        compact.checksum_verified
        and aligned.checksum_verified
        and compact.first_replay_cache_misses == args.steps
        and compact.cached_replay_cache_hits == args.steps
        and aligned.first_replay_cache_misses == args.steps
        and aligned.cached_replay_cache_hits == args.steps
        and compact.build.atomic_replace
        and aligned.build.atomic_replace
        and compact.build.temporary_file_removed
        and aligned.build.temporary_file_removed
    )

    payload = {
        "experiment": "044_host_indexed_decision_vm",
        "evidence_level": (
            "E1/E2 constructive mmap VM and nonrepresentative CPU "
            "benchmark"
        ),
        "table": {
            "chains": config.chains,
            "steps": config.steps,
            "records": config.cells,
            "source_record_bits": config.record_bits,
        },
        "formats": [compact.to_dict(), aligned.to_dict()],
        "target_projection": target.to_dict(),
        "scope_separation": {
            "atomic_binary_builder_proven": functional_pass,
            "format_corruption_tests_proven": functional_pass,
            "mmap_exact_replay_proven": functional_pass,
            "bounded_cache_accounting_proven": functional_pass,
            "compact_and_aligned_formats_compared": True,
            "os_page_cache_state_controlled": False,
            "ci_timing_is_target_representative": False,
            "timing_projected_to_405b": False,
            "released_model_decision_compiler_exists": False,
            "gpu_or_pinned_memory_integration_exists": False,
            "real_405b_execution_performed": False,
        },
        "conclusion": {
            "functional_vm_passes": functional_pass,
            "compact_file_bytes": compact.build.file_bytes,
            "aligned_file_bytes": aligned.build.file_bytes,
            "compact_dependent_p50_ns": compact.dependent.p50_ns,
            "compact_dependent_p99_ns": compact.dependent.p99_ns,
            "aligned_dependent_p50_ns": aligned.dependent.p50_ns,
            "aligned_dependent_p99_ns": aligned.dependent.p99_ns,
            "compact_cached_replay_ns_per_token": (
                compact.cached_replay_ns_per_token
            ),
            "aligned_cached_replay_ns_per_token": (
                aligned.cached_replay_ns_per_token
            ),
            "architecture_status": (
                "portable host-indexed exact pointer VM is functionally "
                "viable; advance to pinned-memory/GPU-facing lookup and "
                "a real checkpoint decision-index compiler"
                if functional_pass
                else "VM Gate failed and requires repair"
            ),
            "fixed_target_status": (
                "unsolved: CPU mmap viability does not establish a "
                "universal decision compiler, target GPU latency, or "
                "real 405B quality preservation"
            ),
        },
        "next_obligation": (
            "implement a pinned-host-memory or shared-memory lookup bridge "
            "with batched and dependent GPU-facing requests, while separately "
            "testing whether exact decision records can be compiled from an "
            "unmodified released transformer checkpoint"
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
