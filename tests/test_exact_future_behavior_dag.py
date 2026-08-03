from __future__ import annotations

from pathlib import Path

import pytest

from vortex_runtime.exact_future_behavior_dag import (
    BehaviorPath,
    build_exact_future_dag,
    depth_compression_frontier,
    evaluate_heldout_future_body,
    export_future_dag,
    quotient_frontier_point,
    replay_future_dag,
)


def path(
    prompt_id: str,
    tokens: tuple[int, ...],
    *,
    split: str = "compiled",
    duplicate_of: str | None = None,
) -> BehaviorPath:
    return BehaviorPath(
        prompt_id=prompt_id,
        token_ids=tokens,
        split=split,
        duplicate_of=duplicate_of,
    )


def test_backward_interning_builds_minimal_exact_suffix_dag() -> None:
    paths = [
        path("left", (1, 2, 3)),
        path("right", (9, 2, 3)),
    ]
    dag = build_exact_future_dag(paths, horizon=3)

    assert dag.node_count == 4
    assert dag.start_addresses[0] != dag.start_addresses[1]
    assert dag.suffix_to_address[(2, 3)] == dag.next_addresses[
        dag.start_addresses[0]
    ]
    assert dag.suffix_to_address[(2, 3)] == dag.next_addresses[
        dag.start_addresses[1]
    ]

    frontier = quotient_frontier_point(paths, horizon=3)
    assert frontier.raw_path_records == 6
    assert frontier.quotient_nodes == 4
    assert frontier.merged_records == 2
    assert frontier.compression_fraction == 1 / 3
    assert sum(
        point.unique_suffixes for point in frontier.depth_frontier
    ) == dag.node_count


def test_duplicate_path_shares_complete_start_address() -> None:
    paths = [
        path("source", (4, 5, 6, 7)),
        path(
            "duplicate",
            (4, 5, 6, 7),
            split="duplicate_control",
            duplicate_of="source",
        ),
    ]
    dag = build_exact_future_dag(paths, horizon=4)
    assert dag.node_count == 4
    assert dag.start_addresses[0] == dag.start_addresses[1]


def test_depth_frontier_counts_unique_suffixes_exactly() -> None:
    paths = [
        path("a", (1, 2, 3, 4)),
        path("b", (9, 2, 3, 4)),
        path("c", (8, 7, 3, 4)),
    ]
    points = depth_compression_frontier(paths, horizon=4)
    by_remaining = {
        point.remaining_tokens: point for point in points
    }
    assert by_remaining[1].unique_suffixes == 1
    assert by_remaining[2].unique_suffixes == 1
    assert by_remaining[3].unique_suffixes == 2
    assert by_remaining[4].unique_suffixes == 3
    assert sum(point.unique_suffixes for point in points) == 7


def test_compact40_export_replays_every_quotient_path(
    tmp_path: Path,
) -> None:
    paths = [
        path("a", (1, 2, 3, 4)),
        path("b", (9, 2, 3, 4)),
        path("c", (8, 7, 3, 4)),
    ]
    dag = build_exact_future_dag(paths, horizon=4)
    vm_path = tmp_path / "future.vtx"
    manifest_path = tmp_path / "future.json"
    build, manifest_bytes = export_future_dag(
        dag,
        vm_path=vm_path,
        manifest_path=manifest_path,
        metadata={"source": "synthetic"},
    )
    replay = replay_future_dag(dag, paths, vm_path=vm_path)

    assert build.format_name == "compact40"
    assert manifest_bytes == manifest_path.stat().st_size
    assert replay.paths == 3
    assert replay.exact_paths == 3
    assert replay.exact_tokens == 12
    assert replay.expected_tokens == 12
    assert replay.all_exact


def test_future_body_oracle_does_not_become_a_start_router() -> None:
    compiled = [
        path("a", (1, 2, 3, 4)),
        path("b", (9, 2, 3, 4)),
    ]
    dag = build_exact_future_dag(compiled, horizon=4)
    heldout = [
        path("heldout", (1, 2, 3, 4), split="heldout")
    ]
    coverage = evaluate_heldout_future_body(dag, heldout)

    assert coverage.full_continuation_hits == 1
    assert coverage.future_suffix_hits == 4
    assert coverage.future_suffix_coverage == 1.0
    assert coverage.causal_start_router_hits == 0
    assert coverage.causal_start_router_coverage == 0.0
    assert coverage.first_future_suffix_hit_positions == {"heldout": 0}


def test_codebook_overflow_is_rejected() -> None:
    paths = [
        path(f"p-{index}", (index,))
        for index in range(17)
    ]
    with pytest.raises(ValueError, match="codebook exceeds"):
        build_exact_future_dag(
            paths,
            horizon=1,
            codebook_limit=16,
        )
