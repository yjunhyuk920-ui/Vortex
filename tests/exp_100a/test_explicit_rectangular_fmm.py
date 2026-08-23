from __future__ import annotations

from dataclasses import replace
import io

import numpy as np

from experiments.exp_100a.run_experiment import load_catalog, validate_config
from vortex_runtime.explicit_rectangular_fmm import (
    SequenceState,
    exact_factorization,
    execute_factorization,
    expand_beam,
    git_blob_sha1,
    instantiate_family_candidate,
    orientations,
    prune_family_candidates,
    select_joint_plan,
    evaluate_sequence,
)


def strassen_222():
    # A=[a,b;c,d], B=[e,f;g,h]. Third factor emits vec(C^T).
    u = np.asarray(
        [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ],
        dtype=np.int64,
    )
    v = np.asarray(
        [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    w = np.asarray(
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    return exact_factorization("2,2,2-test", 2, 2, 2, u, v, w)


def test_config_contract() -> None:
    import json
    from pathlib import Path

    validate_config(
        json.loads(
            Path("experiments/exp_100a/config.json").read_text(encoding="utf-8")
        )
    )


def test_git_blob_identity_matches_git_object_contract() -> None:
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_strassen_tensor_and_execution_are_exact() -> None:
    factorization = strassen_222()
    assert factorization.rank == 7
    assert factorization.rank_ratio == 7 / 8
    for seed in range(8):
        rng = np.random.default_rng(seed)
        left = rng.integers(-8, 8, size=(2, 2), dtype=np.int64)
        right = rng.integers(-8, 8, size=(2, 2), dtype=np.int64)
        assert np.array_equal(
            execute_factorization(factorization, left, right), left @ right
        )


def test_all_six_tensor_orientations_are_exposed() -> None:
    rows = orientations(strassen_222())
    assert len(rows) == 6
    assert {row.permutation for row in rows} == {
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    }
    assert all(row.rank == 7 and row.classical_rank == 8 for row in rows)


def test_two_level_staged_counts_match_strassen_recurrence() -> None:
    step = orientations(strassen_222())[0]
    plan = evaluate_sequence(
        m=4,
        k=4,
        n=4,
        steps=(step, step),
        cut_depth=0,
        static_scalar_bytes=2,
        activation_bytes=2,
        output_bytes=2,
        accumulator_bytes=4,
    )
    assert plan.leaf_multiplications == 49
    assert plan.leaf_additions == 0
    assert plan.left_transform_additions == 5 * (4 + 7)
    assert plan.online_weight_transform_additions == 5 * (4 + 7)
    assert plan.output_transform_additions == 8 * (4 + 7)
    assert plan.free_transform_operations_without_repair == 49
    assert plan.free_transform_ratio_without_repair == 49 / (64 + 48)


def test_cut_depth_trades_static_stream_for_online_weight_transform() -> None:
    step = orientations(strassen_222())[0]
    plans = [
        evaluate_sequence(
            m=4,
            k=4,
            n=4,
            steps=(step, step),
            cut_depth=cut,
            static_scalar_bytes=2,
            activation_bytes=2,
            output_bytes=2,
            accumulator_bytes=4,
        )
        for cut in (0, 1, 2)
    ]
    zero, one, all_static = plans
    assert zero.transformed_weight_byte_ratio == 1.0
    assert one.transformed_weight_byte_ratio == 7 / 4
    assert all_static.transformed_weight_byte_ratio == 49 / 16
    assert (
        zero.online_weight_transform_additions
        > one.online_weight_transform_additions
        > all_static.online_weight_transform_additions
    )
    assert all_static.online_weight_transform_additions == 0


def test_static_transform_repair_charges_original_row_side_stream() -> None:
    step = orientations(strassen_222())[0]
    dynamic = evaluate_sequence(
        m=8,
        k=8,
        n=8,
        steps=(step,),
        cut_depth=0,
        static_scalar_bytes=2,
        activation_bytes=2,
        output_bytes=2,
        accumulator_bytes=4,
    )
    static = replace(dynamic, cut_depth=1, transformed_weight_byte_ratio=7 / 4)
    dynamic_candidate = instantiate_family_candidate(
        family="x",
        rows=8,
        columns=8,
        count=1,
        block_length=8,
        repair_fraction=0.1,
        plan=dynamic,
        lossless_ratio=2.0,
    )
    static_candidate = instantiate_family_candidate(
        family="x",
        rows=8,
        columns=8,
        count=1,
        block_length=8,
        repair_fraction=0.1,
        plan=static,
        lossless_ratio=2.0,
    )
    assert dynamic_candidate.repair_streamed_bytes_per_block == 0
    assert static_candidate.repair_streamed_bytes_per_block == 6.4


def test_beam_can_compose_mixed_exact_schemes() -> None:
    schemes = orientations(strassen_222())[:2]
    start = (SequenceState((), 1, 1, 1, 1, 0),)
    depth1 = expand_beam(
        states=start,
        schemes=schemes,
        m=8,
        k=8,
        n=8,
        maximum_padding_multiplier=2.0,
        per_split_survivors=2,
        beam_width=8,
    )
    depth2 = expand_beam(
        states=depth1,
        schemes=schemes,
        m=8,
        k=8,
        n=8,
        maximum_padding_multiplier=2.0,
        per_split_survivors=2,
        beam_width=8,
    )
    assert depth1 and depth2
    assert all(row.depth == 2 for row in depth2)
    assert all(
        (row.a_product, row.b_product, row.c_product) == (4, 4, 4)
        for row in depth2
    )


def test_joint_dp_respects_io_budget_and_picks_lower_arithmetic() -> None:
    step = orientations(strassen_222())[0]
    base = evaluate_sequence(
        m=8,
        k=8,
        n=8,
        steps=(step,),
        cut_depth=0,
        static_scalar_bytes=2,
        activation_bytes=2,
        output_bytes=2,
        accumulator_bytes=4,
    )
    low_compute_high_io = replace(
        base,
        transformed_weight_byte_ratio=1.5,
        online_operations_without_repair=100,
    )
    high_compute_low_io = replace(
        base,
        transformed_weight_byte_ratio=0.5,
        online_operations_without_repair=200,
    )

    def options(name: str):
        candidates = [
            instantiate_family_candidate(
                family=name,
                rows=8,
                columns=8,
                count=1,
                block_length=8,
                repair_fraction=0.0,
                plan=plan,
                lossless_ratio=1.0,
            )
            for plan in (low_compute_high_io, high_compute_low_io)
        ]
        return prune_family_candidates(
            candidates, maximum_workspace_bytes=2**30, maximum_candidates=8
        )

    plan = select_joint_plan(
        block_length=8,
        plan_kind="test",
        candidates_by_family={"a": options("a"), "b": options("b")},
        io_fraction_limit=0.10,
        io_bins=1024,
    )
    assert plan is not None
    assert plan.io_fraction <= 0.10
    assert all(
        row.plan.transformed_weight_byte_ratio == 0.5
        for row in plan.family_candidates
    )


def test_runner_parses_official_style_npz_object_values() -> None:
    factorization = strassen_222()
    archive = io.BytesIO()
    value = np.empty(3, dtype=object)
    value[:] = [factorization.u, factorization.v, factorization.w]
    np.savez_compressed(archive, **{"2,2,2": value})
    rows, manifest, controls = load_catalog(
        archive.getvalue(),
        maximum_coefficient_absolute_value=2,
        controls_per_factorization=2,
        seed=1,
    )
    assert len(rows) == 1
    assert manifest[0]["status"] == "eligible"
    assert len(controls) == 2
    assert all(row["exact"] for row in controls)


def test_ledger_updater_is_idempotent_and_marks_every_required_file(tmp_path) -> None:
    import json
    import experiments.exp_100a.update_ledgers as updater

    for relative in updater.MANDATORY_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")

    result = {
        "authoritative_decision": "RETAIN_RECTANGULAR_RANK_HEADROOM_REQUIRE_EXPLICIT_TRANSFORM_CIRCUIT",
        "deterministic_core_sha256": "a" * 64,
        "integrity_failures": [],
        "catalog_summary": {
            "eligible_factorization_count": 3,
            "catalog_key_count": 5,
            "pareto_orientation_count": 8,
            "synthetic_control_count": 6,
            "synthetic_control_mismatch_count": 0,
        },
        "direct_ten_x_pass_blocks": [],
        "free_transform_ten_x_pass_blocks": [1024],
        "direct_final_p50_pass_blocks": [],
        "best_direct_row": {
            "block_length": 1024,
            "direct": {
                "arithmetic_ratio": 0.25,
                "io_fraction": 0.001,
                "maximum_workspace_bytes": 1234,
                "total_compiled_static_bytes": 5678,
                "ten_x_arithmetic_pass": False,
                "p50_arithmetic_pass": False,
                "p95_arithmetic_pass": False,
                "p50_io_pass": True,
                "workspace_pass": True,
            },
        },
        "best_free_transform_oracle_row": {
            "block_length": 1024,
            "free_transform_oracle": {
                "arithmetic_ratio": 0.05,
                "io_fraction": 0.001,
                "maximum_workspace_bytes": 1234,
                "total_compiled_static_bytes": 5678,
                "ten_x_arithmetic_pass": True,
                "p50_arithmetic_pass": False,
                "p95_arithmetic_pass": False,
                "p50_io_pass": True,
                "workspace_pass": True,
            },
        },
        "catalog_identity": {"commit": "c", "git_blob_sha1": "b"},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    old_root = updater.ROOT
    updater.ROOT = tmp_path
    try:
        updater.update(result_path, "deadbeef")
        first = {
            relative: (tmp_path / relative).read_text(encoding="utf-8")
            for relative in updater.MANDATORY_PATHS
        }
        updater.update(result_path, "deadbeef")
        second = {
            relative: (tmp_path / relative).read_text(encoding="utf-8")
            for relative in updater.MANDATORY_PATHS
        }
    finally:
        updater.ROOT = old_root

    assert first == second
    assert (tmp_path / "docs/research/EXP_100A_LATEST_RESULT.md").exists()
    for text in second.values():
        assert text.count(updater.START) == 1
        assert text.count(updater.END) == 1


def test_resource_empty_shape_frontier_is_diagnostic_not_integrity_failure() -> None:
    from experiments.exp_100a.run_experiment import (
        SearchResult,
        search_coverage_diagnostics,
    )

    empty = SearchResult(
        block_length=16384,
        rows=128256,
        columns=16384,
        state_count_by_depth=(96, 96, 96, 96, 0),
        direct_structural_plans=(),
        oracle_structural_plans=(),
    )
    diagnostics = search_coverage_diagnostics(
        {(16384, 128256, 16384): empty}
    )
    assert diagnostics["search_count"] == 1
    assert diagnostics["empty_direct_count"] == 1
    assert diagnostics["empty_oracle_count"] == 1
    assert diagnostics["empty_direct"][0]["rows"] == 128256
    assert "not a control failure" in diagnostics["empty_frontier_interpretation"]
