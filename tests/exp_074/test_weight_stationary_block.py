from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from vortex_runtime.weight_stationary_block import (
    MoeActiveSet,
    WeightStationaryBudgetError,
    block_target_parameters,
    decimal_gb_to_gib,
    expected_independent_routed_union,
    maximum_distinct_routed_union,
    minimum_accepted_tokens,
    normalized_weight_traffic,
    search_minimum_block,
)


@pytest.fixture
def qwen122() -> MoeActiveSet:
    return MoeActiveSet(
        total_parameters=122_000_000_000,
        active_parameters=10_000_000_000,
        baseline_parameters=1_000_000_000,
        hidden_size=3072,
        expert_intermediate_size=1024,
        layer_count=48,
        expert_count=256,
        routed_experts_per_token=8,
        shared_experts_per_token=1,
    )


def test_qwen_active_decomposition_reconstructs_declared_10b(qwen122: MoeActiveSet) -> None:
    qwen122.validate()
    assert qwen122.parameters_per_expert_per_layer == 9_437_184
    assert qwen122.parameters_per_expert_model_wide == 452_984_832
    assert qwen122.single_token_active_expert_parameters == 4_076_863_488
    assert qwen122.nonexpert_active_parameters == 5_923_136_512
    assert block_target_parameters(qwen122, routed_expert_union=8) == 10_000_000_000


def test_route_union_models_have_expected_boundaries(qwen122: MoeActiveSet) -> None:
    assert expected_independent_routed_union(
        expert_count=256, routed_per_token=8, block_tokens=1
    ) == pytest.approx(8.0)
    assert expected_independent_routed_union(
        expert_count=256, routed_per_token=8, block_tokens=16
    ) > 90
    assert maximum_distinct_routed_union(
        expert_count=256, routed_per_token=8, block_tokens=64
    ) == 256


def test_optimistic_fixed_route_minimums(qwen122: MoeActiveSet) -> None:
    assert search_minimum_block(
        qwen122,
        route_model="fixed",
        allowed_multiplier=1.2,
        draft_parameters_per_token=0,
    ) == 9
    assert search_minimum_block(
        qwen122,
        route_model="fixed",
        allowed_multiplier=1.2,
        draft_parameters_per_token=800_000_000,
    ) == 25
    assert search_minimum_block(
        qwen122,
        route_model="fixed",
        allowed_multiplier=1.2,
        draft_parameters_per_token=1_000_000_000,
    ) == 50


def test_mtp1_fails_even_with_zero_draft_and_fixed_routes(qwen122: MoeActiveSet) -> None:
    fraction = normalized_weight_traffic(
        qwen122,
        accepted_tokens=1,
        routed_expert_union=8,
        draft_parameters_per_token=0,
    )
    assert fraction == 10.0
    assert fraction > 1.2


def test_route_diversity_moves_required_block_toward_one_hundred(qwen122: MoeActiveSet) -> None:
    independent = search_minimum_block(
        qwen122,
        route_model="independent_uniform_expected",
        allowed_multiplier=1.2,
    )
    distinct = search_minimum_block(
        qwen122,
        route_model="maximally_distinct",
        allowed_multiplier=1.2,
    )
    assert independent is not None and 64 < independent < 128
    assert distinct is not None and 64 < distinct < 128


def test_dense_405b_block_requirements_match_prior_equations() -> None:
    assert minimum_accepted_tokens(
        block_target_parameter_reads=405_000_000_000,
        baseline_parameters=4_000_000_000,
        allowed_multiplier=1.2,
    ) == 85
    assert minimum_accepted_tokens(
        block_target_parameter_reads=405_000_000_000,
        baseline_parameters=4_000_000_000,
        allowed_multiplier=1.2,
        draft_parameters_per_token=4_000_000_000,
    ) == 507


def test_storage_units_show_nominal_fit_but_insufficient_workspace() -> None:
    model_gib = decimal_gb_to_gib(81.0)
    free_bytes = 104_816_832_512
    remaining = free_bytes - 81_000_000_000
    assert model_gib == pytest.approx(75.43712854385376)
    assert remaining > 0
    assert remaining < 30 * 1024**3


def test_invalid_contracts_fail_closed(qwen122: MoeActiveSet) -> None:
    with pytest.raises(WeightStationaryBudgetError):
        expected_independent_routed_union(
            expert_count=8, routed_per_token=9, block_tokens=1
        )
    with pytest.raises(WeightStationaryBudgetError):
        block_target_parameters(qwen122, routed_expert_union=7)
    with pytest.raises(WeightStationaryBudgetError):
        minimum_accepted_tokens(
            block_target_parameter_reads=10,
            baseline_parameters=1,
            allowed_multiplier=1.2,
            draft_parameters_per_token=1.2,
        )
    with pytest.raises(WeightStationaryBudgetError):
        search_minimum_block(qwen122, route_model="unknown", allowed_multiplier=1.2)


def test_experiment_writes_complete_evidence_bundle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "exp_074_test_bundle"
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/exp_074/run_experiment.py"),
            "--config",
            str(root / "experiments/exp_074/config.json"),
            "--output-dir",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["authoritative_decision"].startswith("REVISE_MTP1")
    assert summary["DERIVED"]["mtp1_p50_gate_pass"] is False
    assert summary["DERIVED"]["fixed_route_zero_draft_p50_minimum_tokens"] == 9
    assert summary["claim_boundary"]["122b_download"] == "NOT_AUTHORIZED_OR_PERFORMED"
    assert (output / "logs/run.log").is_file()
    assert (output / "checksums.sha256").is_file()
