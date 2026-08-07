import math

import pytest

from vortex_runtime.tangent_macroblock import (
    aggregate_quality_rows,
    charged_cycle_fraction,
    dense_macroblock_cost,
    gate_decision,
    minimum_hot_tokens,
    valid_prefix_length,
)


def test_registered_dense_costs_and_integer_lifetimes():
    small = dense_macroblock_cost(
        hidden_size=1024, intermediate_size=3584, active_paths=1
    )
    assert small["hot_fraction_of_exact"] == pytest.approx(2 / 21)
    assert small["materialization_exact_token_equivalents"] == pytest.approx(
        1024 / 3
    )
    assert minimum_hot_tokens(
        allowance_fraction=0.12,
        hot_fraction=float(small["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            small["materialization_exact_token_equivalents"]
        ),
    ) == 13821
    assert minimum_hot_tokens(
        allowance_fraction=0.15,
        hot_fraction=float(small["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            small["materialization_exact_token_equivalents"]
        ),
    ) == 6249

    surrogate = dense_macroblock_cost(
        hidden_size=3072, intermediate_size=1024, active_paths=9
    )
    assert surrogate["hot_fraction_of_exact"] == pytest.approx(1 / 9)
    assert minimum_hot_tokens(
        allowance_fraction=0.12,
        hot_fraction=float(surrogate["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            surrogate["materialization_exact_token_equivalents"]
        ),
    ) == 115299


def test_minimum_lifetime_is_the_first_passing_integer():
    cost = dense_macroblock_cost(
        hidden_size=1024, intermediate_size=3584, active_paths=1
    )
    required = minimum_hot_tokens(
        allowance_fraction=0.12,
        hot_fraction=float(cost["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            cost["materialization_exact_token_equivalents"]
        ),
    )
    assert required is not None
    assert charged_cycle_fraction(
        hot_tokens=required,
        hot_fraction=float(cost["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            cost["materialization_exact_token_equivalents"]
        ),
    ) <= 0.12
    assert charged_cycle_fraction(
        hot_tokens=required - 1,
        hot_fraction=float(cost["hot_fraction_of_exact"]),
        construction_token_equivalents=float(
            cost["materialization_exact_token_equivalents"]
        ),
    ) > 0.12


def test_exact_small_matrix_macro_equals_factorized_expression():
    np = pytest.importorskip("numpy")
    wd = np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    wu = np.array([[2.0, 1.0], [-1.0, 4.0], [3.0, -0.5]])
    coefficient = np.array([0.2, -1.5, 2.0])
    x = np.array([0.75, -2.0])
    macro = wd @ np.diag(coefficient) @ wu
    factorized = wd @ (coefficient * (wu @ x))
    assert np.allclose(macro @ x, factorized, rtol=0.0, atol=1e-12)


def test_prefix_stops_at_first_top1_or_kl_failure():
    assert valid_prefix_length(
        target_top_tokens=[1, 2, 3],
        candidate_top_tokens=[1, 2, 3],
        token_kls=[0.0, 0.01, 0.02],
        max_token_kl=0.05,
    ) == 3
    assert valid_prefix_length(
        target_top_tokens=[1, 2, 3],
        candidate_top_tokens=[1, 9, 3],
        token_kls=[0.0, 0.0, 0.0],
        max_token_kl=0.05,
    ) == 1
    assert valid_prefix_length(
        target_top_tokens=[1, 2, 3],
        candidate_top_tokens=[1, 2, 3],
        token_kls=[0.0, 0.06, 0.0],
        max_token_kl=0.05,
    ) == 1


def quality_row(*, matches: int, kls: list[float], valid: int) -> dict:
    return {
        "token_count": len(kls),
        "top1_matches": matches,
        "token_kls": kls,
        "valid_prefix_length": valid,
        "mlp_relative_l2": 0.1,
        "family": "control",
    }


def test_right_censored_and_rejected_decisions_are_distinct():
    perfect_rows = [quality_row(matches=7, kls=[0.0] * 7, valid=7) for _ in range(4)]
    perfect = aggregate_quality_rows(perfect_rows)
    gates, decision = gate_decision(
        baseline_mismatches=0,
        population=perfect,
        families={"control": perfect},
        observed_horizon=7,
        required_p50_hot_tokens=13821,
        required_p05_hot_tokens=6249,
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
    )
    assert gates["all_observations_right_censored"]
    assert decision == "INCONCLUSIVE_TANGENT_LIFETIME_EXTEND_TRACE"

    failed_rows = [quality_row(matches=6, kls=[0.0] * 7, valid=1) for _ in range(4)]
    failed = aggregate_quality_rows(failed_rows)
    gates, decision = gate_decision(
        baseline_mismatches=0,
        population=failed,
        families={"control": failed},
        observed_horizon=7,
        required_p50_hot_tokens=13821,
        required_p05_hot_tokens=6249,
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
    )
    assert not gates["population_quality_passed"]
    assert decision == "REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH"


def test_nonfinite_inputs_fail_closed():
    with pytest.raises(ValueError):
        charged_cycle_fraction(
            hot_tokens=1,
            hot_fraction=math.nan,
            construction_token_equivalents=1.0,
        )
