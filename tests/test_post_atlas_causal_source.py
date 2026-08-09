from decimal import Decimal

import numpy as np
import pytest

from vortex_runtime.post_atlas_causal_source import (
    ATLAS_REGISTERED_TRAFFIC_FRACTION,
    KNOWN_VERIFIER_TRAFFIC_FRACTION,
    TARGET_FRACTION,
    cross_residual_perturbation,
    decompose_decision_bilinear,
    derive_post_atlas_resource_audit,
    minimum_service_tokens_for_dense_builds,
)


def _basis(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, columns)))[0]


def test_bilinear_decomposition_is_exact_and_keeps_cross_residual() -> None:
    rng = np.random.default_rng(84001)
    weight = rng.normal(size=(7, 9))
    activation = rng.normal(size=9)
    direction = rng.normal(size=7)
    primal = _basis(rng, 9, 3)
    dual = _basis(rng, 7, 2)

    result = decompose_decision_bilinear(
        weight, activation, direction, primal, dual
    )
    assert result.reconstructed_value == pytest.approx(result.exact_value, abs=1e-10)
    assert result.input_residual_l2 > 0
    assert result.dual_residual_l2 > 0


def test_cross_residual_is_indistinguishable_to_both_cached_images() -> None:
    rng = np.random.default_rng(84002)
    weight = rng.normal(size=(6, 8))
    primal = _basis(rng, 8, 3)
    dual = _basis(rng, 6, 2)
    activation = rng.normal(size=8)
    direction = rng.normal(size=6)
    u = activation - primal @ (primal.T @ activation)
    r = direction - dual @ (dual.T @ direction)
    delta = cross_residual_perturbation(u, r, scale=0.25)

    assert np.allclose(delta @ primal, 0.0, atol=1e-10)
    assert np.allclose(delta.T @ dual, 0.0, atol=1e-10)
    assert float(r @ delta @ u) > 0

    before = decompose_decision_bilinear(
        weight, activation, direction, primal, dual
    )
    after = decompose_decision_bilinear(
        weight + delta, activation, direction, primal, dual
    )
    assert after.exact_value != pytest.approx(before.exact_value)
    assert np.allclose(weight @ primal, (weight + delta) @ primal, atol=1e-10)
    assert np.allclose(weight.T @ dual, (weight + delta).T @ dual, atol=1e-10)


def test_cross_residual_vanishes_when_either_side_is_in_span() -> None:
    rng = np.random.default_rng(84003)
    weight = rng.normal(size=(5, 7))
    primal = _basis(rng, 7, 2)
    dual = _basis(rng, 5, 2)
    activation = primal @ np.asarray([1.0, -2.0])
    direction = rng.normal(size=5)
    result = decompose_decision_bilinear(
        weight, activation, direction, primal, dual
    )
    assert result.cross_residual_term == pytest.approx(0.0, abs=1e-10)

    activation = rng.normal(size=7)
    direction = dual @ np.asarray([0.5, 3.0])
    result = decompose_decision_bilinear(
        weight, activation, direction, primal, dual
    )
    assert result.cross_residual_term == pytest.approx(0.0, abs=1e-10)


def test_dynamic_dual_build_conservation_thresholds() -> None:
    assert TARGET_FRACTION == Decimal(8) / Decimal(675)
    assert minimum_service_tokens_for_dense_builds(1) == 85
    assert minimum_service_tokens_for_dense_builds(16) == 1350
    assert (
        minimum_service_tokens_for_dense_builds(
            1, common_fraction=KNOWN_VERIFIER_TRAFFIC_FRACTION
        )
        == 109
    )
    assert (
        minimum_service_tokens_for_dense_builds(
            1, common_fraction=ATLAS_REGISTERED_TRAFFIC_FRACTION
        )
        == 999
    )


def test_static_vocabulary_dual_code_misses_before_exactness_overhead() -> None:
    audit = derive_post_atlas_resource_audit()
    static = audit["static_vocabulary_dual_code"]
    assert static["one_last_down_coefficients"] == 6_829_375_488
    assert static["one_last_down_bytes"] == 13_658_750_976
    assert Decimal(static["one_last_down_gib"]) == Decimal("12.720703125")
    assert Decimal(static["one_last_down_full_scan_fraction"]) > TARGET_FRACTION
    assert Decimal(static["one_last_down_target_miss_factor"]) > Decimal(5)
    assert static["all_down_bytes"] == 1_721_002_622_976


def test_invalid_bases_and_resource_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        decompose_decision_bilinear(
            np.eye(2), np.ones(2), np.ones(2), np.ones((2, 1)), np.eye(2)
        )
    with pytest.raises(ValueError):
        minimum_service_tokens_for_dense_builds(-1)
