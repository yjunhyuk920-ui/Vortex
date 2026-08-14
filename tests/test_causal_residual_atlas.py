from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from vortex_runtime.causal_residual_atlas import (
    REGISTERED_OUTPUT_ROWS,
    REGISTERED_SHARED_SITE_INPUT_DIMENSIONS,
    certify_top1,
    compile_prefix_atlas,
    compile_block_frobenius_bounds,
    derive_atlas_budget,
    evaluate_residual_atlas,
    prefix_basis,
)


def test_committed_prefix_span_reconstructs_exact_linear_result() -> None:
    rng = np.random.default_rng(17)
    weight = rng.normal(size=(9, 7))
    prefix = rng.normal(size=(3, 7))
    basis, image = compile_prefix_atlas(prefix, prefix @ weight.T)
    x = prefix.T @ np.asarray([0.5, -1.25, 2.0])

    result = evaluate_residual_atlas(
        weight,
        basis,
        image,
        x,
        block_columns=2,
    )

    assert np.linalg.norm(result.input_residual) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(image, weight @ basis, rtol=0.0, atol=1e-11)
    assert result.certified_unread_radius == pytest.approx(0.0, abs=1e-11)
    assert np.allclose(result.estimate, result.exact, rtol=0.0, atol=1e-11)


def test_prefix_basis_keeps_late_direction_after_dependent_arrival() -> None:
    prefix = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    basis = prefix_basis(prefix)
    assert basis.shape == (3, 1)
    assert np.allclose(
        basis @ (basis.T @ np.asarray([3.0, 0.0, 0.0])),
        np.asarray([3.0, 0.0, 0.0]),
    )


def test_prefix_atlas_image_uses_pairs_without_weight_input() -> None:
    rng = np.random.default_rng(19)
    weight = rng.normal(size=(5, 6))
    prefix = rng.normal(size=(4, 6))
    basis, image = compile_prefix_atlas(prefix, prefix @ weight.T)
    assert np.allclose(image, weight @ basis, rtol=0.0, atol=1e-11)


def test_unread_residual_is_never_larger_than_frobenius_certificate() -> None:
    rng = np.random.default_rng(23)
    weight = rng.normal(size=(11, 13))
    prefix = rng.normal(size=(4, 13))
    basis = prefix_basis(prefix)
    image = weight @ basis
    x = rng.normal(size=13)
    bounds = compile_block_frobenius_bounds(weight, block_columns=3)

    result = evaluate_residual_atlas(
        weight,
        basis,
        image,
        x,
        block_columns=3,
        revealed_blocks=(0, 3),
        block_bounds=bounds,
    )

    assert result.exact_unread_error_norm <= (
        result.certified_unread_radius + 1e-12
    )
    assert result.certified_unread_radius > 0


def test_revealing_every_page_produces_exact_fallback_boundary() -> None:
    rng = np.random.default_rng(29)
    weight = rng.normal(size=(8, 10))
    prefix = rng.normal(size=(2, 10))
    basis = prefix_basis(prefix)
    result = evaluate_residual_atlas(
        weight,
        basis,
        weight @ basis,
        rng.normal(size=10),
        block_columns=4,
        revealed_blocks=(0, 1, 2),
    )

    assert result.certified_unread_radius == 0
    assert np.allclose(result.estimate, result.exact, rtol=0.0, atol=1e-11)


def test_top1_certificate_is_strict_and_fail_closed() -> None:
    assert certify_top1(np.asarray([1.0, 4.0, 2.0]), 0.9) == 1
    assert certify_top1(np.asarray([1.0, 4.0, 2.0]), 1.0) is None
    with pytest.raises(ValueError):
        certify_top1(np.asarray([1.0, np.nan]), 0.0)


def test_registered_shape_population_and_rank16_page_screen() -> None:
    assert REGISTERED_OUTPUT_ROWS == 19_997_952
    assert REGISTERED_SHARED_SITE_INPUT_DIMENSIONS == 12_918_784

    budget = derive_atlas_budget(
        rank=16,
        requested_cold_fraction="0.002",
        page_columns=64,
        capsule_scalar_bytes=2,
        metadata_bytes_per_block=4,
        service_tokens=64,
    )
    assert budget.capsule_bytes == 1_053_335_552
    assert budget.capsule_gib == Decimal("0.98099517822265625")
    assert budget.metadata_blocks == 298_624
    assert budget.metadata_bytes == 1_194_496
    assert budget.selected_pages_per_token == 1_009
    assert budget.selected_coefficients_per_token == 1_411_989_504
    assert budget.capsule_traffic_bytes == 1_466_736_640
    assert budget.actual_cold_fraction == pytest.approx(
        Decimal("0.00349720583881322654")
    )
    assert budget.fast_traffic_fraction < Decimal(8) / Decimal(675)
    assert budget.fast_operation_fraction < Decimal(8) / Decimal(675)
    assert budget.minimum_build_traffic_bytes == budget.capsule_bytes
    assert budget.minimum_build_operations == 6_026_930_176
    assert budget.amortized_traffic_fraction > budget.fast_traffic_fraction
    assert budget.amortized_operation_fraction > budget.fast_operation_fraction
    assert budget.amortized_traffic_fraction == pytest.approx(
        Decimal("0.010850257155067018315")
    )
    assert budget.amortized_operation_fraction == pytest.approx(
        Decimal("0.005579585753435199938")
    )
    assert budget.minimum_traffic_coverage == pytest.approx(
        Decimal("0.998998405303215166463")
    )


def test_page_rounding_and_fallback_coverage_are_not_hidden() -> None:
    budget = derive_atlas_budget(
        rank=16,
        requested_cold_fraction="0.002",
        page_columns=64,
    )
    assert budget.actual_cold_fraction > Decimal("0.002")
    assert budget.minimum_traffic_coverage is not None
    assert budget.minimum_traffic_coverage > Decimal("0.998")

    impossible = derive_atlas_budget(
        rank=32,
        requested_cold_fraction="0.002",
        page_columns=64,
    )
    assert impossible.minimum_traffic_coverage is None


def test_invalid_budget_and_reference_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        derive_atlas_budget(rank=0, requested_cold_fraction="0.1")
    with pytest.raises(ValueError):
        derive_atlas_budget(rank=1, requested_cold_fraction="1.1")
    with pytest.raises(ValueError):
        prefix_basis(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        prefix_basis(np.asarray([[0.0, np.nan, 1.0]]))
    with pytest.raises(ValueError):
        compile_prefix_atlas(np.ones((2, 3)), np.ones((3, 2)))

    weight = np.eye(3)
    with pytest.raises(ValueError):
        evaluate_residual_atlas(
            weight,
            np.ones((3, 1)),
            np.ones((3, 1)),
            np.ones(3),
            block_columns=1,
        )
    with pytest.raises(ValueError):
        evaluate_residual_atlas(
            weight,
            np.eye(3)[:, :1],
            np.zeros((3, 1)),
            np.ones(3),
            block_columns=1,
        )


def test_randomized_certificate_bound_has_no_false_accept() -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        weight = rng.normal(size=(7, 9))
        prefix = rng.normal(size=(3, 9))
        basis = prefix_basis(prefix)
        image = weight @ basis
        x = rng.normal(size=9)
        for revealed in ((), (0,), (1, 2)):
            result = evaluate_residual_atlas(
                weight,
                basis,
                image,
                x,
                block_columns=3,
                revealed_blocks=revealed,
            )
            assert result.exact_unread_error_norm <= (
                result.certified_unread_radius + 1e-11
            )
