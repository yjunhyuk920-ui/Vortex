from __future__ import annotations

from decimal import Decimal

import pytest

from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION,
    P50_TARGET_FRACTION,
    REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
    REGISTERED_NON_EMBEDDING_Q4_BYTES,
    REGISTERED_TOTAL_Q4_BYTES,
    ResourceTerms,
    charged_fraction,
    maximum_pages,
    maximum_payload_bytes_per_call,
    minimum_coverage,
    minimum_service_tokens,
    pcie_transfer_floor_seconds,
    resident_metadata_bytes,
)


def test_registered_shape_and_exact_target_fraction() -> None:
    assert REGISTERED_NON_EMBEDDING_Q4_BYTES == 201_873_948_672
    assert REGISTERED_TOTAL_Q4_BYTES == 202_924_621_824
    assert P50_TARGET_FRACTION == Decimal(8) / Decimal(675)


def test_zero_cost_path_still_requires_98_8148_percent_coverage() -> None:
    required = minimum_coverage(P50_TARGET_FRACTION, ResourceTerms())
    assert required is not None
    assert required == Decimal(1) - P50_TARGET_FRACTION
    assert required * 100 == pytest.approx(Decimal("98.814814814815"))


def test_known_verifier_raises_coverage_and_reduces_page_budget() -> None:
    terms = ResourceTerms(common=KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION)
    required = minimum_coverage(P50_TARGET_FRACTION, terms)
    assert required is not None
    assert required * 100 == pytest.approx(Decimal("99.081653738815"))
    assert maximum_pages(P50_TARGET_FRACTION, 4096) == 584_126
    assert (
        maximum_pages(
            P50_TARGET_FRACTION,
            4096,
            other_traffic_fraction=(
                KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION
            ),
        )
        == 452_612
    )


def test_general_equation_inverts_and_charges_wasted_miss_work() -> None:
    terms = ResourceTerms(
        common=Decimal("0.001"),
        hit=Decimal("0.002"),
        miss=Decimal("0.003"),
        compile_dense_equivalents=10,
        service_tokens=100_000,
    )
    required = minimum_coverage(P50_TARGET_FRACTION, terms)
    assert required is not None
    assert charged_fraction(required, terms) == pytest.approx(
        P50_TARGET_FRACTION, abs=Decimal("1e-49")
    )
    assert charged_fraction(Decimal("0.99"), terms) > charged_fraction(
        Decimal("0.999"), terms
    )


def test_exp081_fallback_frontier_is_reproduced() -> None:
    terms = ResourceTerms(common=Decimal("0.009266579409111565"))
    required = minimum_coverage(P50_TARGET_FRACTION, terms)
    assert required is not None
    assert required == pytest.approx(Decimal("0.9974147275572597"))


def test_conditional_one_page_per_matrix_payload_ceiling() -> None:
    assert REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES == 883
    assert (
        maximum_payload_bytes_per_call(
            P50_TARGET_FRACTION,
            REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
        )
        == 2_709_603
    )
    assert (
        maximum_payload_bytes_per_call(
            P50_TARGET_FRACTION,
            REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
            other_traffic_fraction=(
                KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION
            ),
        )
        == 2_099_549
    )


def test_page_metadata_and_compile_amortization_are_not_free() -> None:
    assert resident_metadata_bytes(
        REGISTERED_TOTAL_Q4_BYTES, 4096, 32
    ) == 1_585_348_608
    assert Decimal(1_585_348_608) / Decimal(GIB) == Decimal(
        "1.476470947265625"
    )
    assert minimum_service_tokens(1, Decimal("0.001")) == 1_000
    assert minimum_service_tokens(10, Decimal("0.001")) == 10_000


def test_favorable_gen2_transfer_floor_is_about_299_ms() -> None:
    floor = pcie_transfer_floor_seconds(
        P50_TARGET_FRACTION, Decimal("7.450580596923828")
    )
    assert floor == pytest.approx(Decimal("0.299072517"), abs=Decimal("1e-9"))


def test_perfect_coverage_can_still_be_infeasible() -> None:
    assert (
        minimum_coverage(
            P50_TARGET_FRACTION,
            ResourceTerms(common=P50_TARGET_FRACTION + Decimal("0.0001")),
        )
        is None
    )
