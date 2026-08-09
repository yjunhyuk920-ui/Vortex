from __future__ import annotations

from fractions import Fraction
import itertools

from vortex_runtime.bilinear_cross_residual_frontier import (
    HOT_SIDE_BITS,
    P50_TARGET_FRACTION,
    RADIUS_FRACTION_WITNESS,
    SIDE_RATE_THRESHOLD,
    binary_entropy,
    derive_frontier,
    entropy_witness_holds,
    hamming_ball_volume,
    hot_side_fraction,
    low_side_population_fraction,
    minimum_hot_bits_to_escape_witness,
    scoped_cross_probe_fraction,
    separable_image_rank,
)
from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)


def _gf2_rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for value in rows:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = current
                break
            current ^= basis[pivot]
    return len(basis)


def _image_map_rows(m: int, n: int, a: int, b: int) -> list[int]:
    """Independent coordinate-basis reference for A.T W and W B."""

    equations: list[int] = []
    for row in range(a):
        for column in range(n):
            equations.append(1 << (row * n + column))
    for row in range(m):
        for column in range(b):
            equations.append(1 << (row * n + column))
    return equations


def test_registered_binary_reduction_grants_all_eight_gib() -> None:
    assert HOT_SIDE_BITS == 8 * GIB * 8
    assert hot_side_fraction() == Fraction(16_384, 96_261)
    assert P50_TARGET_FRACTION == Fraction(8, 675)


def test_separable_image_rank_matches_independent_gf2_reference() -> None:
    for m, n, a, b in itertools.product(range(1, 6), repeat=4):
        if a > m or b > n:
            continue
        expected = separable_image_rank(m, n, a, b)
        assert _gf2_rank(_image_map_rows(m, n, a, b)) == expected


def test_bilinear_span_plus_cross_identity_is_exact_over_f2() -> None:
    # Coordinate row/column subspaces are sufficient for an exhaustive,
    # implementation-independent identity control.
    m, n, a, b = 3, 4, 1, 2
    for matrix_bits in range(1 << (m * n)):
        matrix = [
            [(matrix_bits >> (i * n + j)) & 1 for j in range(n)]
            for i in range(m)
        ]
        for u_bits in range(1 << m):
            u = [(u_bits >> i) & 1 for i in range(m)]
            u_center = u[:a] + [0] * (m - a)
            e = [left ^ right for left, right in zip(u, u_center)]
            for v_bits in range(1 << n):
                v = [(v_bits >> j) & 1 for j in range(n)]
                v_center = v[:b] + [0] * (n - b)
                f = [left ^ right for left, right in zip(v, v_center)]

                def bilinear(left: list[int], right: list[int]) -> int:
                    return sum(
                        left[i] * matrix[i][j] * right[j]
                        for i in range(m)
                        for j in range(n)
                    ) % 2

                native = bilinear(u, v)
                split = (
                    bilinear(u_center, v_center)
                    ^ bilinear(e, v_center)
                    ^ bilinear(u_center, f)
                    ^ bilinear(e, f)
                )
                assert split == native


def test_entropy_witness_is_strictly_inside_sphere_cover_bound() -> None:
    assert SIDE_RATE_THRESHOLD == Fraction(3, 10)
    assert RADIUS_FRACTION_WITNESS == Fraction(3, 16)
    assert binary_entropy(RADIUS_FRACTION_WITNESS) < 0.7
    assert entropy_witness_holds()
    # A finite exact-volume control at d=16: 2^floor(.3d) centers and
    # radius floor(3d/16) cannot cover the cube.
    centers = 1 << 4
    assert centers * hamming_ball_volume(16, 3) < (1 << 16)


def test_markov_population_bound_is_exact_and_favorable() -> None:
    low_mass = low_side_population_fraction(hot_side_fraction())
    assert low_mass == Fraction(124_943, 288_783)
    assert float(low_mass) > 0.4326


def test_scoped_cross_probe_bound_rejects_p50_before_other_costs() -> None:
    bound = scoped_cross_probe_fraction()
    assert bound == Fraction(124_943, 8_214_272)
    assert float(bound) == 0.015210477568797332
    assert bound > P50_TARGET_FRACTION
    assert float(bound / P50_TARGET_FRACTION) > 1.283
    cells = bound * REGISTERED_NON_EMBEDDING_COEFFICIENTS
    assert cells == 6_141_198_336


def test_escape_value_exceeds_complete_fixed_hot_budget() -> None:
    required_bits = minimum_hot_bits_to_escape_witness()
    assert required_bits > HOT_SIDE_BITS
    assert float(required_bits / (8 * GIB)) > 9.347
    # This is only where the witness stops rejecting, never a sufficiency
    # or architecture claim.
    assert float(required_bits / (8 * GIB)) < 9.348


def test_result_keeps_general_data_structures_and_execution_unverified() -> None:
    result = derive_frontier()
    assert result["decision"] == (
        "REJECT_MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE_AS_CORE"
    )
    boundary = result["claim_boundary"]
    assert boundary["independent_cross_matrix_query_tuple"] == (
        "MODEL ASSUMPTION"
    )
    assert boundary["general_nonlinear_data_structure"] == "NOT RULED OUT"
    assert boundary["cross_matrix_shared_advice"] == "NOT RULED OUT"
    assert boundary["model_or_hardware_execution"] == "NOT RUN"
