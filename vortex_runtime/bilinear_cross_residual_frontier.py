"""Finite E0 bound for matrix-local separable bilinear residual codes.

The model in this module is deliberately scoped.  For every binary matrix
``W`` it chooses a row subspace ``A`` and a column subspace ``B``, stores the
lossless matrix-local images needed for the three span terms, and probes raw
coordinates for the remaining ``e.T @ W @ f`` cross residual.  Access to all
stored images, nearest-codeword selection, decoding, and arithmetic outside
the raw cross residual are granted for free.

This is not a lower bound for arbitrary nonlinear, cross-matrix, or word-RAM
data structures.  It is a favorable coefficient-probe lower bound for the
separable extension of the Causal Residual Atlas family.
"""

from __future__ import annotations

from fractions import Fraction
import math

from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)


HOT_SIDE_BITS = 8 * GIB * 8
P50_TARGET_FRACTION = Fraction(8, 675)

# A rational witness chosen well inside the entropy inequality.  It avoids an
# optimized floating threshold and leaves a wide numerical margin:
# H_2(3/16) = 0.6962122601... < 0.7 = 1 - 0.3.
SIDE_RATE_THRESHOLD = Fraction(3, 10)
RADIUS_FRACTION_WITNESS = Fraction(3, 16)


def binary_entropy(probability: Fraction | float) -> float:
    """Return the binary entropy H_2(p), with the endpoint convention."""

    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hamming_ball_volume(length: int, radius: int) -> int:
    """Exact number of binary words within Hamming radius ``radius``."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0 <= radius <= length:
        raise ValueError("radius must be in [0, length]")
    return sum(math.comb(length, index) for index in range(radius + 1))


def separable_image_rank(
    rows: int,
    columns: int,
    row_code_dimension: int,
    column_code_dimension: int,
) -> int:
    """Independent binary image bits for ``A.T W`` and ``W B``.

    For full-column-rank row/column bases, the two image maps have ranks
    ``a*n`` and ``m*b`` and share an ``a*b`` overlap.  The result is therefore
    ``a*n + m*b - a*b``.  It is also the minimum lossless bit count for the
    pair of images over arbitrary binary ``W``.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    if not 0 <= row_code_dimension <= rows:
        raise ValueError("row code dimension is out of range")
    if not 0 <= column_code_dimension <= columns:
        raise ValueError("column code dimension is out of range")
    return (
        row_code_dimension * columns
        + rows * column_code_dimension
        - row_code_dimension * column_code_dimension
    )


def hot_side_fraction(
    hot_side_bits: int = HOT_SIDE_BITS,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
) -> Fraction:
    """Hot bits per binary coefficient under the most favorable reduction."""

    if hot_side_bits < 0 or coefficient_count <= 0:
        raise ValueError("invalid state or coefficient count")
    return Fraction(hot_side_bits, coefficient_count)


def low_side_population_fraction(
    side_fraction: Fraction,
    threshold: Fraction = SIDE_RATE_THRESHOLD,
) -> Fraction:
    """Weighted mass whose matrix-local side rate is at most ``threshold``.

    This is Markov's inequality applied with coefficient counts as weights.
    """

    if side_fraction < 0:
        raise ValueError("side fraction must be non-negative")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    return max(Fraction(0), Fraction(1) - side_fraction / threshold)


def scoped_cross_probe_fraction(
    side_fraction: Fraction | None = None,
    threshold: Fraction = SIDE_RATE_THRESHOLD,
    radius_fraction: Fraction = RADIUS_FRACTION_WITNESS,
) -> Fraction:
    """Conservative whole-population raw cross-coordinate lower bound.

    If a matrix-local separable image rate is at most 0.3, each row and column
    code rate is at most 0.3.  The binary sphere-covering inequality then gives
    a worst-case residual radius strictly above 3/16 in both directions.  The
    cross residual therefore probes at least (3/16)^2 of that matrix.  Markov's
    inequality supplies the coefficient-weighted population mass on which the
    per-matrix premise must hold.
    """

    if side_fraction is None:
        side_fraction = hot_side_fraction()
    if not 0 <= radius_fraction <= Fraction(1, 2):
        raise ValueError("radius fraction must be in [0, 1/2]")
    return (
        low_side_population_fraction(side_fraction, threshold)
        * radius_fraction
        * radius_fraction
    )


def minimum_hot_bits_to_escape_witness(
    target_fraction: Fraction = P50_TARGET_FRACTION,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    threshold: Fraction = SIDE_RATE_THRESHOLD,
    radius_fraction: Fraction = RADIUS_FRACTION_WITNESS,
) -> Fraction:
    """Hot bits at which this particular Markov witness stops rejecting.

    Reaching this value is necessary only to evade this proof witness.  It is
    not sufficient to construct a code or to satisfy any runtime resource.
    """

    residual_square = radius_fraction * radius_fraction
    if target_fraction >= residual_square:
        return Fraction(0)
    required_side_fraction = threshold * (
        Fraction(1) - target_fraction / residual_square
    )
    return required_side_fraction * coefficient_count


def entropy_witness_holds() -> bool:
    """Numerically check the conservative finite sphere-cover witness."""

    return (
        float(SIDE_RATE_THRESHOLD)
        + binary_entropy(RADIUS_FRACTION_WITNESS)
        < 1.0
    )


def derive_frontier() -> dict[str, object]:
    """Return the frozen E0 quantities with exact rational provenance."""

    side_fraction = hot_side_fraction()
    low_mass = low_side_population_fraction(side_fraction)
    cross_fraction = scoped_cross_probe_fraction(side_fraction)
    cross_cells = cross_fraction * REGISTERED_NON_EMBEDDING_COEFFICIENTS
    target_cells = (
        P50_TARGET_FRACTION * REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    escape_bits = minimum_hot_bits_to_escape_witness()

    return {
        "classification": "E0_SCOPED_COEFFICIENT_PROBE_LOWER_BOUND",
        "model": "MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE",
        "normalization": {
            "non_embedding_coefficients": (
                REGISTERED_NON_EMBEDDING_COEFFICIENTS
            ),
            "hot_side_bits": HOT_SIDE_BITS,
            "hot_side_fraction_binary": str(side_fraction),
            "hot_side_fraction_binary_decimal": float(side_fraction),
            "p50_target_fraction": str(P50_TARGET_FRACTION),
            "p50_target_fraction_decimal": float(P50_TARGET_FRACTION),
        },
        "sphere_cover_witness": {
            "side_rate_threshold": str(SIDE_RATE_THRESHOLD),
            "radius_fraction": str(RADIUS_FRACTION_WITNESS),
            "binary_entropy_radius": binary_entropy(
                RADIUS_FRACTION_WITNESS
            ),
            "rate_plus_entropy": (
                float(SIDE_RATE_THRESHOLD)
                + binary_entropy(RADIUS_FRACTION_WITNESS)
            ),
            "holds": entropy_witness_holds(),
        },
        "weighted_bound": {
            "low_side_population_fraction": str(low_mass),
            "low_side_population_fraction_decimal": float(low_mass),
            "cross_probe_fraction": str(cross_fraction),
            "cross_probe_fraction_decimal": float(cross_fraction),
            "cross_probe_cells": int(cross_cells),
            "target_cells": str(target_cells),
            "target_cells_decimal": float(target_cells),
            "bound_over_target": float(
                cross_fraction / P50_TARGET_FRACTION
            ),
            "minimum_hot_bits_to_escape_this_witness": str(escape_bits),
            "minimum_hot_gib_to_escape_this_witness": float(
                escape_bits / (8 * GIB)
            ),
        },
        "decision": (
            "REJECT_MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE_AS_CORE"
        ),
        "claim_boundary": {
            "independent_cross_matrix_query_tuple": "MODEL ASSUMPTION",
            "general_nonlinear_data_structure": "NOT RULED OUT",
            "cross_matrix_shared_advice": "NOT RULED OUT",
            "word_packed_probe_mapping": "NOT RULED OUT",
            "real_transformer_reachability": "NOT TESTED",
            "model_or_hardware_execution": "NOT RUN",
        },
    }
