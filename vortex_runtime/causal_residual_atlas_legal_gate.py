"""Leakage-safe reference equations for the next Causal Residual Atlas Gate.

This module deliberately contains no checkpoint runner.  It fixes a selector
whose arguments contain no native current output or candidate logits, an
outward local residual radius, the final-RMSNorm/LM-head certificate, and the
fully charged favorable 405B accounting before the held-out model run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
import hashlib
import math
from typing import Sequence

import numpy as np

from vortex_runtime.causal_residual_atlas import (
    GIB,
    REGISTERED_ACTIVATION_SITES,
    REGISTERED_MATRIX_FAMILIES,
    REGISTERED_OUTPUT_ROWS,
    REGISTERED_SHARED_SITE_INPUT_DIMENSIONS,
    derive_atlas_budget,
)
from vortex_runtime.query_adaptive_cold_equation import (
    P50_TARGET_FRACTION,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    REGISTERED_NON_EMBEDDING_Q4_BYTES,
    ResourceTerms,
    minimum_coverage,
)


BF16_UNIT_ROUNDOFF = Decimal(1) / Decimal(256)
FP32_UNIT_ROUNDOFF = Decimal(1) / Decimal(2**24)
LM_HEAD_ROWS = 128_256
MATRIX_INSTANCES = sum(family.count for family in REGISTERED_MATRIX_FAMILIES)
ACTIVATION_SITE_INSTANCES = sum(site.count for site in REGISTERED_ACTIVATION_SITES)
LEGAL_PROMPT_PATH = (
    "docs/research/inputs/causal_residual_atlas_legal_gate_prompts.json"
)
LEGAL_PROMPT_SHA256 = (
    "67bd16d4f1e63a6f4e4a9131119335aff8d4d676549ce55287bf69b8fb879fb5"
)
PROMOTE_LEGAL_DECISION = (
    "PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_BACKWARD_LAYER_AND_POSITION_EXPANSION_GATE"
)
REJECT_LEGAL_DECISION = "REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH"
INVALID_LEGAL_DECISION = "INVALID_LEGAL_PAIR_OUTWARD_CONTROL_FAILURE"
INFRASTRUCTURE_LEGAL_DECISION = "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION"


def _up(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("outward quantity must be finite and non-negative")
    return math.nextafter(float(value), math.inf)


def _up_signed(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("outward quantity must be finite")
    return math.nextafter(float(value), math.inf)


def _down_signed(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("lower quantity must be finite")
    return math.nextafter(float(value), -math.inf)


def _down_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("lower quantity must be finite and non-negative")
    return max(0.0, math.nextafter(float(value), -math.inf))


def outward_sum(*values: float) -> float:
    total = 0.0
    for value in values:
        if not math.isfinite(value) or value < 0:
            raise ValueError("outward addends must be finite and non-negative")
        total = _up(total + float(value))
    return total


def outward_product(left: float, right: float) -> float:
    if not all(math.isfinite(value) and value >= 0 for value in (left, right)):
        raise ValueError("outward factors must be finite and non-negative")
    return _up(float(left) * float(right))


def outward_squared_norm(values: Sequence[float] | np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("norm input must be a finite vector")
    total = 0.0
    for value in vector:
        square = _up(abs(float(value)) * abs(float(value)))
        total = _up(total + square)
    return total


def _float64_gamma(operation_count: int) -> float:
    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    unit = math.ldexp(1.0, -53)
    product = operation_count * unit
    if product >= 1.0:
        raise ValueError("operation_count times float64 unit roundoff is too large")
    return math.nextafter(product / (1.0 - product), math.inf)


def array_frobenius_norm_upper(
    values: Sequence[float] | np.ndarray,
) -> float:
    """Fast directed upper bound for a finite float array's Frobenius norm."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("norm input must be a nonempty finite array")
    squares = array * array
    total = float(np.sum(squares, dtype=np.float64))
    gamma = _float64_gamma(2 * array.size)
    upper_square = math.nextafter(
        total / (1.0 - gamma)
        + array.size * float(np.nextafter(0.0, 1.0)),
        math.inf,
    )
    return math.nextafter(math.sqrt(upper_square), math.inf)


def matrix_row_norm_uppers(
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return directed L2 upper bounds for every row of a finite matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("row-norm input must be a nonempty finite matrix")
    width = matrix.shape[1]
    gamma = _float64_gamma(2 * width)
    sums = np.sum(matrix * matrix, axis=1, dtype=np.float64)
    upper_squares = np.nextafter(
        sums / (1.0 - gamma)
        + width * float(np.nextafter(0.0, 1.0)),
        np.inf,
    )
    return np.nextafter(np.sqrt(upper_squares), np.inf)


def bf16_vector_addition_l2_bound(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Bound one BF16 elementwise addition relative to exact-real addition."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.ndim != 1
        or right_array.shape != left_array.shape
        or left_array.size == 0
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        raise ValueError("addition inputs must be aligned finite vectors")
    magnitude = outward_sum(
        array_frobenius_norm_upper(left_array),
        array_frobenius_norm_upper(right_array),
    )
    relative = outward_product(float(BF16_UNIT_ROUNDOFF), magnitude)
    subnormal = _up(math.sqrt(left_array.size) * math.ldexp(1.0, -134))
    return outward_sum(relative, subnormal)


def bf16_addition_l2_bound_from_norms(
    *,
    left_norm_upper: float,
    right_norm_upper: float,
    dimension: int,
) -> float:
    """Bound one BF16 vector addition without observing either vector."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (left_norm_upper, right_norm_upper)
    ):
        raise ValueError("addition norm bounds must be finite and non-negative")
    magnitude = outward_sum(left_norm_upper, right_norm_upper)
    relative = outward_product(float(BF16_UNIT_ROUNDOFF), magnitude)
    subnormal = _up(math.sqrt(dimension) * math.ldexp(1.0, -134))
    return outward_sum(relative, subnormal)


def native_linear_per_row_rounding_bounds(
    row_norm_uppers: Sequence[float] | np.ndarray,
    *,
    input_norm_upper: float,
    columns: int,
) -> np.ndarray:
    """Vectorized FP32-accumulate/BF16-output bound for scalar rows."""

    norms = np.asarray(row_norm_uppers, dtype=np.float64)
    if (
        norms.ndim != 1
        or norms.size == 0
        or not np.all(np.isfinite(norms))
        or np.any(norms < 0)
        or not math.isfinite(input_norm_upper)
        or input_norm_upper < 0
        or columns <= 0
    ):
        raise ValueError("malformed per-row rounding inputs")
    gamma = float(fp_accumulation_gamma(columns))
    exact = np.nextafter(norms * input_norm_upper, np.inf)
    accumulation = np.nextafter(gamma * exact, np.inf)
    cast = np.nextafter(
        float(BF16_UNIT_ROUNDOFF)
        * np.nextafter(exact + accumulation, np.inf),
        np.inf,
    )
    return np.nextafter(
        np.nextafter(accumulation + cast, np.inf)
        + math.ldexp(1.0, -134),
        np.inf,
    )


def _inward_squared_norm(values: Sequence[float] | np.ndarray) -> float:
    """Return a directed lower bound for the squared norm of float inputs."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("norm input must be a finite vector")
    total = 0.0
    for value in vector:
        magnitude = abs(float(value))
        square = max(0.0, _down_signed(magnitude * magnitude))
        total = max(0.0, _down_signed(total + square))
    return total


def select_max_residual_energy_page(
    residual: Sequence[float] | np.ndarray,
    *,
    page_columns: int,
) -> int:
    """Choose a page using the current input residual and nothing else.

    The common certified operator bound makes the page with maximum residual
    energy minimize the registered unread-radius upper bound.  Exact ties use
    the lowest page index.  No weight, native output, or logit is an argument.
    """

    vector = np.asarray(residual, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a nonempty finite vector")
    if page_columns <= 0:
        raise ValueError("page_columns must be positive")
    energies = [
        math.fsum(
            float(value) * float(value)
            for value in vector[start : start + page_columns]
        )
        for start in range(0, vector.size, page_columns)
    ]
    return min(range(len(energies)), key=lambda index: (-energies[index], index))


@dataclass(frozen=True)
class PairOnlyMGS:
    basis: np.ndarray
    image: np.ndarray
    coefficient_map: np.ndarray
    accepted_prefix_indices: tuple[int, ...]
    accepted_residual_norm_lowers: tuple[float, ...]
    accepted_threshold_uppers: tuple[float, ...]


def compile_pair_only_mgs(
    prefix_inputs: Sequence[Sequence[float]] | np.ndarray,
    prefix_images: Sequence[Sequence[float]] | np.ndarray,
    *,
    maximum_rank: int,
) -> PairOnlyMGS:
    """Two-pass causal MGS using input/image pairs and no weight argument."""

    inputs = np.asarray(prefix_inputs, dtype=np.float64)
    images = np.asarray(prefix_images, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("prefix_inputs must be a nonempty matrix")
    if images.ndim != 2 or images.shape[0] != inputs.shape[0] or images.shape[1] == 0:
        raise ValueError("prefix_images must align and have nonempty width")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(images)):
        raise ValueError("prefix pairs must be finite")
    if maximum_rank <= 0:
        raise ValueError("maximum_rank must be positive")

    basis_columns: list[np.ndarray] = []
    image_columns: list[np.ndarray] = []
    coefficient_columns: list[np.ndarray] = []
    accepted: list[int] = []
    accepted_norm_lowers: list[float] = []
    accepted_threshold_uppers: list[float] = []
    epsilon = float(np.finfo(np.float64).eps)
    threshold_scale = max(inputs.shape)
    for prefix_index, (input_row, image_row) in enumerate(zip(inputs, images)):
        residual = input_row.copy()
        image_residual = image_row.copy()
        coefficients = np.zeros(inputs.shape[0], dtype=np.float64)
        coefficients[prefix_index] = 1.0
        for _ in range(2):
            if not basis_columns:
                continue
            basis = np.column_stack(basis_columns)
            image = np.column_stack(image_columns)
            coefficient_map = np.column_stack(coefficient_columns)
            projection = basis.T @ residual
            residual = residual - basis @ projection
            image_residual = image_residual - image @ projection
            coefficients = coefficients - coefficient_map @ projection
        norm_lower = _down_nonnegative(
            math.sqrt(_inward_squared_norm(residual))
        )
        input_norm_upper = _up(math.sqrt(outward_squared_norm(input_row)))
        threshold = _up(
            threshold_scale * epsilon * max(1.0, input_norm_upper)
        )
        if norm_lower <= threshold:
            continue
        norm = float(np.linalg.norm(residual))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("accepted residual norm must be finite and positive")
        basis_columns.append(np.ascontiguousarray(residual / norm))
        image_columns.append(np.ascontiguousarray(image_residual / norm))
        coefficient_columns.append(np.ascontiguousarray(coefficients / norm))
        accepted.append(prefix_index)
        accepted_norm_lowers.append(norm_lower)
        accepted_threshold_uppers.append(threshold)
        if len(basis_columns) == maximum_rank:
            break

    basis = (
        np.column_stack(basis_columns)
        if basis_columns
        else np.empty((inputs.shape[1], 0), dtype=np.float64)
    )
    image = (
        np.column_stack(image_columns)
        if image_columns
        else np.empty((images.shape[1], 0), dtype=np.float64)
    )
    coefficient_map = (
        np.column_stack(coefficient_columns)
        if coefficient_columns
        else np.empty((inputs.shape[0], 0), dtype=np.float64)
    )
    return PairOnlyMGS(
        basis=np.ascontiguousarray(basis),
        image=np.ascontiguousarray(image),
        coefficient_map=np.ascontiguousarray(coefficient_map),
        accepted_prefix_indices=tuple(accepted),
        accepted_residual_norm_lowers=tuple(accepted_norm_lowers),
        accepted_threshold_uppers=tuple(accepted_threshold_uppers),
    )


def certified_unread_radius(
    operator_norm_upper: float,
    residual: Sequence[float] | np.ndarray,
    *,
    page_columns: int,
    selected_page: int,
) -> float:
    """Bound ``||W[:, unread] u[unread]||_2`` with directed rounding."""

    if not math.isfinite(operator_norm_upper) or operator_norm_upper < 0:
        raise ValueError("operator_norm_upper must be finite and non-negative")
    vector = np.asarray(residual, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a nonempty finite vector")
    if page_columns <= 0:
        raise ValueError("page_columns must be positive")
    page_count = math.ceil(vector.size / page_columns)
    if not 0 <= selected_page < page_count:
        raise ValueError("selected_page is out of range")
    unread = vector.copy()
    start = selected_page * page_columns
    unread[start : start + page_columns] = 0.0
    norm_upper = _up(math.sqrt(outward_squared_norm(unread)))
    return outward_product(operator_norm_upper, norm_upper)


def pair_image_defect_bound(
    *,
    operator_norm_upper: float,
    stored_basis_error_operator: float,
    prefix_image_error_frobenius: float,
    pair_coefficient_operator: float,
    stored_image_error_operator: float,
) -> float:
    """Bound ``||W Q_stored - Z_stored||_2`` from pair-only compilation."""

    return outward_sum(
        outward_product(operator_norm_upper, stored_basis_error_operator),
        outward_product(prefix_image_error_frobenius, pair_coefficient_operator),
        stored_image_error_operator,
    )


def projection_output_radius(
    *,
    pair_image_defect: float,
    coordinate_norm_upper: float,
    unread_radius: float,
    selected_page_arithmetic_radius: float = 0.0,
    candidate_cast_radius: float = 0.0,
    native_dense_arithmetic_radius: float = 0.0,
) -> float:
    """Return a native-output comparison radius for one legal page result."""

    return outward_sum(
        outward_product(pair_image_defect, coordinate_norm_upper),
        unread_radius,
        selected_page_arithmetic_radius,
        candidate_cast_radius,
        native_dense_arithmetic_radius,
    )


def fp_accumulation_gamma(width: int, *, unit_roundoff: Decimal = FP32_UNIT_ROUNDOFF) -> Decimal:
    if width <= 0:
        raise ValueError("width must be positive")
    product = Decimal(width) * unit_roundoff
    if product >= 1:
        raise ValueError("width times unit roundoff must be below one")
    return product / (Decimal(1) - product)


def native_linear_l2_rounding_bound(
    *,
    input_norm_upper: float,
    frobenius_norm_upper: float,
    operator_norm_upper: float,
    rows: int,
    columns: int,
) -> float:
    """Conservative FP32-accumulate/BF16-output linear-map error bound.

    The future runner must first prove that the pinned native control satisfies
    this registered arithmetic model.  A mismatch invalidates the Gate rather
    than widening the bound after observation.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (input_norm_upper, frobenius_norm_upper, operator_norm_upper)
    ):
        raise ValueError("norm bounds must be finite and non-negative")
    gamma = float(fp_accumulation_gamma(columns))
    accumulation = outward_product(
        gamma,
        outward_product(frobenius_norm_upper, input_norm_upper),
    )
    exact_output = outward_product(operator_norm_upper, input_norm_upper)
    bf16_rounding = outward_product(
        float(BF16_UNIT_ROUNDOFF),
        outward_sum(exact_output, accumulation),
    )
    # Half a minimum BF16 subnormal per output is included explicitly.
    subnormal = _up(math.sqrt(rows) * math.ldexp(1.0, -134))
    return outward_sum(accumulation, bf16_rounding, subnormal)


def rmsnorm_output_radius(
    center: Sequence[float] | np.ndarray,
    *,
    input_radius: float,
    gain_abs_max: float,
    epsilon: float,
    two_path_cast_radius: float = 0.0,
) -> float:
    """Propagate an L2 ball through RMSNorm with a local sound Lipschitz bound."""

    vector = np.asarray(center, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("center must be a nonempty finite vector")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (input_radius, gain_abs_max, epsilon, two_path_cast_radius)
    ) or epsilon == 0:
        raise ValueError("RMSNorm bounds must be finite and epsilon positive")
    center_norm_lower = _down_nonnegative(math.sqrt(_inward_squared_norm(vector)))
    ball_norm_lower = max(
        0.0,
        _down_signed(center_norm_lower - input_radius),
    )
    ball_square_lower = max(
        0.0,
        _down_signed(ball_norm_lower * ball_norm_lower),
    )
    mean_square_lower = max(
        0.0,
        _down_signed(ball_square_lower / vector.size),
    )
    denominator_square_lower = max(
        0.0,
        _down_signed(mean_square_lower + epsilon),
    )
    denominator_lower = max(
        0.0,
        _down_signed(math.sqrt(denominator_square_lower)),
    )
    if denominator_lower <= 0.0:
        raise ValueError("RMSNorm denominator lower bound vanished")
    lipschitz_upper = _up(gain_abs_max / denominator_lower)
    return outward_sum(
        outward_product(lipschitz_upper, input_radius),
        two_path_cast_radius,
    )


def rmsnorm_native_implementation_l2_bound(
    *,
    input_norm_lower: float,
    input_norm_upper: float,
    dimension: int,
    gain_abs_max: float,
    epsilon: float,
) -> float:
    """Enclose one pinned FP32-compute/BF16-output RMSNorm path.

    The reference is the exact-real RMSNorm evaluated with the exact float32
    gain values used by the checkpoint.  The bound charges square/reduction,
    epsilon addition, reciprocal square root, both FP32 multiplies, and the
    final BF16 cast.  It depends only on an input-norm interval, so the target
    path can be enclosed before its hidden state is evaluated.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (
            input_norm_lower,
            input_norm_upper,
            gain_abs_max,
            epsilon,
        )
    ) or epsilon == 0 or input_norm_lower > input_norm_upper:
        raise ValueError("malformed RMSNorm implementation bounds")

    u32 = float(FP32_UNIT_ROUNDOFF)
    gamma_reduction = float(fp_accumulation_gamma(2 * dimension + 2))
    norm_square_upper = outward_product(input_norm_upper, input_norm_upper)
    variance_upper = _up(norm_square_upper / dimension)
    variance_error = outward_sum(
        outward_product(gamma_reduction, variance_upper),
        _up(dimension * math.ldexp(1.0, -149)),
    )
    denominator_square_upper = outward_sum(
        variance_upper,
        epsilon,
        variance_error,
    )
    epsilon_add_error = outward_sum(
        outward_product(u32, denominator_square_upper),
        math.ldexp(1.0, -149),
    )
    scalar_error = outward_sum(variance_error, epsilon_add_error)

    norm_square_lower = _down_nonnegative(
        input_norm_lower * input_norm_lower
    )
    variance_lower = _down_nonnegative(norm_square_lower / dimension)
    exact_denominator_square_lower = max(
        epsilon,
        _down_nonnegative(variance_lower + epsilon),
    )
    implemented_denominator_square_lower = max(
        math.ldexp(1.0, -126),
        _down_nonnegative(
            max(0.0, exact_denominator_square_lower - scalar_error)
        ),
    )
    inverse_exact_upper = _up(1.0 / math.sqrt(exact_denominator_square_lower))
    inverse_implemented_upper = _up(
        1.0 / math.sqrt(implemented_denominator_square_lower)
    )
    inverse_input_error = outward_product(
        0.5 * implemented_denominator_square_lower ** -1.5,
        scalar_error,
    )
    inverse_rounding_error = outward_product(
        u32,
        inverse_implemented_upper,
    )
    inverse_error = outward_sum(
        inverse_input_error,
        inverse_rounding_error,
        math.ldexp(1.0, -149),
    )

    normalized_norm_upper = outward_product(
        input_norm_upper,
        inverse_implemented_upper,
    )
    inverse_effect = outward_product(
        input_norm_upper,
        inverse_error,
    )
    normalize_multiply = outward_sum(
        outward_product(u32, normalized_norm_upper),
        _up(math.sqrt(dimension) * math.ldexp(1.0, -149)),
    )
    gain_input_norm = outward_product(
        gain_abs_max,
        normalized_norm_upper,
    )
    gain_multiply = outward_sum(
        outward_product(u32, gain_input_norm),
        _up(math.sqrt(dimension) * math.ldexp(1.0, -149)),
    )
    fp32_error = outward_sum(
        outward_product(gain_abs_max, inverse_effect),
        outward_product(gain_abs_max, normalize_multiply),
        gain_multiply,
    )
    exact_output_norm_upper = outward_product(
        gain_abs_max,
        outward_product(input_norm_upper, inverse_exact_upper),
    )
    cast_input_upper = outward_sum(exact_output_norm_upper, fp32_error)
    bf16_cast = outward_sum(
        outward_product(float(BF16_UNIT_ROUNDOFF), cast_input_upper),
        _up(math.sqrt(dimension) * math.ldexp(1.0, -134)),
    )
    return outward_sum(fp32_error, bf16_cast)


@dataclass(frozen=True)
class Top1OutwardCertificate:
    winner: int
    certified: bool
    minimum_margin_lower: float


def certify_top1_from_row_norms(
    candidate_logits: Sequence[float] | np.ndarray,
    *,
    hidden_radius: float,
    row_norm_uppers: Sequence[float] | np.ndarray,
    per_logit_rounding_uppers: Sequence[float] | np.ndarray | None = None,
) -> Top1OutwardCertificate:
    """Certify native greedy top-1 without target logits or a candidate sweep."""

    logits = np.asarray(candidate_logits, dtype=np.float64)
    norms = np.asarray(row_norm_uppers, dtype=np.float64)
    rounding = (
        np.zeros_like(logits)
        if per_logit_rounding_uppers is None
        else np.asarray(per_logit_rounding_uppers, dtype=np.float64)
    )
    if logits.ndim != 1 or logits.size < 2 or norms.shape != logits.shape or rounding.shape != logits.shape:
        raise ValueError("logits, row norms, and rounding bounds must align")
    if not np.all(np.isfinite(logits)):
        raise ValueError("candidate_logits must be finite")
    if not np.all(np.isfinite(norms)) or np.any(norms < 0):
        raise ValueError("row_norm_uppers must be finite and non-negative")
    if not np.all(np.isfinite(rounding)) or np.any(rounding < 0):
        raise ValueError("rounding bounds must be finite and non-negative")
    if not math.isfinite(hidden_radius) or hidden_radius < 0:
        raise ValueError("hidden_radius must be finite and non-negative")

    winner = int(np.argmax(logits))
    errors = np.asarray(
        [outward_sum(outward_product(float(norm), hidden_radius), float(extra)) for norm, extra in zip(norms, rounding)],
        dtype=np.float64,
    )
    lower_winner = _down_signed(float(logits[winner] - errors[winner]))
    margin_lowers = [
        _down_signed(
            lower_winner
            - _up_signed(float(logits[index] + errors[index]))
        )
        for index in range(logits.size)
        if index != winner
    ]
    minimum = min(margin_lowers)
    return Top1OutwardCertificate(
        winner=winner,
        certified=minimum > 0.0,
        minimum_margin_lower=minimum,
    )


def verify_exact_spectral_upper_bound(
    weight: Sequence[Sequence[float]] | np.ndarray,
    bound: float,
    *,
    maximum_columns: int = 32,
) -> bool:
    """Exact-dyadic LDL proof that ``bound > ||weight||_2`` for controls.

    This intentionally small reference proves the certificate logic without
    pretending to be the target compiler.  The registered large-matrix method
    is outward ``A.T @ A`` plus a positive-definiteness verifier; its full
    cubic cost is charged separately.
    """

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a nonempty finite matrix")
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    # The nonzero singular values are identical for W.T @ W and W @ W.T.
    # Orient the matrix so the exact Gram proof uses the smaller dimension.
    oriented = matrix if matrix.shape[1] <= matrix.shape[0] else matrix.T
    if oriented.shape[1] > maximum_columns:
        raise ValueError("exact control verifier is intentionally dimension-bounded")
    rows = [[Fraction.from_float(float(value)) for value in row] for row in oriented]
    beta = Fraction.from_float(float(bound))
    columns = oriented.shape[1]
    gram = [
        [sum((row[i] * row[j] for row in rows), Fraction(0)) for j in range(columns)]
        for i in range(columns)
    ]
    certificate = [
        [
            (beta * beta if i == j else Fraction(0)) - gram[i][j]
            for j in range(columns)
        ]
        for i in range(columns)
    ]
    lower = [[Fraction(0) for _ in range(columns)] for _ in range(columns)]
    diagonal = [Fraction(0) for _ in range(columns)]
    for j in range(columns):
        pivot = certificate[j][j] - sum(
            lower[j][k] * lower[j][k] * diagonal[k] for k in range(j)
        )
        if pivot <= 0:
            return False
        diagonal[j] = pivot
        lower[j][j] = Fraction(1)
        for i in range(j + 1, columns):
            numerator = certificate[i][j] - sum(
                lower[i][k] * lower[j][k] * diagonal[k] for k in range(j)
            )
            lower[i][j] = numerator / pivot
    return True


@dataclass(frozen=True)
class VerifiedSpectralBound:
    operator_norm_upper: float
    operator_norm_squared_lower: float
    gram_absolute_row_upper: float
    positive_definite_margin_lower: float
    frobenius_norm_upper: float
    gram_dimension: int
    contracted_dimension: int
    orientation: str
    dot_gamma_upper: float
    gram_center_sha256: str
    gram_radius_sha256: str
    proof_method: str = "outward_gram_strict_diagonal_dominance"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerifiedCholeskySpectralBound:
    operator_norm_upper: float
    operator_norm_squared_lower: float
    proposed_operator_norm: float
    proposal_multiplier: float
    positive_definite_margin_lower: float
    gram_interval_operator_radius_upper: float
    cholesky_residual_operator_upper: float
    cholesky_minimum_eigenvalue_lower: float
    inverse_residual_operator_upper: float
    inverse_operator_norm_upper: float
    frobenius_norm_upper: float
    gram_dimension: int
    contracted_dimension: int
    orientation: str
    verification_attempts: int
    gram_center_sha256: str
    gram_radius_sha256: str
    cholesky_factor_sha256: str
    proof_method: str = "outward_gram_verified_cholesky_residual"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _matmul_center_and_radius(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 matrix product and an entrywise outward error radius."""

    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or not left.size
        or not right.size
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("malformed finite matrix product")
    contracted = left.shape[1]
    gamma = _float64_gamma(2 * contracted)
    center = left @ right
    absolute_center = np.abs(left) @ np.abs(right)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(absolute_center)):
        raise ValueError("matrix product overflowed")
    absolute_exact_upper = np.nextafter(
        absolute_center / (1.0 - gamma),
        np.inf,
    )
    radius = np.nextafter(
        np.nextafter(gamma * absolute_exact_upper, np.inf)
        + contracted * float(np.nextafter(0.0, 1.0)),
        np.inf,
    )
    return np.ascontiguousarray(center), np.ascontiguousarray(radius)


def matrix_product_difference_frobenius_upper(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    observed: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Enclose ``||left @ right - observed||_F`` with outward rounding."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    center, radius = _matmul_center_and_radius(left_array, right_array)
    if observed_array.shape != center.shape or not np.all(
        np.isfinite(observed_array)
    ):
        raise ValueError("observed product must be finite and shape-aligned")
    subtraction = np.nextafter(
        _float64_gamma(1)
        * np.nextafter(np.abs(center) + np.abs(observed_array), np.inf),
        np.inf,
    )
    absolute_upper = np.nextafter(
        np.abs(center - observed_array) + radius + subtraction,
        np.inf,
    )
    return array_frobenius_norm_upper(absolute_upper)


def round_to_bfloat16_values(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return exact float64 values produced by IEEE BF16 round-to-nearest-even."""

    array = np.asarray(values, dtype=np.float32)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("BF16 storage input must be nonempty and finite")
    bits = np.ascontiguousarray(array).view(np.uint32)
    least_kept_bit = (bits >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + least_kept_bit
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return np.ascontiguousarray(rounded.view(np.float32).astype(np.float64))


def _matrix_operator_norm_from_absolute_upper(
    absolute_upper: np.ndarray,
) -> float:
    """Bound ``||A||_2`` from an entrywise nonnegative upper matrix."""

    matrix = np.asarray(absolute_upper, dtype=np.float64)
    if (
        matrix.ndim != 2
        or not matrix.size
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
    ):
        raise ValueError("operator upper matrix must be finite and nonnegative")
    row_sums = np.asarray(
        [
            math.nextafter(math.fsum(float(value) for value in row), math.inf)
            for row in matrix
        ],
        dtype=np.float64,
    )
    column_sums = np.asarray(
        [
            math.nextafter(
                math.fsum(float(value) for value in matrix[:, index]),
                math.inf,
            )
            for index in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    product = outward_product(float(np.max(row_sums)), float(np.max(column_sums)))
    return _up(math.sqrt(product))


def verified_cholesky_spectral_upper_bound(
    weight: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_multiplier: float = 1.000001,
    multiplier_step: float = 1.01,
    maximum_attempts: int = 32,
) -> VerifiedCholeskySpectralBound:
    """Propose with LAPACK, then independently prove ``beta > ||W||_2``.

    The numerical eigensolver is not trusted as a certificate.  The proof
    encloses the smaller exact Gram matrix, factors the proposed shifted
    center, bounds the Cholesky residual, and verifies a lower singular-value
    bound for the factor through an approximate inverse residual.  Weyl's
    inequality then proves the exact shifted Gram positive definite.
    """

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a nonempty finite matrix")
    if (
        not math.isfinite(initial_multiplier)
        or initial_multiplier <= 1.0
        or not math.isfinite(multiplier_step)
        or multiplier_step <= 1.0
        or maximum_attempts <= 0
    ):
        raise ValueError("malformed verified-Cholesky search parameters")
    if matrix.shape[0] <= matrix.shape[1]:
        oriented = np.ascontiguousarray(matrix)
        orientation = "W_W_transpose"
    else:
        oriented = np.ascontiguousarray(matrix.T)
        orientation = "W_transpose_W"
    gram_dimension, contracted = oriented.shape
    gram, gram_radius = _matmul_center_and_radius(oriented, oriented.T)
    gram_interval_operator = _matrix_operator_norm_from_absolute_upper(
        gram_radius
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    proposed = _up(math.sqrt(max(0.0, float(eigenvalues[-1]))))
    frobenius_upper = array_frobenius_norm_upper(matrix)

    identity = np.eye(gram_dimension, dtype=np.float64)
    multiplier = initial_multiplier
    last_reason = "no attempt"
    for attempt in range(1, maximum_attempts + 1):
        beta = _up(proposed * multiplier)
        beta_square_center = beta * beta
        beta_gamma = _float64_gamma(1)
        beta_square_absolute_upper = _up(
            abs(beta_square_center) / (1.0 - beta_gamma)
        )
        beta_square_radius = outward_sum(
            outward_product(beta_gamma, beta_square_absolute_upper),
            float(np.nextafter(0.0, 1.0)),
        )
        shifted = np.ascontiguousarray(
            beta_square_center * identity - gram
        )
        try:
            lower = np.linalg.cholesky(shifted)
        except np.linalg.LinAlgError:
            last_reason = "center Cholesky failed"
            multiplier *= multiplier_step
            continue
        if not np.all(np.isfinite(lower)):
            last_reason = "non-finite Cholesky factor"
            multiplier *= multiplier_step
            continue

        lower_product, lower_product_radius = _matmul_center_and_radius(
            lower,
            lower.T,
        )
        subtraction_radius = np.nextafter(
            _float64_gamma(2)
            * np.nextafter(np.abs(shifted) + np.abs(lower_product), np.inf),
            np.inf,
        )
        exact_residual_upper = np.nextafter(
            np.abs(shifted - lower_product)
            + lower_product_radius
            + subtraction_radius
            + gram_radius,
            np.inf,
        )
        diagonal = np.diag_indices(gram_dimension)
        shifted_formation_radius = np.nextafter(
            beta_gamma
            * np.nextafter(
                (
                    abs(beta_square_center)
                    + np.abs(np.diag(gram))
                )
                / (1.0 - beta_gamma),
                np.inf,
            ),
            np.inf,
        )
        exact_residual_upper[diagonal] = np.nextafter(
            exact_residual_upper[diagonal]
            + beta_square_radius
            + shifted_formation_radius,
            np.inf,
        )
        cholesky_residual_operator = (
            _matrix_operator_norm_from_absolute_upper(exact_residual_upper)
        )

        approximate_inverse = np.linalg.inv(lower)
        if not np.all(np.isfinite(approximate_inverse)):
            last_reason = "non-finite approximate inverse"
            multiplier *= multiplier_step
            continue
        inverse_product, inverse_product_radius = _matmul_center_and_radius(
            approximate_inverse,
            lower,
        )
        inverse_subtraction_radius = np.nextafter(
            _float64_gamma(1)
            * np.nextafter(np.abs(identity) + np.abs(inverse_product), np.inf),
            np.inf,
        )
        inverse_residual_upper = np.nextafter(
            np.abs(identity - inverse_product)
            + inverse_product_radius
            + inverse_subtraction_radius,
            np.inf,
        )
        inverse_residual_operator = (
            _matrix_operator_norm_from_absolute_upper(inverse_residual_upper)
        )
        if inverse_residual_operator >= 1.0:
            last_reason = "inverse residual did not contract"
            multiplier *= multiplier_step
            continue
        inverse_abs = np.nextafter(np.abs(approximate_inverse), np.inf)
        inverse_operator = _matrix_operator_norm_from_absolute_upper(inverse_abs)
        contraction_lower = _down_nonnegative(
            1.0 - inverse_residual_operator
        )
        singular_lower = _down_nonnegative(
            contraction_lower / inverse_operator
        )
        factor_eigenvalue_lower = _down_nonnegative(
            singular_lower * singular_lower
        )
        margin = _down_nonnegative(
            max(0.0, factor_eigenvalue_lower - cholesky_residual_operator)
        )
        if margin <= 0.0:
            last_reason = "factor lower bound did not dominate residual"
            multiplier *= multiplier_step
            continue
        beta_squared_lower = _down_nonnegative(beta * beta)
        if beta_squared_lower <= 0.0:
            raise ValueError("verified beta square vanished")
        return VerifiedCholeskySpectralBound(
            operator_norm_upper=beta,
            operator_norm_squared_lower=beta_squared_lower,
            proposed_operator_norm=proposed,
            proposal_multiplier=multiplier,
            positive_definite_margin_lower=margin,
            gram_interval_operator_radius_upper=gram_interval_operator,
            cholesky_residual_operator_upper=cholesky_residual_operator,
            cholesky_minimum_eigenvalue_lower=factor_eigenvalue_lower,
            inverse_residual_operator_upper=inverse_residual_operator,
            inverse_operator_norm_upper=inverse_operator,
            frobenius_norm_upper=frobenius_upper,
            gram_dimension=gram_dimension,
            contracted_dimension=contracted,
            orientation=orientation,
            verification_attempts=attempt,
            gram_center_sha256=_array_sha256(gram),
            gram_radius_sha256=_array_sha256(gram_radius),
            cholesky_factor_sha256=_array_sha256(lower),
        )
    raise ValueError(
        "could not verify shifted Gram after "
        f"{maximum_attempts} attempts: {last_reason}"
    )


def verified_gram_gershgorin_upper_bound(
    weight: Sequence[Sequence[float]] | np.ndarray,
) -> VerifiedSpectralBound:
    """Prove ``beta > ||W||_2`` from an outward Gram inclusion.

    This static reference forms the smaller Gram orientation.  Every dot
    product receives a float64 ``gamma_(2n)`` enclosure, and strict interval
    diagonal dominance proves ``beta^2 I - G`` positive definite.  The proof
    is deliberately independent of a plain numerical SVD.
    """

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a nonempty finite matrix")
    if matrix.shape[0] <= matrix.shape[1]:
        oriented = np.ascontiguousarray(matrix)
        orientation = "W_W_transpose"
    else:
        oriented = np.ascontiguousarray(matrix.T)
        orientation = "W_transpose_W"
    gram_dimension, contracted = oriented.shape
    gamma = _float64_gamma(2 * contracted)
    gram = oriented @ oriented.T
    absolute_dot_sums = np.abs(oriented) @ np.abs(oriented).T
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(absolute_dot_sums)):
        raise ValueError("Gram construction overflowed")
    absolute_dot_uppers = np.nextafter(
        absolute_dot_sums / (1.0 - gamma),
        np.inf,
    )
    radius_products = np.nextafter(gamma * absolute_dot_uppers, np.inf)
    radii = np.nextafter(
        radius_products
        + contracted * float(np.nextafter(0.0, 1.0)),
        np.inf,
    )
    absolute_gram_uppers = np.nextafter(np.abs(gram) + radii, np.inf)
    row_uppers = np.asarray(
        [
            math.nextafter(math.fsum(float(value) for value in row), math.inf)
            for row in absolute_gram_uppers
        ],
        dtype=np.float64,
    )
    maximum_row_upper = float(np.max(row_uppers))
    beta = math.nextafter(math.sqrt(maximum_row_upper), math.inf)
    for _ in range(16):
        beta_squared_lower = _down_nonnegative(beta * beta)
        if beta_squared_lower > maximum_row_upper:
            break
        beta = math.nextafter(beta, math.inf)
    else:
        raise ValueError("could not separate the verified spectral bound")
    margin = _down_nonnegative(beta_squared_lower - maximum_row_upper)
    if margin <= 0.0:
        raise ValueError("verified positive-definite margin is not strict")

    diagonal_uppers = np.nextafter(
        np.diag(gram) + np.diag(radii),
        np.inf,
    )
    trace_upper = math.nextafter(
        math.fsum(float(value) for value in diagonal_uppers),
        math.inf,
    )
    frobenius_upper = math.nextafter(math.sqrt(trace_upper), math.inf)
    return VerifiedSpectralBound(
        operator_norm_upper=beta,
        operator_norm_squared_lower=beta_squared_lower,
        gram_absolute_row_upper=maximum_row_upper,
        positive_definite_margin_lower=margin,
        frobenius_norm_upper=frobenius_upper,
        gram_dimension=gram_dimension,
        contracted_dimension=contracted,
        orientation=orientation,
        dot_gamma_upper=gamma,
        gram_center_sha256=_array_sha256(gram),
        gram_radius_sha256=_array_sha256(radii),
    )


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    clean = sorted(float(value) for value in values)
    if not clean or not all(math.isfinite(value) for value in clean):
        raise ValueError("percentile input must be nonempty and finite")
    index = int(math.ceil(quantile * len(clean)) - 1)
    return clean[max(0, min(index, len(clean) - 1))]


def summarize_legal_pair_outward_gate(
    token_rows: Sequence[dict[str, object]],
    *,
    expected_token_states: int,
    expected_families: Sequence[str],
    prompts_per_family: int,
    maximum_mean_kl: float,
    maximum_p95_kl: float,
    control_failures: Sequence[str] = (),
    leakage_failures: Sequence[str] = (),
    malformed_state_count: int = 0,
    execution_complete: bool = False,
) -> dict[str, object]:
    """Aggregate the frozen zero-fallback one-shot legal Gate."""

    rows = [dict(row) for row in token_rows]
    families = tuple(str(value) for value in expected_families)
    successes = sum(bool(row.get("success")) for row in rows)
    certified = sum(bool(row.get("certificate_resolved")) for row in rows)
    fallbacks = sum(bool(row.get("fallback_executed")) for row in rows)
    false_accepts = sum(bool(row.get("false_accept")) for row in rows)
    exact_completion_failures = sum(
        not bool(row.get("dense_completion_matches_baseline", False))
        for row in rows
        if row.get("fallback_executed")
    )
    family_successes = {
        family: sum(
            bool(row.get("success")) and str(row.get("family")) == family
            for row in rows
        )
        for family in families
    }
    kls = [
        float(row["target_to_candidate_kl"])
        for row in rows
        if row.get("target_to_candidate_kl") is not None
    ]
    mean_kl = math.fsum(kls) / len(kls) if kls else None
    p95_kl = nearest_rank_percentile(kls, 0.95) if kls else None
    controls_clean = (
        not control_failures
        and not leakage_failures
        and malformed_state_count == 0
        and false_accepts == 0
        and exact_completion_failures == 0
    )
    gates = {
        "execution_complete": execution_complete,
        "token_successes": successes == expected_token_states,
        "certified_tokens": certified == expected_token_states,
        "family_successes": all(
            family_successes[family] == prompts_per_family
            for family in families
        ),
        "zero_fallback": fallbacks == 0,
        "zero_false_accept": false_accepts == 0,
        "exact_dense_completion": exact_completion_failures == 0,
        "mean_kl_within_limit": (
            mean_kl is not None and mean_kl <= maximum_mean_kl
        ),
        "p95_kl_within_limit": (
            p95_kl is not None and p95_kl <= maximum_p95_kl
        ),
        "controls_clean": controls_clean,
    }
    scientific_failure = any(
        bool(row.get("scientific_failure")) for row in rows
    )
    if not controls_clean:
        decision = INVALID_LEGAL_DECISION
    elif scientific_failure:
        decision = REJECT_LEGAL_DECISION
    elif execution_complete:
        decision = (
            PROMOTE_LEGAL_DECISION
            if all(gates.values())
            else REJECT_LEGAL_DECISION
        )
    else:
        decision = INFRASTRUCTURE_LEGAL_DECISION
    return {
        "decision": decision,
        "execution_complete": execution_complete,
        "observed_scientific_failure": scientific_failure,
        "evaluated_token_states": len(rows),
        "token_successes": successes,
        "certified_tokens": certified,
        "fallbacks": fallbacks,
        "false_accepts": false_accepts,
        "exact_completion_failures": exact_completion_failures,
        "family_successes": family_successes,
        "mean_target_to_candidate_kl": mean_kl,
        "p95_target_to_candidate_kl": p95_kl,
        "control_failures": list(control_failures),
        "leakage_failures": list(leakage_failures),
        "malformed_state_count": malformed_state_count,
        "gates": gates,
    }


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _verified_spectral_compile_operations() -> int:
    """Registered Rump-style Gram inclusion plus two-Cholesky charge.

    For a rectangular ``large x small`` matrix, outward inclusion of the Gram
    matrix is charged as ``4*large*small^2`` scalar operations.  The two
    positive-definiteness Cholesky passes are charged as ``2/3*small^3`` plus
    explicit quadratic/linear bookkeeping.  The future implementation must
    report its measured count and may not substitute this formula if larger.
    """

    total = 0
    for family in REGISTERED_MATRIX_FAMILIES:
        small = min(family.rows, family.columns)
        large = max(family.rows, family.columns)
        per_matrix = (
            4 * large * small * small
            + (2 * small**3 + 2) // 3
            + 10 * small * small
            + 10 * small
        )
        total += family.count * per_matrix
    return total


@dataclass(frozen=True)
class LegalPairOutwardAccounting:
    proof_metadata_scalars: int
    proof_metadata_bytes: int
    proof_metadata_traffic_fraction: Decimal
    pair_build_operations: int
    pair_build_dense_equivalents: Decimal
    pair_build_traffic_bytes: int
    pair_build_traffic_dense_equivalents: Decimal
    proof_operations_per_token: int
    dynamic_traffic_fraction: Decimal
    dynamic_operation_fraction: Decimal
    spectral_compile_operations: int
    spectral_compile_dense_equivalents: Decimal
    spectral_compile_traffic_dense_equivalents: Decimal
    model_service_tokens: int
    charged_traffic_fraction: Decimal
    charged_operation_fraction: Decimal
    minimum_traffic_coverage: Decimal | None
    minimum_operation_coverage: Decimal | None
    controlling_minimum_coverage: Decimal | None
    minimum_static_service_tokens_traffic: int | None
    minimum_static_service_tokens_operations: int | None
    capsule_and_metadata_bytes: int
    capsule_and_metadata_gib: Decimal
    unallocated_hot_gib: Decimal

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        return {key: str(value) if isinstance(value, Decimal) else value for key, value in row.items()}


def derive_legal_pair_outward_accounting(
    *,
    model_service_tokens: int = 20_000_000,
) -> LegalPairOutwardAccounting:
    if model_service_tokens <= 0:
        raise ValueError("model_service_tokens must be positive")
    base = derive_atlas_budget(
        rank=16,
        requested_cold_fraction=Decimal("0.002"),
        page_columns=64,
        capsule_scalar_bytes=2,
        metadata_bytes_per_block=4,
        service_tokens=64,
    )
    proof_scalars = (
        3 * MATRIX_INSTANCES
        + 2 * ACTIVATION_SITE_INSTANCES
        + LM_HEAD_ROWS
    )
    proof_bytes = proof_scalars * 8
    proof_traffic = Decimal(proof_bytes) / Decimal(REGISTERED_NON_EMBEDDING_Q4_BYTES)

    # Two-pass causal MGS and its identical pair-image recurrence.  This is a
    # complete charged replacement for the earlier one-pass minimum build.
    pair_build_ops = 2 * base.minimum_build_operations
    pair_build_equiv = Decimal(pair_build_ops) / Decimal(REGISTERED_NON_EMBEDDING_COEFFICIENTS)
    # Read the 16 selected committed pairs once and write Q/Z once.  The two
    # populations have the same element count as the stored capsule.
    pair_build_traffic = 2 * base.capsule_bytes
    pair_build_traffic_equiv = Decimal(pair_build_traffic) / Decimal(REGISTERED_NON_EMBEDDING_Q4_BYTES)

    proof_ops = (
        8 * MATRIX_INSTANCES
        + 4 * ACTIVATION_SITE_INSTANCES
        + 2 * LM_HEAD_ROWS
    )
    dynamic_traffic = (
        base.fast_traffic_fraction
        + pair_build_traffic_equiv / Decimal(base.service_tokens)
        + proof_traffic
    )
    dynamic_operations = (
        base.fast_operation_fraction
        + pair_build_equiv / Decimal(base.service_tokens)
        + Decimal(proof_ops) / Decimal(REGISTERED_NON_EMBEDDING_COEFFICIENTS)
    )

    compile_ops = _verified_spectral_compile_operations()
    compile_equiv = Decimal(compile_ops) / Decimal(REGISTERED_NON_EMBEDDING_COEFFICIENTS)
    compile_traffic_equiv = Decimal(1) + Decimal(proof_bytes) / Decimal(REGISTERED_NON_EMBEDDING_Q4_BYTES)
    charged_traffic = dynamic_traffic + compile_traffic_equiv / Decimal(model_service_tokens)
    charged_operations = dynamic_operations + compile_equiv / Decimal(model_service_tokens)
    traffic_coverage = minimum_coverage(P50_TARGET_FRACTION, ResourceTerms(common=charged_traffic))
    operation_coverage = minimum_coverage(P50_TARGET_FRACTION, ResourceTerms(common=charged_operations))
    coverages = [value for value in (traffic_coverage, operation_coverage) if value is not None]
    controlling = max(coverages) if len(coverages) == 2 else None

    def service_floor(dynamic: Decimal, static: Decimal) -> int | None:
        headroom = P50_TARGET_FRACTION - dynamic
        return None if headroom <= 0 else _ceil_decimal(static / headroom)

    state_bytes = base.capsule_bytes + base.metadata_bytes + proof_bytes
    return LegalPairOutwardAccounting(
        proof_metadata_scalars=proof_scalars,
        proof_metadata_bytes=proof_bytes,
        proof_metadata_traffic_fraction=proof_traffic,
        pair_build_operations=pair_build_ops,
        pair_build_dense_equivalents=pair_build_equiv,
        pair_build_traffic_bytes=pair_build_traffic,
        pair_build_traffic_dense_equivalents=pair_build_traffic_equiv,
        proof_operations_per_token=proof_ops,
        dynamic_traffic_fraction=dynamic_traffic,
        dynamic_operation_fraction=dynamic_operations,
        spectral_compile_operations=compile_ops,
        spectral_compile_dense_equivalents=compile_equiv,
        spectral_compile_traffic_dense_equivalents=compile_traffic_equiv,
        model_service_tokens=model_service_tokens,
        charged_traffic_fraction=charged_traffic,
        charged_operation_fraction=charged_operations,
        minimum_traffic_coverage=traffic_coverage,
        minimum_operation_coverage=operation_coverage,
        controlling_minimum_coverage=controlling,
        minimum_static_service_tokens_traffic=service_floor(dynamic_traffic, compile_traffic_equiv),
        minimum_static_service_tokens_operations=service_floor(dynamic_operations, compile_equiv),
        capsule_and_metadata_bytes=state_bytes,
        capsule_and_metadata_gib=Decimal(state_bytes) / Decimal(GIB),
        unallocated_hot_gib=Decimal(8 * GIB - state_bytes) / Decimal(GIB),
    )


@dataclass(frozen=True)
class LegalPairOutwardGate:
    rank: int
    page_columns: int
    layer_index: int
    projection: str
    teacher_position_index: int
    evaluation_prompts: int
    families: int
    prompts_per_family: int
    prompt_path: str
    prompt_sha256: str
    required_token_successes: int
    required_family_successes: int
    maximum_failures: int
    maximum_mean_kl: Decimal
    maximum_p95_kl: Decimal
    minimum_coverage: Decimal
    selector: str
    declared_output: str
    accounting: LegalPairOutwardAccounting

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["maximum_mean_kl"] = str(self.maximum_mean_kl)
        row["maximum_p95_kl"] = str(self.maximum_p95_kl)
        row["minimum_coverage"] = str(self.minimum_coverage)
        row["accounting"] = self.accounting.to_dict()
        return row


def derive_legal_pair_outward_gate() -> LegalPairOutwardGate:
    accounting = derive_legal_pair_outward_accounting()
    if accounting.controlling_minimum_coverage is None:
        raise ValueError("registered legal Gate has no feasible resource coverage")
    coverage = accounting.controlling_minimum_coverage
    population = 24
    per_family = 4
    required = _ceil_decimal(Decimal(population) * coverage)
    family_required = _ceil_decimal(Decimal(per_family) * coverage)
    return LegalPairOutwardGate(
        rank=16,
        page_columns=64,
        layer_index=23,
        projection="down_proj",
        teacher_position_index=1,
        evaluation_prompts=population,
        families=6,
        prompts_per_family=per_family,
        prompt_path=LEGAL_PROMPT_PATH,
        prompt_sha256=LEGAL_PROMPT_SHA256,
        required_token_successes=required,
        required_family_successes=family_required,
        maximum_failures=population - required,
        maximum_mean_kl=Decimal("0.02"),
        maximum_p95_kl=Decimal("0.05"),
        minimum_coverage=coverage,
        selector="maximum_current_residual_page_energy_then_lowest_index",
        declared_output="pinned_native_bf16_greedy_top1",
        accounting=accounting,
    )
