"""Pure certificate assembly for the EXP-083B legal last-down Gate.

The pair compiler in this module has no weight argument.  Current-query
helpers accept only pair-derived state, the current activation, and the one
weight page selected by the separately frozen target-free selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from vortex_runtime.causal_residual_atlas_legal_gate import (
    array_frobenius_norm_upper,
    bf16_addition_l2_bound_from_norms,
    bf16_vector_addition_l2_bound,
    certified_unread_radius,
    compile_pair_only_mgs,
    matrix_product_difference_frobenius_upper,
    native_linear_l2_rounding_bound,
    outward_product,
    outward_squared_norm,
    outward_sum,
    pair_image_defect_bound,
    projection_output_radius,
    rmsnorm_native_implementation_l2_bound,
    rmsnorm_output_radius,
    round_to_bfloat16_values,
)


def vector_norm_upper(values: Sequence[float] | np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError("norm input must be a nonempty finite vector")
    return math.nextafter(math.sqrt(outward_squared_norm(vector)), math.inf)


def vector_norm_lower(values: Sequence[float] | np.ndarray) -> float:
    """Conservative lower bound using an upper error on a float64 norm."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError("norm input must be a nonempty finite vector")
    computed = float(np.linalg.norm(vector))
    gamma = (2 * vector.size * 2.0**-53) / (
        1.0 - 2 * vector.size * 2.0**-53
    )
    return max(
        0.0,
        math.nextafter(computed / math.sqrt(1.0 + gamma), -math.inf),
    )


@dataclass(frozen=True)
class StoredPairCertificate:
    stored_basis: np.ndarray
    stored_image: np.ndarray
    coefficient_map: np.ndarray
    accepted_prefix_indices: tuple[int, ...]
    accepted_residual_norm_lowers: tuple[float, ...]
    accepted_threshold_uppers: tuple[float, ...]
    prefix_native_error_bounds: tuple[float, ...]
    prefix_image_error_frobenius: float
    coefficient_operator_upper: float
    stored_basis_relation_error_operator: float
    stored_image_relation_error_operator: float
    pair_image_defect_operator: float

    @property
    def rank(self) -> int:
        return int(self.stored_basis.shape[1])

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("stored_basis", "stored_image", "coefficient_map"):
            array = np.asarray(payload.pop(key))
            payload[f"{key}_shape"] = list(array.shape)
        payload["rank"] = self.rank
        return payload


def compile_stored_pair_certificate(
    prefix_inputs: Sequence[Sequence[float]] | np.ndarray,
    prefix_images: Sequence[Sequence[float]] | np.ndarray,
    *,
    maximum_rank: int,
    operator_norm_upper: float,
    frobenius_norm_upper: float,
) -> StoredPairCertificate:
    """Compile BF16 pair state and a defect bound without a weight argument."""

    inputs = np.asarray(prefix_inputs, dtype=np.float64)
    images = np.asarray(prefix_images, dtype=np.float64)
    pair = compile_pair_only_mgs(inputs, images, maximum_rank=maximum_rank)
    stored_basis = round_to_bfloat16_values(pair.basis)
    stored_image = round_to_bfloat16_values(pair.image)

    input_norms = [vector_norm_upper(row) for row in inputs]
    native_bounds = tuple(
        native_linear_l2_rounding_bound(
            input_norm_upper=norm,
            frobenius_norm_upper=frobenius_norm_upper,
            operator_norm_upper=operator_norm_upper,
            rows=images.shape[1],
            columns=inputs.shape[1],
        )
        for norm in input_norms
    )
    prefix_error_frobenius = vector_norm_upper(native_bounds)
    coefficient_operator = array_frobenius_norm_upper(pair.coefficient_map)
    basis_relation_error = matrix_product_difference_frobenius_upper(
        inputs.T,
        pair.coefficient_map,
        stored_basis,
    )
    image_relation_error = matrix_product_difference_frobenius_upper(
        images.T,
        pair.coefficient_map,
        stored_image,
    )
    image_defect = pair_image_defect_bound(
        operator_norm_upper=operator_norm_upper,
        stored_basis_error_operator=basis_relation_error,
        prefix_image_error_frobenius=prefix_error_frobenius,
        pair_coefficient_operator=coefficient_operator,
        stored_image_error_operator=image_relation_error,
    )
    return StoredPairCertificate(
        stored_basis=stored_basis,
        stored_image=stored_image,
        coefficient_map=np.ascontiguousarray(pair.coefficient_map),
        accepted_prefix_indices=pair.accepted_prefix_indices,
        accepted_residual_norm_lowers=pair.accepted_residual_norm_lowers,
        accepted_threshold_uppers=pair.accepted_threshold_uppers,
        prefix_native_error_bounds=native_bounds,
        prefix_image_error_frobenius=prefix_error_frobenius,
        coefficient_operator_upper=coefficient_operator,
        stored_basis_relation_error_operator=basis_relation_error,
        stored_image_relation_error_operator=image_relation_error,
        pair_image_defect_operator=image_defect,
    )


@dataclass(frozen=True)
class ProjectionRadiusBreakdown:
    coordinate_norm_upper: float
    basis_apply_rounding_radius: float
    residual_subtraction_rounding_radius: float
    decomposition_radius: float
    pair_image_radius: float
    unread_radius: float
    stored_image_apply_rounding_radius: float
    selected_page_apply_rounding_radius: float
    candidate_addition_rounding_radius: float
    candidate_arithmetic_radius: float
    native_dense_rounding_radius: float
    projection_output_radius: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def bound_legal_projection_candidate(
    *,
    stored_basis: Sequence[Sequence[float]] | np.ndarray,
    stored_image: Sequence[Sequence[float]] | np.ndarray,
    pair_image_defect: float,
    operator_norm_upper: float,
    frobenius_norm_upper: float,
    current_input: Sequence[float] | np.ndarray,
    coordinates: Sequence[float] | np.ndarray,
    basis_application: Sequence[float] | np.ndarray,
    residual: Sequence[float] | np.ndarray,
    selected_page: int,
    page_columns: int,
    selected_weight_page: Sequence[Sequence[float]] | np.ndarray,
    stored_image_application: Sequence[float] | np.ndarray,
    selected_page_application: Sequence[float] | np.ndarray,
) -> ProjectionRadiusBreakdown:
    """Enclose the selected-page BF16 candidate against native dense output."""

    basis = np.asarray(stored_basis, dtype=np.float64)
    image = np.asarray(stored_image, dtype=np.float64)
    x = np.asarray(current_input, dtype=np.float64)
    a = np.asarray(coordinates, dtype=np.float64)
    q_a = np.asarray(basis_application, dtype=np.float64)
    u = np.asarray(residual, dtype=np.float64)
    page = np.asarray(selected_weight_page, dtype=np.float64)
    z_a = np.asarray(stored_image_application, dtype=np.float64)
    page_u = np.asarray(selected_page_application, dtype=np.float64)
    if basis.shape != (x.size, a.size):
        raise ValueError("stored basis shape mismatch")
    if image.shape[1] != a.size or image.shape[0] != page.shape[0]:
        raise ValueError("stored image shape mismatch")
    if q_a.shape != x.shape or u.shape != x.shape:
        raise ValueError("decomposition vector shape mismatch")
    if z_a.shape != page_u.shape or z_a.size != page.shape[0]:
        raise ValueError("candidate output shape mismatch")
    if page.shape[1] <= 0 or page.shape[1] > page_columns:
        raise ValueError("selected page width mismatch")

    a_norm = vector_norm_upper(a)
    basis_frobenius = array_frobenius_norm_upper(basis)
    basis_apply = native_linear_l2_rounding_bound(
        input_norm_upper=a_norm,
        frobenius_norm_upper=basis_frobenius,
        operator_norm_upper=basis_frobenius,
        rows=basis.shape[0],
        columns=basis.shape[1],
    )
    subtraction = bf16_vector_addition_l2_bound(x, -q_a)
    decomposition = outward_sum(basis_apply, subtraction)
    pair_radius = outward_product(pair_image_defect, a_norm)
    unread = certified_unread_radius(
        operator_norm_upper,
        u,
        page_columns=page_columns,
        selected_page=selected_page,
    )
    image_frobenius = array_frobenius_norm_upper(image)
    image_apply = native_linear_l2_rounding_bound(
        input_norm_upper=a_norm,
        frobenius_norm_upper=image_frobenius,
        operator_norm_upper=image_frobenius,
        rows=image.shape[0],
        columns=image.shape[1],
    )
    page_frobenius = array_frobenius_norm_upper(page)
    start = selected_page * page_columns
    page_input = u[start : start + page.shape[1]]
    page_apply = native_linear_l2_rounding_bound(
        input_norm_upper=vector_norm_upper(page_input),
        frobenius_norm_upper=page_frobenius,
        operator_norm_upper=page_frobenius,
        rows=page.shape[0],
        columns=page.shape[1],
    )
    candidate_add = bf16_vector_addition_l2_bound(z_a, page_u)
    candidate_arithmetic = outward_sum(image_apply, page_apply, candidate_add)
    native_dense = native_linear_l2_rounding_bound(
        input_norm_upper=vector_norm_upper(x),
        frobenius_norm_upper=frobenius_norm_upper,
        operator_norm_upper=operator_norm_upper,
        rows=image.shape[0],
        columns=x.size,
    )
    radius = projection_output_radius(
        pair_image_defect=pair_image_defect,
        coordinate_norm_upper=a_norm,
        unread_radius=outward_sum(
            outward_product(operator_norm_upper, decomposition),
            unread,
        ),
        selected_page_arithmetic_radius=candidate_arithmetic,
        native_dense_arithmetic_radius=native_dense,
    )
    return ProjectionRadiusBreakdown(
        coordinate_norm_upper=a_norm,
        basis_apply_rounding_radius=basis_apply,
        residual_subtraction_rounding_radius=subtraction,
        decomposition_radius=decomposition,
        pair_image_radius=pair_radius,
        unread_radius=unread,
        stored_image_apply_rounding_radius=image_apply,
        selected_page_apply_rounding_radius=page_apply,
        candidate_addition_rounding_radius=candidate_add,
        candidate_arithmetic_radius=candidate_arithmetic,
        native_dense_rounding_radius=native_dense,
        projection_output_radius=radius,
    )


@dataclass(frozen=True)
class FinalRadiusBreakdown:
    candidate_residual_addition_radius: float
    target_residual_addition_radius: float
    pre_norm_hidden_radius: float
    candidate_rmsnorm_implementation_radius: float
    target_rmsnorm_implementation_radius: float
    two_path_rmsnorm_implementation_radius: float
    final_hidden_radius: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def propagate_last_down_to_final_hidden(
    *,
    residual_branch: Sequence[float] | np.ndarray,
    candidate_down: Sequence[float] | np.ndarray,
    candidate_pre_norm: Sequence[float] | np.ndarray,
    projection_radius: float,
    gain_abs_max: float,
    epsilon: float,
) -> FinalRadiusBreakdown:
    """Propagate a down-projection radius through add and pinned RMSNorm."""

    residual = np.asarray(residual_branch, dtype=np.float64)
    candidate = np.asarray(candidate_down, dtype=np.float64)
    center = np.asarray(candidate_pre_norm, dtype=np.float64)
    if residual.ndim != 1 or candidate.shape != residual.shape or center.shape != residual.shape:
        raise ValueError("last-down residual state must be aligned vectors")
    if not math.isfinite(projection_radius) or projection_radius < 0:
        raise ValueError("projection radius must be finite and non-negative")

    candidate_add = bf16_vector_addition_l2_bound(residual, candidate)
    target_down_norm = outward_sum(
        vector_norm_upper(candidate),
        projection_radius,
    )
    target_add = bf16_addition_l2_bound_from_norms(
        left_norm_upper=vector_norm_upper(residual),
        right_norm_upper=target_down_norm,
        dimension=residual.size,
    )
    pre_norm_radius = outward_sum(
        projection_radius,
        candidate_add,
        target_add,
    )
    center_lower = vector_norm_lower(center)
    center_upper = vector_norm_upper(center)
    target_lower = max(0.0, center_lower - pre_norm_radius)
    target_upper = outward_sum(center_upper, pre_norm_radius)
    candidate_impl = rmsnorm_native_implementation_l2_bound(
        input_norm_lower=center_lower,
        input_norm_upper=center_upper,
        dimension=center.size,
        gain_abs_max=gain_abs_max,
        epsilon=epsilon,
    )
    target_impl = rmsnorm_native_implementation_l2_bound(
        input_norm_lower=target_lower,
        input_norm_upper=target_upper,
        dimension=center.size,
        gain_abs_max=gain_abs_max,
        epsilon=epsilon,
    )
    two_path = outward_sum(candidate_impl, target_impl)
    final_radius = rmsnorm_output_radius(
        center,
        input_radius=pre_norm_radius,
        gain_abs_max=gain_abs_max,
        epsilon=epsilon,
        two_path_cast_radius=two_path,
    )
    return FinalRadiusBreakdown(
        candidate_residual_addition_radius=candidate_add,
        target_residual_addition_radius=target_add,
        pre_norm_hidden_radius=pre_norm_radius,
        candidate_rmsnorm_implementation_radius=candidate_impl,
        target_rmsnorm_implementation_radius=target_impl,
        two_path_rmsnorm_implementation_radius=two_path,
        final_hidden_radius=final_radius,
    )
