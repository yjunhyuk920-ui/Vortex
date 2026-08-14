"""E0 reference and accounting for a Causal Residual Atlas.

The reference exposes two separate claims:

* exact committed-prefix input/image pairs define a causal control subspace;
* every component outside that subspace remains an explicit, bounded residual.

It is not a deployable Transformer executor.  In particular, a useful final
token certificate, physical cold-page service, and the rest of the 8 GiB
runtime have not been demonstrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import math
from typing import Iterable, Sequence

import numpy as np

from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    P50_TARGET_FRACTION,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    REGISTERED_NON_EMBEDDING_Q4_BYTES,
    ResourceTerms,
    minimum_coverage,
)


@dataclass(frozen=True)
class MatrixFamily:
    name: str
    rows: int
    columns: int
    count: int

    @property
    def coefficients(self) -> int:
        return self.rows * self.columns * self.count

    @property
    def output_rows(self) -> int:
        return self.rows * self.count


@dataclass(frozen=True)
class ActivationSite:
    name: str
    width: int
    count: int

    @property
    def input_dimensions(self) -> int:
        return self.width * self.count


REGISTERED_MATRIX_FAMILIES = (
    MatrixFamily("q_proj", 16_384, 16_384, 126),
    MatrixFamily("k_proj", 1_024, 16_384, 126),
    MatrixFamily("v_proj", 1_024, 16_384, 126),
    MatrixFamily("o_proj", 16_384, 16_384, 126),
    MatrixFamily("gate_proj", 53_248, 16_384, 126),
    MatrixFamily("up_proj", 53_248, 16_384, 126),
    MatrixFamily("down_proj", 16_384, 53_248, 126),
    MatrixFamily("lm_head", 128_256, 16_384, 1),
)

# Q/K/V and gate/up share their current input basis.  O, down, and the final
# head each have a distinct causal activation site.
REGISTERED_ACTIVATION_SITES = (
    ActivationSite("pre_attention_norm", 16_384, 126),
    ActivationSite("attention_output", 16_384, 126),
    ActivationSite("post_attention_norm", 16_384, 126),
    ActivationSite("mlp_intermediate", 53_248, 126),
    ActivationSite("final_hidden", 16_384, 1),
)

REGISTERED_OUTPUT_ROWS = sum(
    family.output_rows for family in REGISTERED_MATRIX_FAMILIES
)
REGISTERED_SHARED_SITE_INPUT_DIMENSIONS = sum(
    site.input_dimensions for site in REGISTERED_ACTIVATION_SITES
)


@dataclass(frozen=True)
class AtlasBudget:
    rank: int
    requested_cold_fraction: Decimal
    page_columns: int
    capsule_scalar_bytes: int
    metadata_bytes_per_block: int
    service_tokens: int
    capsule_elements: int
    capsule_bytes: int
    capsule_gib: Decimal
    metadata_blocks: int
    metadata_bytes: int
    selected_pages_per_token: int
    selected_coefficients_per_token: int
    actual_cold_fraction: Decimal
    capsule_traffic_bytes: int
    capsule_traffic_fraction: Decimal
    metadata_traffic_fraction: Decimal
    fast_traffic_fraction: Decimal
    minimum_build_traffic_bytes: int
    minimum_build_traffic_dense_equivalents: Decimal
    amortized_traffic_fraction: Decimal
    center_operations: int
    selector_bound_operations: int
    fast_operations: int
    fast_operation_fraction: Decimal
    minimum_build_operations: int
    minimum_build_operation_dense_equivalents: Decimal
    amortized_operation_fraction: Decimal
    minimum_traffic_coverage: Decimal | None
    minimum_operation_coverage: Decimal | None
    unallocated_hot_gib: Decimal

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in row.items()
        }


def _decimal(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def derive_atlas_budget(
    *,
    rank: int,
    requested_cold_fraction: Decimal | float | str,
    page_columns: int = 64,
    capsule_scalar_bytes: int = 2,
    metadata_bytes_per_block: int = 4,
    service_tokens: int = 64,
) -> AtlasBudget:
    """Return a favorable, logical 405B resource screen.

    The capsule contains one shared input basis per causal activation site and
    one image basis per matrix instance.  It is read once per logical token.
    Cold selection is rounded up independently for each matrix instance, so
    page granularity is never hidden.  The returned 8 GiB remainder is only
    unallocated headroom; KV, workspaces, buffers, and the fallback are absent.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    fraction = _decimal(requested_cold_fraction)
    if not Decimal(0) <= fraction <= Decimal(1):
        raise ValueError("requested_cold_fraction must be in [0, 1]")
    if page_columns <= 0:
        raise ValueError("page_columns must be positive")
    if capsule_scalar_bytes <= 0:
        raise ValueError("capsule_scalar_bytes must be positive")
    if metadata_bytes_per_block < 0:
        raise ValueError("metadata_bytes_per_block must be non-negative")
    if service_tokens <= 0:
        raise ValueError("service_tokens must be positive")

    capsule_elements = rank * (
        REGISTERED_OUTPUT_ROWS + REGISTERED_SHARED_SITE_INPUT_DIMENSIONS
    )
    capsule_bytes = capsule_elements * capsule_scalar_bytes

    metadata_blocks = 0
    selected_pages = 0
    selected_coefficients = 0
    for family in REGISTERED_MATRIX_FAMILIES:
        blocks = math.ceil(family.columns / page_columns)
        metadata_blocks += family.count * blocks
        selected = 0 if fraction == 0 else max(
            1, _ceil_decimal(fraction * Decimal(blocks))
        )
        selected = min(blocks, selected)
        selected_columns = min(family.columns, selected * page_columns)
        selected_pages += family.count * selected
        selected_coefficients += (
            family.count * family.rows * selected_columns
        )

    metadata_bytes = metadata_blocks * metadata_bytes_per_block
    actual_cold_fraction = Decimal(selected_coefficients) / Decimal(
        REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    # Q is read once to form Q^T x and again to form x - Qa after the
    # coordinates are known.  Z is read once.  The full Q capsule cannot be
    # retained in on-chip storage, so storage bytes and per-token read bytes
    # are deliberately different quantities.
    capsule_traffic_bytes = capsule_scalar_bytes * rank * (
        2 * REGISTERED_SHARED_SITE_INPUT_DIMENSIONS + REGISTERED_OUTPUT_ROWS
    )
    capsule_traffic_fraction = Decimal(capsule_traffic_bytes) / Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )
    metadata_traffic_fraction = Decimal(metadata_bytes) / Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )
    fast_traffic_fraction = (
        actual_cold_fraction
        + capsule_traffic_fraction
        + metadata_traffic_fraction
    )
    # At minimum, construction must write the capsule once.  Prefix activation
    # and image reads are favorably treated as already live during exact
    # prefill; a physical Gate must measure them rather than reuse this grant.
    minimum_build_traffic_bytes = capsule_bytes
    minimum_build_traffic_dense_equivalents = Decimal(
        minimum_build_traffic_bytes
    ) / Decimal(REGISTERED_NON_EMBEDDING_Q4_BYTES)
    amortized_traffic_fraction = fast_traffic_fraction + (
        minimum_build_traffic_dense_equivalents / Decimal(service_tokens)
    )

    # Q^T x and Q a are each charged once per shared activation site.  Z a is
    # charged for every matrix image.  A cheap lower-bound selector scans each
    # residual scalar for block norms and each matrix block for its score.
    center_operations = rank * (
        2 * REGISTERED_SHARED_SITE_INPUT_DIMENSIONS + REGISTERED_OUTPUT_ROWS
    )
    selector_bound_operations = (
        REGISTERED_SHARED_SITE_INPUT_DIMENSIONS + metadata_blocks
    )
    fast_operations = (
        center_operations
        + selector_bound_operations
        + selected_coefficients
    )
    fast_operation_fraction = Decimal(fast_operations) / Decimal(
        REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )

    triangular_rank = rank * (rank - 1) // 2
    minimum_build_operations = (
        REGISTERED_SHARED_SITE_INPUT_DIMENSIONS
        * (2 * triangular_rank + rank)
        + REGISTERED_OUTPUT_ROWS * (triangular_rank + rank)
    )
    minimum_build_operation_dense_equivalents = Decimal(
        minimum_build_operations
    ) / Decimal(REGISTERED_NON_EMBEDDING_COEFFICIENTS)
    amortized_operation_fraction = fast_operation_fraction + (
        minimum_build_operation_dense_equivalents / Decimal(service_tokens)
    )

    minimum_traffic = minimum_coverage(
        P50_TARGET_FRACTION,
        ResourceTerms(
            common=fast_traffic_fraction,
            compile_dense_equivalents=(
                minimum_build_traffic_dense_equivalents
            ),
            service_tokens=service_tokens,
        ),
    )
    minimum_operations = minimum_coverage(
        P50_TARGET_FRACTION,
        ResourceTerms(
            common=fast_operation_fraction,
            compile_dense_equivalents=(
                minimum_build_operation_dense_equivalents
            ),
            service_tokens=service_tokens,
        ),
    )
    unallocated_hot_bytes = 8 * GIB - capsule_bytes - metadata_bytes

    return AtlasBudget(
        rank=rank,
        requested_cold_fraction=fraction,
        page_columns=page_columns,
        capsule_scalar_bytes=capsule_scalar_bytes,
        metadata_bytes_per_block=metadata_bytes_per_block,
        service_tokens=service_tokens,
        capsule_elements=capsule_elements,
        capsule_bytes=capsule_bytes,
        capsule_gib=Decimal(capsule_bytes) / Decimal(GIB),
        metadata_blocks=metadata_blocks,
        metadata_bytes=metadata_bytes,
        selected_pages_per_token=selected_pages,
        selected_coefficients_per_token=selected_coefficients,
        actual_cold_fraction=actual_cold_fraction,
        capsule_traffic_bytes=capsule_traffic_bytes,
        capsule_traffic_fraction=capsule_traffic_fraction,
        metadata_traffic_fraction=metadata_traffic_fraction,
        fast_traffic_fraction=fast_traffic_fraction,
        minimum_build_traffic_bytes=minimum_build_traffic_bytes,
        minimum_build_traffic_dense_equivalents=(
            minimum_build_traffic_dense_equivalents
        ),
        amortized_traffic_fraction=amortized_traffic_fraction,
        center_operations=center_operations,
        selector_bound_operations=selector_bound_operations,
        fast_operations=fast_operations,
        fast_operation_fraction=fast_operation_fraction,
        minimum_build_operations=minimum_build_operations,
        minimum_build_operation_dense_equivalents=(
            minimum_build_operation_dense_equivalents
        ),
        amortized_operation_fraction=amortized_operation_fraction,
        minimum_traffic_coverage=minimum_traffic,
        minimum_operation_coverage=minimum_operations,
        unallocated_hot_gib=Decimal(unallocated_hot_bytes) / Decimal(GIB),
    )


def prefix_basis(prefix_inputs: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    """Build an orthonormal basis from committed prefix vectors only."""

    source = np.asarray(prefix_inputs, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("prefix_inputs must be a nonempty 2D array")
    if not np.all(np.isfinite(source)):
        raise ValueError("prefix_inputs must be finite")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    _, singular_values, right_vectors = np.linalg.svd(
        source, full_matrices=False
    )
    keep = singular_values > tolerance
    return np.ascontiguousarray(right_vectors[keep].T)


def compile_prefix_atlas(
    prefix_inputs: np.ndarray,
    prefix_images: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``Q`` and ``WQ`` using committed input/image pairs only.

    Prefix vectors and images are rows of ``X`` and ``Y = X W^T``.  For the
    thin SVD ``X = U S V^T``, the causal basis is ``Q = V`` and its image is
    ``WQ = Y^T U S^-1``.  The unchanged weight is deliberately not an input.
    """

    inputs = np.asarray(prefix_inputs, dtype=np.float64)
    images = np.asarray(prefix_images, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
        raise ValueError("prefix_inputs must be a nonempty 2D array")
    if images.ndim != 2 or images.shape[0] != inputs.shape[0]:
        raise ValueError("prefix_images must have one row per prefix input")
    if images.shape[1] == 0:
        raise ValueError("prefix_images must have a nonempty output width")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(images)):
        raise ValueError("prefix input/image pairs must be finite")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        inputs, full_matrices=False
    )
    keep = singular_values > tolerance
    basis = np.ascontiguousarray(right_vectors[keep].T)
    if not np.any(keep):
        return basis, np.empty((images.shape[1], 0), dtype=np.float64)
    scaled_left = left_vectors[:, keep] / singular_values[keep][None, :]
    image = np.ascontiguousarray(images.T @ scaled_left)
    return basis, image


def compile_block_frobenius_bounds(
    weight: np.ndarray, *, block_columns: int
) -> np.ndarray:
    """Compile safe L2 operator upper bounds for contiguous column pages."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("weight must be a nonempty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be finite")
    if block_columns <= 0:
        raise ValueError("block_columns must be positive")
    return np.asarray(
        [
            np.linalg.norm(matrix[:, start : start + block_columns], ord="fro")
            for start in range(0, matrix.shape[1], block_columns)
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ResidualEvaluation:
    coordinates: np.ndarray
    input_residual: np.ndarray
    center: np.ndarray
    correction: np.ndarray
    estimate: np.ndarray
    exact: np.ndarray
    exact_unread_error_norm: float
    certified_unread_radius: float
    revealed_blocks: tuple[int, ...]


def evaluate_residual_atlas(
    weight: np.ndarray,
    basis: np.ndarray,
    image: np.ndarray,
    x: np.ndarray,
    *,
    block_columns: int,
    revealed_blocks: Iterable[int] = (),
    block_bounds: Sequence[float] | np.ndarray | None = None,
) -> ResidualEvaluation:
    """Evaluate a synthetic Atlas center and a fail-closed unread radius."""

    matrix = np.asarray(weight, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    z = np.asarray(image, dtype=np.float64)
    vector = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2 or vector.shape != (matrix.shape[1],):
        raise ValueError("weight/x shape mismatch")
    if q.ndim != 2 or q.shape[0] != matrix.shape[1]:
        raise ValueError("basis shape mismatch")
    if z.shape != (matrix.shape[0], q.shape[1]):
        raise ValueError("image shape mismatch")
    if block_columns <= 0:
        raise ValueError("block_columns must be positive")
    if not all(
        np.all(np.isfinite(value)) for value in (matrix, q, z, vector)
    ):
        raise ValueError("weight, basis, image, and x must be finite")
    if q.shape[1] and not np.allclose(
        q.T @ q, np.eye(q.shape[1]), rtol=0.0, atol=1e-10
    ):
        raise ValueError("basis must be orthonormal")

    bounds = (
        compile_block_frobenius_bounds(matrix, block_columns=block_columns)
        if block_bounds is None
        else np.asarray(block_bounds, dtype=np.float64)
    )
    block_count = math.ceil(matrix.shape[1] / block_columns)
    if bounds.shape != (block_count,) or np.any(bounds < 0):
        raise ValueError("block_bounds shape/value mismatch")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("block_bounds must be finite")
    if not np.allclose(z, matrix @ q, rtol=1e-10, atol=1e-10):
        raise ValueError("image must equal weight @ basis")
    for index, start in enumerate(range(0, matrix.shape[1], block_columns)):
        stop = min(matrix.shape[1], start + block_columns)
        spectral = float(np.linalg.norm(matrix[:, start:stop], ord=2))
        if float(bounds[index]) + 1e-12 < spectral:
            raise ValueError("block bound is not a certified operator bound")

    revealed = tuple(sorted(set(int(index) for index in revealed_blocks)))
    if any(index < 0 or index >= block_count for index in revealed):
        raise ValueError("revealed block index out of range")
    revealed_set = set(revealed)

    coordinates = q.T @ vector
    input_residual = vector - q @ coordinates
    center = z @ coordinates
    correction = np.zeros(matrix.shape[0], dtype=np.float64)
    radius = 0.0
    for index, start in enumerate(range(0, matrix.shape[1], block_columns)):
        stop = min(matrix.shape[1], start + block_columns)
        residual_block = input_residual[start:stop]
        if index in revealed_set:
            correction += matrix[:, start:stop] @ residual_block
        else:
            radius += float(bounds[index]) * float(
                np.linalg.norm(residual_block)
            )

    estimate = center + correction
    exact = matrix @ vector
    error_norm = float(np.linalg.norm(exact - estimate))
    return ResidualEvaluation(
        coordinates=coordinates,
        input_residual=input_residual,
        center=center,
        correction=correction,
        estimate=estimate,
        exact=exact,
        exact_unread_error_norm=error_norm,
        certified_unread_radius=radius,
        revealed_blocks=revealed,
    )


def certify_top1(logit_center: np.ndarray, l2_radius: float) -> int | None:
    """Certify a unique top-1 class from one shared L2 error radius.

    Since every coordinate error is at most the L2 radius, the center's winner
    is invariant when its margin is strictly greater than twice that radius.
    Ties and non-finite inputs fail closed.
    """

    logits = np.asarray(logit_center, dtype=np.float64)
    if logits.ndim != 1 or logits.size < 2 or not np.all(np.isfinite(logits)):
        raise ValueError("logit_center must be a finite vector of size >= 2")
    if not math.isfinite(l2_radius) or l2_radius < 0:
        raise ValueError("l2_radius must be finite and non-negative")
    order = np.argsort(logits, kind="stable")
    winner = int(order[-1])
    runner_up = int(order[-2])
    margin = float(logits[winner] - logits[runner_up])
    return winner if margin > 2.0 * l2_radius else None
