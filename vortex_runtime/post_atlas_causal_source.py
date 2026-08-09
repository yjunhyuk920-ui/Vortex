"""E0 algebra and resource accounting for post-Atlas causal sources.

The rejected Causal Residual Atlas cached primal images ``W @ Q``.  A
direction-aware continuation can additionally cache dual images ``W.T @ P``.
This module makes the resulting exact bilinear decomposition explicit and
keeps its still-unknown cross residual visible.  It is a proof-first reference,
not a runtime implementation or a claim that a dual source is cheap to build.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, getcontext
from typing import Sequence

import numpy as np


getcontext().prec = 50

TARGET_FRACTION = Decimal(8) / Decimal(675)
KNOWN_VERIFIER_TRAFFIC_FRACTION = Decimal("0.00266838924")
ATLAS_REGISTERED_TRAFFIC_FRACTION = Decimal("0.01085025716")

REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
REGISTERED_NON_EMBEDDING_Q4_BYTES = 201_873_948_672
REGISTERED_VOCABULARY = 128_256
REGISTERED_HIDDEN_WIDTH = 16_384
REGISTERED_INTERMEDIATE_WIDTH = 53_248
REGISTERED_LAYER_COUNT = 126


@dataclass(frozen=True)
class BilinearResidualDecomposition:
    """Exact decomposition of one decision functional ``v.T @ W @ x``."""

    exact_value: float
    primal_span_term: float
    dual_span_term: float
    cross_residual_term: float
    reconstructed_value: float
    input_residual_l2: float
    dual_residual_l2: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _orthonormal_basis(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    ambient_dimension: int,
    name: str,
) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != ambient_dimension:
        raise ValueError(f"{name} has an invalid shape")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must be finite")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name} must have orthonormal columns")
    return basis


def decompose_decision_bilinear(
    weight: Sequence[Sequence[float]] | np.ndarray,
    activation: Sequence[float] | np.ndarray,
    decision_direction: Sequence[float] | np.ndarray,
    primal_basis: Sequence[Sequence[float]] | np.ndarray,
    dual_basis: Sequence[Sequence[float]] | np.ndarray,
) -> BilinearResidualDecomposition:
    """Expose the irreducible residual after exact primal and dual images.

    With orthonormal bases ``Q`` and ``P`` and residuals

    ``u = x - Q Q.T x`` and ``r = v - P P.T v``, the exact identity is

    ``v.T W x = v.T (WQ) a + (W.T P b).T u + r.T W u``.

    The first two terms are determined by cached primal/dual images.  The last
    term is not; dropping or merely renaming it would repeat the Atlas error.
    """
    matrix = np.asarray(weight, dtype=np.float64)
    x = np.asarray(activation, dtype=np.float64)
    v = np.asarray(decision_direction, dtype=np.float64)
    if matrix.ndim != 2 or x.ndim != 1 or v.ndim != 1:
        raise ValueError("weight must be 2D and vectors must be 1D")
    if matrix.shape != (v.size, x.size):
        raise ValueError("weight and vector dimensions do not match")
    if not (
        np.all(np.isfinite(matrix))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(v))
    ):
        raise ValueError("bilinear inputs must be finite")

    q = _orthonormal_basis(
        primal_basis, ambient_dimension=x.size, name="primal_basis"
    )
    p = _orthonormal_basis(
        dual_basis, ambient_dimension=v.size, name="dual_basis"
    )
    a = q.T @ x
    u = x - q @ a
    b = p.T @ v
    r = v - p @ b

    primal_image = matrix @ q
    dual_image = matrix.T @ p
    primal_term = float(v @ (primal_image @ a))
    dual_term = float((dual_image @ b) @ u)
    cross_term = float(r @ matrix @ u)
    reconstructed = primal_term + dual_term + cross_term
    exact = float(v @ matrix @ x)
    return BilinearResidualDecomposition(
        exact_value=exact,
        primal_span_term=primal_term,
        dual_span_term=dual_term,
        cross_residual_term=cross_term,
        reconstructed_value=reconstructed,
        input_residual_l2=float(np.linalg.norm(u)),
        dual_residual_l2=float(np.linalg.norm(r)),
    )


def cross_residual_perturbation(
    input_residual: Sequence[float] | np.ndarray,
    dual_residual: Sequence[float] | np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Construct ``Delta W = scale * r u.T`` for the indistinguishability test."""
    u = np.asarray(input_residual, dtype=np.float64)
    r = np.asarray(dual_residual, dtype=np.float64)
    if u.ndim != 1 or r.ndim != 1 or not np.isfinite(scale):
        raise ValueError("residuals must be vectors and scale must be finite")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(r))):
        raise ValueError("residuals must be finite")
    return float(scale) * np.outer(r, u)


def minimum_service_tokens_for_dense_builds(
    dense_equivalent_builds: int,
    *,
    common_fraction: Decimal = Decimal(0),
    target_fraction: Decimal = TARGET_FRACTION,
) -> int | None:
    """Return the first service life that amortizes dense-equivalent builds.

    ``None`` means the common cost already consumes the complete target budget.
    """
    if dense_equivalent_builds < 0:
        raise ValueError("dense_equivalent_builds must be nonnegative")
    if common_fraction < 0 or target_fraction <= 0:
        raise ValueError("fractions must be nonnegative and target positive")
    remainder = target_fraction - common_fraction
    if remainder <= 0:
        return None
    if dense_equivalent_builds == 0:
        return 0
    required = Decimal(dense_equivalent_builds) / remainder
    return int(required.to_integral_value(rounding=ROUND_CEILING))


def derive_post_atlas_resource_audit() -> dict[str, object]:
    """Return deterministic registered-shape E0 accounting."""
    one_layer_composite_coefficients = (
        REGISTERED_VOCABULARY * REGISTERED_INTERMEDIATE_WIDTH
    )
    all_down_composite_coefficients = (
        one_layer_composite_coefficients * REGISTERED_LAYER_COUNT
    )
    # Two bytes is deliberately favorable.  Exact composed values generally
    # need a wider or corrected representation, so this is a lower-cost grant.
    one_layer_composite_bytes = one_layer_composite_coefficients * 2
    all_down_composite_bytes = all_down_composite_coefficients * 2
    one_layer_fraction = Decimal(one_layer_composite_bytes) / Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )
    all_down_fraction = Decimal(all_down_composite_bytes) / Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )

    return {
        "target": {
            "non_embedding_coefficients": REGISTERED_NON_EMBEDDING_COEFFICIENTS,
            "non_embedding_q4_bytes": REGISTERED_NON_EMBEDDING_Q4_BYTES,
            "p50_fraction": str(TARGET_FRACTION),
        },
        "dynamic_dual_build": {
            "registered_service_tokens": 64,
            "one_dense_build_fraction_at_64": str(Decimal(1) / Decimal(64)),
            "minimum_service_tokens_zero_common": {
                str(count): minimum_service_tokens_for_dense_builds(count)
                for count in (1, 2, 4, 8, 16)
            },
            "minimum_service_tokens_with_known_verifier": (
                minimum_service_tokens_for_dense_builds(
                    1, common_fraction=KNOWN_VERIFIER_TRAFFIC_FRACTION
                )
            ),
            "minimum_service_tokens_when_added_to_registered_atlas": (
                minimum_service_tokens_for_dense_builds(
                    1, common_fraction=ATLAS_REGISTERED_TRAFFIC_FRACTION
                )
            ),
        },
        "static_vocabulary_dual_code": {
            "vocabulary": REGISTERED_VOCABULARY,
            "hidden_width": REGISTERED_HIDDEN_WIDTH,
            "intermediate_width": REGISTERED_INTERMEDIATE_WIDTH,
            "layers": REGISTERED_LAYER_COUNT,
            "favorable_scalar_bytes": 2,
            "one_last_down_coefficients": one_layer_composite_coefficients,
            "one_last_down_bytes": one_layer_composite_bytes,
            "one_last_down_gib": str(
                Decimal(one_layer_composite_bytes) / Decimal(1024**3)
            ),
            "one_last_down_full_scan_fraction": str(one_layer_fraction),
            "one_last_down_target_miss_factor": str(
                one_layer_fraction / TARGET_FRACTION
            ),
            "all_down_coefficients": all_down_composite_coefficients,
            "all_down_bytes": all_down_composite_bytes,
            "all_down_tib": str(
                Decimal(all_down_composite_bytes) / Decimal(1024**4)
            ),
            "all_down_full_scan_fraction": str(all_down_fraction),
        },
    }
