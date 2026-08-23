from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


FLOAT32_MIN_SUBNORMAL = float(np.nextafter(np.float32(0.0), np.float32(1.0)))


def float32_to_bf16_bits(values: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Round finite float32 values to BF16 using round-to-nearest-even."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    is_nan = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + bias
    top = (rounded >> np.uint32(16)).astype(np.uint16)
    # Preserve NaNs as quiet BF16 NaNs rather than accidentally rounding to inf.
    if np.any(is_nan):
        top = top.copy()
        top[is_nan] |= np.uint16(0x0040)
    return top


def bf16_bits_to_float32(bits: np.ndarray | Iterable[int] | int) -> np.ndarray:
    array = np.asarray(bits, dtype=np.uint16)
    widened = (array.astype(np.uint32) << np.uint32(16))
    return widened.view(np.float32)


def bf16_round(values: np.ndarray | Iterable[float] | float) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(values))


def bf16_max_exponent_code(bits: np.ndarray) -> int:
    """Return the largest absolute BF16 exponent code in a finite page."""
    array = np.asarray(bits, dtype=np.uint16)
    exponents = ((array >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint16)
    if np.any(exponents == 0xFF):
        raise ValueError("NaN/Inf weight is outside the registered finite ABI")
    return int(exponents.max(initial=0))


def exponent_code_abs_upper_bound(exponent_code: int) -> float:
    """Strict magnitude upper bound for any finite BF16 value with exp <= code."""
    if not 0 <= exponent_code <= 254:
        raise ValueError("BF16 exponent code must be in [0, 254]")
    if exponent_code == 0:
        # All BF16 subnormals are strictly below the smallest normal.
        return math.ldexp(1.0, -126)
    unbiased = exponent_code - 127
    # A normal BF16 value with exponent e is strictly below 2**(e+1).
    return math.ldexp(1.0, unbiased + 1)


def float32_rounding_cell_radius(value: np.float32 | float) -> float:
    """Strict symmetric radius fully inside value's RNE float32 rounding cell."""
    center32 = np.float32(value)
    if not np.isfinite(center32):
        return 0.0
    center = float(center32)
    previous = float(np.nextafter(center32, np.float32(-np.inf), dtype=np.float32))
    following = float(np.nextafter(center32, np.float32(np.inf), dtype=np.float32))
    lower_midpoint = (previous + center) / 2.0
    upper_midpoint = (center + following) / 2.0
    return min(center - lower_midpoint, upper_midpoint - center)


def rounded_product_abs_upper_bound(weight_abs_upper: float, activation_abs_max: float) -> float:
    """Upper-bound abs(fl32(w*x)) for |w|<weight_abs_upper and |x|<=max."""
    exact_bound = weight_abs_upper * float(activation_abs_max)
    if exact_bound == 0.0:
        return 0.0
    if not math.isfinite(exact_bound):
        return math.inf
    rounded = np.float32(exact_bound)
    if not np.isfinite(rounded):
        return math.inf
    return float(np.nextafter(rounded, np.float32(np.inf), dtype=np.float32))


def certify_page_termwise_noop(
    accumulator: np.float32 | float,
    max_weight_exponent_code: int,
    activation_abs_max: float,
) -> bool:
    """Soundly certify every separate-mul/add term in a page leaves acc unchanged.

    The page is skipped only when an upper bound on every rounded BF16-weight x
    BF16-activation product is strictly inside the current float32 accumulator's
    round-to-nearest-even cell. Since each addition is then a no-op, the
    accumulator remains unchanged throughout the skipped page.
    """
    radius = float32_rounding_cell_radius(accumulator)
    if radius <= 0.0:
        return False
    weight_upper = exponent_code_abs_upper_bound(max_weight_exponent_code)
    product_upper = rounded_product_abs_upper_bound(weight_upper, activation_abs_max)
    return product_upper < radius


def native_separate_mul_add(accumulator: np.float32, weight: np.float32, activation: np.float32) -> np.float32:
    product = np.float32(np.float32(weight) * np.float32(activation))
    return np.float32(np.float32(accumulator) + product)


@dataclass(frozen=True)
class TrafficLedger:
    page_size: int
    metadata_bytes_per_page: int
    target_fraction: float
    metadata_fraction: float
    maximum_full_page_read_fraction: float
    required_skip_fraction: float


def traffic_ledger(page_size: int, metadata_bytes_per_page: int, target_fraction: float) -> TrafficLedger:
    if page_size <= 0 or metadata_bytes_per_page <= 0:
        raise ValueError("page and metadata sizes must be positive")
    metadata_fraction = metadata_bytes_per_page / (2.0 * page_size)
    maximum_read = max(0.0, target_fraction - metadata_fraction)
    return TrafficLedger(
        page_size=page_size,
        metadata_bytes_per_page=metadata_bytes_per_page,
        target_fraction=target_fraction,
        metadata_fraction=metadata_fraction,
        maximum_full_page_read_fraction=maximum_read,
        required_skip_fraction=1.0 - maximum_read,
    )


def simulate_row_page_certificate(weights: np.ndarray, activations: np.ndarray, page_size: int) -> dict[str, float | int | bool]:
    """Execute one BF16 row using the exact skip certificate and reference ABI."""
    weights_bf16 = bf16_round(weights).astype(np.float32)
    activations_bf16 = bf16_round(activations).astype(np.float32)
    if weights_bf16.shape != activations_bf16.shape or weights_bf16.ndim != 1:
        raise ValueError("weights and activations must be equal-length vectors")

    reference = np.float32(0.0)
    for weight, activation in zip(weights_bf16, activations_bf16, strict=True):
        reference = native_separate_mul_add(reference, weight, activation)

    candidate = np.float32(0.0)
    pages = 0
    skipped = 0
    terms_executed = 0
    for start in range(0, len(weights_bf16), page_size):
        stop = min(start + page_size, len(weights_bf16))
        weight_page = weights_bf16[start:stop]
        activation_page = activations_bf16[start:stop]
        exponent_code = bf16_max_exponent_code(float32_to_bf16_bits(weight_page))
        activation_abs_max = float(np.max(np.abs(activation_page), initial=np.float32(0.0)))
        pages += 1
        if certify_page_termwise_noop(candidate, exponent_code, activation_abs_max):
            skipped += 1
            continue
        for weight, activation in zip(weight_page, activation_page, strict=True):
            candidate = native_separate_mul_add(candidate, weight, activation)
            terms_executed += 1

    return {
        "exact_match": bool(candidate.view(np.uint32) == reference.view(np.uint32)),
        "page_count": pages,
        "pages_skipped": skipped,
        "pages_read": pages - skipped,
        "page_read_fraction": (pages - skipped) / pages if pages else 0.0,
        "term_execution_fraction": terms_executed / len(weights_bf16) if len(weights_bf16) else 0.0,
        "reference_bits": int(reference.view(np.uint32)),
        "candidate_bits": int(candidate.view(np.uint32)),
    }
