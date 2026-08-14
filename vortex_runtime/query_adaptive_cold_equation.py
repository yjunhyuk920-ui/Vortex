"""E0 accounting for a query-adaptive, cold-backed dense executor.

This module is a deterministic research calculator, not an executor and not a
runtime architecture admission.  Fractions are normalized to one complete
non-embedding dense Q4 token evaluation on the registered 405B geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, getcontext


getcontext().prec = 50

GIB = 1024**3

REGISTERED_PARAMETER_COUNT = 405_849_243_648
REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES = 883
REGISTERED_LAYER_COUNT = 126

REGISTERED_TOTAL_Q4_BYTES = REGISTERED_PARAMETER_COUNT // 2
REGISTERED_NON_EMBEDDING_Q4_BYTES = (
    REGISTERED_NON_EMBEDDING_COEFFICIENTS // 2
)

# 1.2 * 4 / 405 and 1.5 * 4 / 405, reduced exactly.
P50_TARGET_FRACTION = Decimal(8) / Decimal(675)
P95_TARGET_FRACTION = Decimal(2) / Decimal(135)

# E0_DECISION_AND_PROOF_TRACE_TRIAGE.md, six independent checks.
KNOWN_SIX_CHECK_VERIFIER_TRAFFIC_FRACTION = Decimal("0.00266838924")


def _decimal(value: Decimal | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(value)


@dataclass(frozen=True)
class ResourceTerms:
    """One operation or traffic equation, expressed in dense equivalents.

    ``common`` is paid on every query (selector, common probes, verification,
    or other mandatory work). ``hit`` is paid only on a successful fast path.
    ``miss`` is wasted work paid only before a miss falls back. A fallback then
    costs one unchanged dense equivalent. ``compile_dense_equivalents`` is the
    one-time build cost amortized across ``service_tokens`` logical tokens.
    """

    common: Decimal = Decimal(0)
    hit: Decimal = Decimal(0)
    miss: Decimal = Decimal(0)
    compile_dense_equivalents: Decimal = Decimal(0)
    service_tokens: int = 1

    def __post_init__(self) -> None:
        for name in ("common", "hit", "miss", "compile_dense_equivalents"):
            value = _decimal(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.service_tokens <= 0:
            raise ValueError("service_tokens must be positive")

    @property
    def compile_amortized(self) -> Decimal:
        return self.compile_dense_equivalents / Decimal(self.service_tokens)


def charged_fraction(coverage: Decimal | str, terms: ResourceTerms) -> Decimal:
    """Return the fully charged mean resource fraction.

    R = common + compile/N + rho*hit + (1-rho)*(miss + 1)
    """

    rho = _decimal(coverage)
    if not Decimal(0) <= rho <= Decimal(1):
        raise ValueError("coverage must be in [0, 1]")
    return (
        terms.common
        + terms.compile_amortized
        + rho * terms.hit
        + (Decimal(1) - rho) * (terms.miss + Decimal(1))
    )


def minimum_coverage(
    budget: Decimal | str,
    terms: ResourceTerms,
) -> Decimal | None:
    """Solve the exact coverage frontier, or return ``None`` if infeasible.

    A fast path must be cheaper than the miss-plus-dense branch for increasing
    coverage to help. Values below zero are clamped to zero; values above one
    mean that even perfect coverage cannot satisfy the budget.
    """

    cap = _decimal(budget)
    if cap < 0:
        raise ValueError("budget must be non-negative")

    miss_branch = terms.miss + Decimal(1)
    fixed = terms.common + terms.compile_amortized
    if terms.hit >= miss_branch:
        return Decimal(0) if fixed + miss_branch <= cap else None

    required = (fixed + miss_branch - cap) / (miss_branch - terms.hit)
    if required <= 0:
        return Decimal(0)
    if required > 1:
        return None
    return required


def maximum_pages(
    budget: Decimal | str,
    page_bytes: int,
    *,
    other_traffic_fraction: Decimal | str = Decimal(0),
    per_page_overhead_bytes: int = 0,
) -> int:
    """Largest page count fitting the non-embedding Q4 traffic budget."""

    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if per_page_overhead_bytes < 0:
        raise ValueError("per_page_overhead_bytes must be non-negative")
    available_fraction = _decimal(budget) - _decimal(other_traffic_fraction)
    if available_fraction <= 0:
        return 0
    available_bytes = available_fraction * Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )
    charged_page_bytes = Decimal(page_bytes + per_page_overhead_bytes)
    return int(available_bytes // charged_page_bytes)


def maximum_payload_bytes_per_call(
    budget: Decimal | str,
    calls: int,
    *,
    other_traffic_fraction: Decimal | str = Decimal(0),
    per_call_overhead_bytes: int = 0,
) -> int:
    """Maximum equal payload size when ``calls`` cold calls are mandatory."""

    if calls <= 0:
        raise ValueError("calls must be positive")
    if per_call_overhead_bytes < 0:
        raise ValueError("per_call_overhead_bytes must be non-negative")
    available_fraction = _decimal(budget) - _decimal(other_traffic_fraction)
    if available_fraction <= 0:
        return 0
    available_bytes = available_fraction * Decimal(
        REGISTERED_NON_EMBEDDING_Q4_BYTES
    )
    per_call = int(available_bytes // Decimal(calls)) - per_call_overhead_bytes
    return max(0, per_call)


def total_page_count(total_bytes: int, page_bytes: int) -> int:
    if total_bytes < 0:
        raise ValueError("total_bytes must be non-negative")
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    return (total_bytes + page_bytes - 1) // page_bytes


def resident_metadata_bytes(
    total_bytes: int,
    page_bytes: int,
    metadata_bytes_per_page: int,
) -> int:
    if metadata_bytes_per_page < 0:
        raise ValueError("metadata_bytes_per_page must be non-negative")
    return total_page_count(total_bytes, page_bytes) * metadata_bytes_per_page


def minimum_service_tokens(
    compile_dense_equivalents: Decimal | int | str,
    per_token_headroom: Decimal | str,
) -> int:
    """Minimum service life needed to amortize a one-time build cost."""

    compile_cost = _decimal(compile_dense_equivalents)
    headroom = _decimal(per_token_headroom)
    if compile_cost < 0:
        raise ValueError("compile_dense_equivalents must be non-negative")
    if headroom <= 0:
        raise ValueError("per_token_headroom must be positive")
    return int(
        (compile_cost / headroom).to_integral_value(rounding=ROUND_CEILING)
    )


def pcie_transfer_floor_seconds(
    traffic_fraction: Decimal | str,
    signaling_ceiling_gib_per_second: Decimal | str,
) -> Decimal:
    """Favorable serialization floor if all charged bytes cross the link."""

    fraction = _decimal(traffic_fraction)
    ceiling = _decimal(signaling_ceiling_gib_per_second)
    if fraction < 0:
        raise ValueError("traffic_fraction must be non-negative")
    if ceiling <= 0:
        raise ValueError("signaling_ceiling_gib_per_second must be positive")
    traffic_gib = (
        fraction
        * Decimal(REGISTERED_NON_EMBEDDING_Q4_BYTES)
        / Decimal(GIB)
    )
    return traffic_gib / ceiling
