from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, e

import torch
from torch.nn import functional as F

from vortex_runtime.feasibility import GIB, ModelSpec


# For x >= 0, x * sigmoid(x) * (1 - sigmoid(x)) <= x * exp(-x) <= 1/e.
# For x < 0 the same bound follows from sigmoid(x) <= exp(x). Therefore
# |d SiLU(x) / dx| <= 1 + 1/e globally. This is deliberately conservative.
SILU_GLOBAL_LIPSCHITZ = 1.0 + 1.0 / e


@dataclass(frozen=True)
class SignedDualCertificate:
    exact_scalar: float
    approximate_scalar: float
    lower_bound: float
    upper_bound: float
    uncertainty_radius: float
    refined_neurons: int
    total_neurons: int
    selected_fraction: float
    certified_sign: bool
    exact_sign: int
    certified_sign_value: int
    target_absolute_error: float | None
    target_error_met: bool
    interval_contains_exact: bool
    unsafe_certificate: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class SignedDualTrafficBudget:
    selected_fraction: float
    selected_neurons_per_layer: int
    source_bits: int
    exact_refinement_gib_per_token: float
    partial_limit_gib: float
    partial_traffic_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass
class SignedDualTerms:
    exact_contributions: torch.Tensor
    approximate_contributions: torch.Tensor
    lower_contributions: torch.Tensor
    upper_contributions: torch.Tensor
    activation_error_bounds: torch.Tensor
    directional_error_bounds: torch.Tensor

    @property
    def neurons(self) -> int:
        return int(self.exact_contributions.numel())


def _sign(value: float, *, tolerance: float = 0.0) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _symmetric_quantize_rows(
    weight: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    source = weight.detach().to("cpu", torch.float32)
    if bits >= 16:
        return source.contiguous(), torch.zeros(source.shape[0], dtype=torch.float32)
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = (1 << (bits - 1)) - 1
    maximum = source.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    quantized = torch.round(source / scale).clamp(-qmax, qmax)
    restored = quantized * scale
    residual_norms = torch.linalg.vector_norm(source - restored, dim=1)
    return restored.contiguous(), residual_norms.contiguous()


def _symmetric_quantize_columns(
    weight: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    restored_t, residual_norms = _symmetric_quantize_rows(weight.T, bits=bits)
    return restored_t.T.contiguous(), residual_norms


def _product_interval(
    left_center: torch.Tensor,
    left_radius: torch.Tensor,
    right_center: torch.Tensor,
    right_radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_low = left_center - left_radius
    left_high = left_center + left_radius
    right_low = right_center - right_radius
    right_high = right_center + right_radius
    products = torch.stack(
        (
            left_low * right_low,
            left_low * right_high,
            left_high * right_low,
            left_high * right_high,
        ),
        dim=0,
    )
    return products.amin(dim=0), products.amax(dim=0)


def build_signed_dual_terms(
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    activation: torch.Tensor,
    output_dual: torch.Tensor,
    bits: int = 4,
) -> SignedDualTerms:
    """Build a sound local interval for q^T down(SiLU(gate(x))*up(x)).

    ``output_dual`` is a fixed directional vector q at the MLP output. The
    interval is exact for this local operator and fixed input/dual; it does not
    by itself certify later nonlinear layers.
    """

    gate = gate_weight.detach().to("cpu", torch.float32)
    up = up_weight.detach().to("cpu", torch.float32)
    down = down_weight.detach().to("cpu", torch.float32)
    x = activation.detach().to("cpu", torch.float32).reshape(-1)
    q = output_dual.detach().to("cpu", torch.float32).reshape(-1)

    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("gate and up weights must have matching matrix shapes")
    intermediate, hidden = gate.shape
    if down.shape != (hidden, intermediate):
        raise ValueError("down weight must have shape [hidden, intermediate]")
    if x.numel() != hidden or q.numel() != hidden:
        raise ValueError("activation and output_dual must match hidden size")

    gate_hat, gate_residual_norm = _symmetric_quantize_rows(gate, bits=bits)
    up_hat, up_residual_norm = _symmetric_quantize_rows(up, bits=bits)
    down_hat, down_residual_norm = _symmetric_quantize_columns(down, bits=bits)

    x_norm = torch.linalg.vector_norm(x)
    q_norm = torch.linalg.vector_norm(q)

    gate_hat_value = gate_hat @ x
    up_hat_value = up_hat @ x
    gate_radius = gate_residual_norm * x_norm
    up_radius = up_residual_norm * x_norm

    silu_hat = F.silu(gate_hat_value)
    activation_hat = silu_hat * up_hat_value
    activation_radius = (
        SILU_GLOBAL_LIPSCHITZ
        * gate_radius
        * (up_hat_value.abs() + up_radius)
        + silu_hat.abs() * up_radius
    )

    directional_hat = down_hat.T @ q
    directional_radius = down_residual_norm * q_norm

    approximate_contribution = activation_hat * directional_hat
    lower, upper = _product_interval(
        activation_hat,
        activation_radius,
        directional_hat,
        directional_radius,
    )

    exact_activation = F.silu(gate @ x) * (up @ x)
    exact_directional = down.T @ q
    exact_contribution = exact_activation * exact_directional

    return SignedDualTerms(
        exact_contributions=exact_contribution.contiguous(),
        approximate_contributions=approximate_contribution.contiguous(),
        lower_contributions=lower.contiguous(),
        upper_contributions=upper.contiguous(),
        activation_error_bounds=activation_radius.contiguous(),
        directional_error_bounds=directional_radius.contiguous(),
    )


def refine_signed_dual_certificate(
    terms: SignedDualTerms,
    *,
    target_absolute_error: float | None = None,
    require_sign: bool = True,
) -> SignedDualCertificate:
    """Refine the widest neuron intervals until the requested proof closes."""

    if target_absolute_error is not None and target_absolute_error < 0:
        raise ValueError("target_absolute_error must be nonnegative")
    exact = terms.exact_contributions.to(torch.float64)
    approximate = terms.approximate_contributions.to(torch.float64)
    lower_items = terms.lower_contributions.to(torch.float64)
    upper_items = terms.upper_contributions.to(torch.float64)
    if not (
        exact.shape == approximate.shape == lower_items.shape == upper_items.shape
    ):
        raise ValueError("all contribution tensors must have identical shapes")

    exact_scalar = float(exact.sum().item())
    approximate_scalar = float(approximate.sum().item())
    lower = float(lower_items.sum().item())
    upper = float(upper_items.sum().item())
    tolerance = 1e-5 * max(1.0, abs(exact_scalar))
    interval_contains_exact = lower - tolerance <= exact_scalar <= upper + tolerance

    widths = upper_items - lower_items
    order = torch.argsort(widths, descending=True)
    refined = 0

    def proof_state() -> tuple[bool, int, float, bool]:
        certified_value = 1 if lower > 0 else (-1 if upper < 0 else 0)
        certified_sign = certified_value != 0
        uncertainty = max(approximate_scalar - lower, upper - approximate_scalar, 0.0)
        error_met = (
            target_absolute_error is None
            or uncertainty <= target_absolute_error + tolerance
        )
        complete = (not require_sign or certified_sign) and error_met
        return certified_sign, certified_value, uncertainty, complete

    certified_sign, certified_value, uncertainty, complete = proof_state()
    while not complete and refined < exact.numel():
        index = int(order[refined].item())
        exact_value = float(exact[index].item())
        approximate_value = float(approximate[index].item())
        lower_value = float(lower_items[index].item())
        upper_value = float(upper_items[index].item())
        approximate_scalar += exact_value - approximate_value
        lower += exact_value - lower_value
        upper += exact_value - upper_value
        refined += 1
        certified_sign, certified_value, uncertainty, complete = proof_state()

    exact_sign = _sign(exact_scalar, tolerance=tolerance)
    unsafe = bool(
        (certified_sign and certified_value != exact_sign)
        or not interval_contains_exact
    )
    return SignedDualCertificate(
        exact_scalar=exact_scalar,
        approximate_scalar=approximate_scalar,
        lower_bound=lower,
        upper_bound=upper,
        uncertainty_radius=uncertainty,
        refined_neurons=refined,
        total_neurons=int(exact.numel()),
        selected_fraction=refined / max(int(exact.numel()), 1),
        certified_sign=certified_sign,
        exact_sign=exact_sign,
        certified_sign_value=certified_value,
        target_absolute_error=target_absolute_error,
        target_error_met=(
            target_absolute_error is None
            or uncertainty <= target_absolute_error + tolerance
        ),
        interval_contains_exact=interval_contains_exact,
        unsafe_certificate=unsafe,
    )


def signed_dual_refinement_budget(
    *,
    target: ModelSpec,
    selected_fraction: float,
    source_bits: int = 16,
    partial_limit_gib: float = 1.6,
) -> SignedDualTrafficBudget:
    if not 0 <= selected_fraction <= 1:
        raise ValueError("selected_fraction must lie in [0, 1]")
    if source_bits <= 0 or partial_limit_gib <= 0:
        raise ValueError("precision and traffic limit must be positive")
    selected = ceil(target.intermediate_size * selected_fraction)
    bytes_per_neuron = 3 * target.hidden_size * source_bits / 8
    traffic = target.layers * selected * bytes_per_neuron / GIB
    return SignedDualTrafficBudget(
        selected_fraction=selected / target.intermediate_size,
        selected_neurons_per_layer=selected,
        source_bits=source_bits,
        exact_refinement_gib_per_token=traffic,
        partial_limit_gib=partial_limit_gib,
        partial_traffic_pass=traffic <= partial_limit_gib,
    )
