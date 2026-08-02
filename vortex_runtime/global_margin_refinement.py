from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.signed_dual_mlp import (
    SignedDualCertificate,
    SignedDualTerms,
    refine_signed_dual_certificate,
)


@dataclass(frozen=True)
class DualPriceSelection:
    price: float
    refined_neurons: int
    refined_fraction: float
    certificate: SignedDualCertificate

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["certificate"] = self.certificate.to_dict()
        return payload


@dataclass(frozen=True)
class GlobalMarginComparison:
    layers: int
    neurons: int
    target_absolute_error: float
    equal_layer_refined_neurons: int
    equal_layer_refined_fraction: float
    width_global_refined_neurons: int
    width_global_refined_fraction: float
    dual_price_refined_neurons: int
    dual_price_refined_fraction: float
    best_dual_price: float
    saved_vs_equal_layer: int
    saved_vs_width_global: int
    relative_to_equal_layer: float
    dual_price_certificate: SignedDualCertificate

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dual_price_certificate"] = self.dual_price_certificate.to_dict()
        return payload


def concatenate_signed_dual_terms(
    terms_by_layer: list[SignedDualTerms],
) -> SignedDualTerms:
    if not terms_by_layer:
        raise ValueError("at least one layer term set is required")
    return SignedDualTerms(
        exact_contributions=torch.cat(
            [terms.exact_contributions.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
        approximate_contributions=torch.cat(
            [terms.approximate_contributions.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
        lower_contributions=torch.cat(
            [terms.lower_contributions.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
        upper_contributions=torch.cat(
            [terms.upper_contributions.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
        activation_error_bounds=torch.cat(
            [terms.activation_error_bounds.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
        directional_error_bounds=torch.cat(
            [terms.directional_error_bounds.reshape(-1) for terms in terms_by_layer]
        ).contiguous(),
    )


def _certificate_for_indices(
    terms: SignedDualTerms,
    *,
    selected: torch.Tensor,
    target_absolute_error: float,
) -> SignedDualCertificate:
    exact = terms.exact_contributions.to(torch.float64).reshape(-1)
    approximate = terms.approximate_contributions.to(torch.float64).reshape(-1)
    lower = terms.lower_contributions.to(torch.float64).reshape(-1)
    upper = terms.upper_contributions.to(torch.float64).reshape(-1)
    chosen = selected.to(torch.bool).reshape(-1)
    if not (exact.shape == approximate.shape == lower.shape == upper.shape == chosen.shape):
        raise ValueError("term and selection shapes must match")

    corrected_approximate = approximate.clone()
    corrected_lower = lower.clone()
    corrected_upper = upper.clone()
    corrected_approximate[chosen] = exact[chosen]
    corrected_lower[chosen] = exact[chosen]
    corrected_upper[chosen] = exact[chosen]

    exact_scalar = float(exact.sum().item())
    approximate_scalar = float(corrected_approximate.sum().item())
    lower_scalar = float(corrected_lower.sum().item())
    upper_scalar = float(corrected_upper.sum().item())
    uncertainty = max(
        approximate_scalar - lower_scalar,
        upper_scalar - approximate_scalar,
        0.0,
    )
    tolerance = 1e-5 * max(1.0, abs(exact_scalar))
    interval_contains = lower_scalar - tolerance <= exact_scalar <= upper_scalar + tolerance
    certified_sign_value = 1 if lower_scalar > 0 else (-1 if upper_scalar < 0 else 0)
    exact_sign = 1 if exact_scalar > tolerance else (-1 if exact_scalar < -tolerance else 0)
    unsafe = bool(
        not interval_contains
        or (
            certified_sign_value != 0
            and certified_sign_value != exact_sign
        )
    )
    return SignedDualCertificate(
        exact_scalar=exact_scalar,
        approximate_scalar=approximate_scalar,
        lower_bound=lower_scalar,
        upper_bound=upper_scalar,
        uncertainty_radius=uncertainty,
        refined_neurons=int(chosen.sum().item()),
        total_neurons=int(chosen.numel()),
        selected_fraction=float(chosen.float().mean().item()),
        certified_sign=certified_sign_value != 0,
        exact_sign=exact_sign,
        certified_sign_value=certified_sign_value,
        target_absolute_error=target_absolute_error,
        target_error_met=uncertainty <= target_absolute_error + tolerance,
        interval_contains_exact=interval_contains,
        unsafe_certificate=unsafe,
    )


def dual_price_global_refinement(
    terms: SignedDualTerms,
    *,
    target_absolute_error: float,
    price_steps: int = 41,
) -> DualPriceSelection:
    """Solve the two-sided interval cover by a dual-price ordering sweep.

    An unrefined neuron contributes `a_i-L_i` to lower uncertainty and
    `U_i-a_i` to upper uncertainty. Exact refinement removes both. For each
    lambda in [0,1], order by `lambda*lower + (1-lambda)*upper`, take the
    shortest prefix satisfying both global constraints, and retain the smallest
    feasible prefix across prices.
    """

    if target_absolute_error < 0 or price_steps < 2:
        raise ValueError("target must be nonnegative and price_steps at least 2")
    approximate = terms.approximate_contributions.to(torch.float64).reshape(-1)
    lower_uncertainty = torch.clamp(
        approximate - terms.lower_contributions.to(torch.float64).reshape(-1),
        min=0.0,
    )
    upper_uncertainty = torch.clamp(
        terms.upper_contributions.to(torch.float64).reshape(-1) - approximate,
        min=0.0,
    )
    total_lower = lower_uncertainty.sum()
    total_upper = upper_uncertainty.sum()
    neurons = approximate.numel()
    best_count = neurons + 1
    best_price = 0.5
    best_selection = torch.ones(neurons, dtype=torch.bool)

    if total_lower <= target_absolute_error and total_upper <= target_absolute_error:
        best_count = 0
        best_selection.zero_()
    else:
        for price in torch.linspace(0.0, 1.0, steps=price_steps):
            score = price * lower_uncertainty + (1.0 - price) * upper_uncertainty
            order = torch.argsort(score, descending=True)
            removed_lower = torch.cumsum(lower_uncertainty[order], dim=0)
            removed_upper = torch.cumsum(upper_uncertainty[order], dim=0)
            feasible = (
                total_lower - removed_lower <= target_absolute_error
            ) & (
                total_upper - removed_upper <= target_absolute_error
            )
            indices = torch.nonzero(feasible, as_tuple=False)
            if indices.numel() == 0:
                count = neurons
            else:
                count = int(indices[0, 0].item()) + 1
            if count < best_count:
                selection = torch.zeros(neurons, dtype=torch.bool)
                selection[order[:count]] = True
                best_count = count
                best_price = float(price.item())
                best_selection = selection

    certificate = _certificate_for_indices(
        terms,
        selected=best_selection,
        target_absolute_error=target_absolute_error,
    )
    return DualPriceSelection(
        price=best_price,
        refined_neurons=best_count,
        refined_fraction=best_count / max(neurons, 1),
        certificate=certificate,
    )


def compare_equal_layer_and_global_refinement(
    terms_by_layer: list[SignedDualTerms],
    *,
    total_absolute_error: float,
    price_steps: int = 41,
) -> GlobalMarginComparison:
    if total_absolute_error < 0:
        raise ValueError("total_absolute_error must be nonnegative")
    layers = len(terms_by_layer)
    if layers == 0:
        raise ValueError("at least one layer is required")
    per_layer = total_absolute_error / layers
    local = [
        refine_signed_dual_certificate(
            terms,
            target_absolute_error=per_layer,
            require_sign=False,
        )
        for terms in terms_by_layer
    ]
    combined = concatenate_signed_dual_terms(terms_by_layer)
    width_global = refine_signed_dual_certificate(
        combined,
        target_absolute_error=total_absolute_error,
        require_sign=False,
    )
    dual_price = dual_price_global_refinement(
        combined,
        target_absolute_error=total_absolute_error,
        price_steps=price_steps,
    )
    equal_refined = sum(item.refined_neurons for item in local)
    neurons = width_global.total_neurons
    return GlobalMarginComparison(
        layers=layers,
        neurons=neurons,
        target_absolute_error=total_absolute_error,
        equal_layer_refined_neurons=equal_refined,
        equal_layer_refined_fraction=equal_refined / max(neurons, 1),
        width_global_refined_neurons=width_global.refined_neurons,
        width_global_refined_fraction=width_global.selected_fraction,
        dual_price_refined_neurons=dual_price.refined_neurons,
        dual_price_refined_fraction=dual_price.refined_fraction,
        best_dual_price=dual_price.price,
        saved_vs_equal_layer=equal_refined - dual_price.refined_neurons,
        saved_vs_width_global=width_global.refined_neurons - dual_price.refined_neurons,
        relative_to_equal_layer=dual_price.refined_neurons / max(equal_refined, 1),
        dual_price_certificate=dual_price.certificate,
    )
