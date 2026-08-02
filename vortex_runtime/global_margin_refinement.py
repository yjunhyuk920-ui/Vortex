from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.signed_dual_mlp import (
    SignedDualCertificate,
    SignedDualTerms,
    refine_signed_dual_certificate,
)


@dataclass(frozen=True)
class GlobalMarginComparison:
    layers: int
    neurons: int
    target_absolute_error: float
    equal_layer_refined_neurons: int
    equal_layer_refined_fraction: float
    global_refined_neurons: int
    global_refined_fraction: float
    saved_refinements: int
    relative_refinement: float
    global_certificate: SignedDualCertificate

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["global_certificate"] = self.global_certificate.to_dict()
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


def compare_equal_layer_and_global_refinement(
    terms_by_layer: list[SignedDualTerms],
    *,
    total_absolute_error: float,
) -> GlobalMarginComparison:
    """Compare equal layer budgets with one globally optimal width refinement.

    Every neuron has the same exact-read cost. Replacing intervals in descending
    width is therefore the optimal greedy rule for reducing total interval width
    per exact neuron within this fixed-dual linearized certificate.
    """

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
    global_certificate = refine_signed_dual_certificate(
        combined,
        target_absolute_error=total_absolute_error,
        require_sign=False,
    )
    equal_refined = sum(item.refined_neurons for item in local)
    neurons = global_certificate.total_neurons
    global_refined = global_certificate.refined_neurons
    return GlobalMarginComparison(
        layers=layers,
        neurons=neurons,
        target_absolute_error=total_absolute_error,
        equal_layer_refined_neurons=equal_refined,
        equal_layer_refined_fraction=equal_refined / max(neurons, 1),
        global_refined_neurons=global_refined,
        global_refined_fraction=global_refined / max(neurons, 1),
        saved_refinements=equal_refined - global_refined,
        relative_refinement=global_refined / max(equal_refined, 1),
        global_certificate=global_certificate,
    )
