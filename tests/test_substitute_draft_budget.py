from __future__ import annotations

from vortex_runtime.feasibility import default_specs
from vortex_runtime.substitute_draft_budget import (
    maximum_retained_layers,
    select_layer_indices,
    substitute_draft_budget,
)


def test_q4_untied_three_layers_fit_but_four_do_not() -> None:
    target, _ = default_specs()
    three = substitute_draft_budget(
        model=target,
        retained_layers=3,
        weight_bits=4,
        tie_word_embeddings=False,
        workspace_gib=1.0,
        memory_limit_gib=8.0,
    )
    four = substitute_draft_budget(
        model=target,
        retained_layers=4,
        weight_bits=4,
        tie_word_embeddings=False,
        workspace_gib=1.0,
        memory_limit_gib=8.0,
    )
    assert three.fits_memory
    assert three.total_gib <= 8.0
    assert not four.fits_memory
    assert four.total_gib > 8.0


def test_maximum_q4_untied_layers_is_three() -> None:
    target, _ = default_specs()
    assert maximum_retained_layers(
        model=target,
        weight_bits=4,
        tie_word_embeddings=False,
        workspace_gib=1.0,
        memory_limit_gib=8.0,
    ) == 3


def test_deterministic_layer_selection_strategies() -> None:
    assert select_layer_indices(
        total_layers=22,
        retained_layers=3,
        strategy="front",
    ) == (0, 1, 2)
    assert select_layer_indices(
        total_layers=22,
        retained_layers=3,
        strategy="uniform",
    ) == (0, 10, 21)
    assert select_layer_indices(
        total_layers=22,
        retained_layers=3,
        strategy="edge",
    ) == (0, 1, 21)
