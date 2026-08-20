from __future__ import annotations

import numpy as np
import pytest
import torch

from vortex_runtime.residual_margin_certificate import (
    MarginCertificateInvariantError,
    ResidualMarginCompiler,
    array_sha256,
    model_free_controls,
    project_target_resources,
    stable_argmax,
    stable_descending_indices,
    summarize_population,
)


def compiler(bank: torch.Tensor, *, topk: int = 1) -> ResidualMarginCompiler:
    return ResidualMarginCompiler(
        bank,
        required_unique_margin=2.0**-20,
        initial_topk_per_score_source=topk,
        violator_batch_size=1,
        maximum_active_iterations=4,
        full_constraint_fallback=True,
        primal_feasibility_tolerance=1e-9,
        dual_feasibility_tolerance=1e-9,
    )


def test_model_free_controls_pass() -> None:
    controls = model_free_controls()
    assert controls["passed"] is True
    assert controls["positive_certified"] is True
    assert controls["negative_infeasible"] is True


def test_stable_order_uses_lowest_id_on_tie() -> None:
    scores = np.asarray([4.0, 4.0, 3.0, 4.0], dtype=np.float64)
    assert stable_argmax(scores) == 0
    assert stable_descending_indices(scores).tolist()[:3] == [0, 1, 3]


def test_positive_certificate_excludes_complete_vocabulary() -> None:
    bank = torch.tensor(
        [[0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    coarse = torch.zeros(4, dtype=torch.float64)
    result = compiler(bank).solve(
        coarse,
        1,
        initial_score_sources=(coarse, coarse + bank[0]),
    )
    assert result.status == "certified"
    assert result.certificate is not None
    assert result.certificate.stable_winner == 1
    assert result.certificate.minimum_certified_margin > 0.0
    assert result.certificate.coefficient_support == 1
    assert result.active_constraint_count < bank.shape[1]


def test_explicit_infeasible_subset_is_recorded() -> None:
    bank = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)
    coarse = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)
    result = compiler(bank).solve(
        coarse,
        1,
        initial_score_sources=(coarse,),
    )
    assert result.status == "infeasible"
    assert result.solver_status == 2
    assert result.infeasible_subset_count >= 1
    assert len(result.infeasible_subset_sha256) == 64


def test_active_set_adds_unseen_violator() -> None:
    # The first score source exposes competitor 0, while the coefficient needed
    # to beat it promotes competitor 2. The complete scan must add competitor 2.
    bank = torch.tensor(
        [[0.0, 1.0, 3.0], [0.0, 0.0, -3.0]], dtype=torch.float64
    )
    coarse = torch.tensor([1.0, 0.0, -10.0], dtype=torch.float64)
    result = compiler(bank).solve(
        coarse,
        1,
        initial_score_sources=(coarse,),
    )
    assert result.status in {"certified", "infeasible"}
    assert result.active_iterations >= 1
    if result.status == "certified":
        assert result.certificate is not None
        assert result.certificate.stable_winner == 1


def test_population_summary_and_resource_equation() -> None:
    bank = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    positive = compiler(bank).solve(
        torch.zeros(2, dtype=torch.float64),
        1,
        initial_score_sources=(torch.zeros(2),),
    )
    negative = compiler(torch.tensor([[1.0, 1.0]], dtype=torch.float64)).solve(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        1,
        initial_score_sources=(torch.tensor([1.0, 0.0]),),
    )
    summary = summarize_population([positive, negative])
    assert summary.total == 2
    assert summary.certified == 1
    assert summary.infeasible == 1
    assert summary.certificate_fraction == pytest.approx(0.5)

    resources = project_target_resources(
        inherited_hot_bytes=1000,
        block_length=128,
        vocabulary_size=256,
    )
    assert resources["float64_coefficient_block_bytes"] == 128 * 128 * 8
    assert resources["residual_mac_count_per_token"] == 128 * 256
    assert resources["total_hot_bytes"] > 1000


def test_digest_and_nonfinite_inputs_fail_closed() -> None:
    left = np.asarray([1.0, 2.0], dtype=np.float64)
    right = left.copy()
    right.view(np.uint64)[0] ^= np.uint64(1)
    assert array_sha256(left) != array_sha256(right)
    with pytest.raises(MarginCertificateInvariantError):
        compiler(torch.tensor([[float("nan"), 0.0]], dtype=torch.float64))
