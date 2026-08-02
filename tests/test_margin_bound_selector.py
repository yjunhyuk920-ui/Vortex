import torch

from vortex_runtime.margin_bound_selector import cauchy_margin_bound


def test_cauchy_margin_bound_contains_actual_tile_contribution() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn((7, 5), generator=generator)
    gradient = torch.randn((11, 7), generator=generator)
    residual = torch.randn((11, 5), generator=generator)

    actual = torch.sum(weight * (gradient.T @ residual)).abs().item()
    bound = cauchy_margin_bound(
        weight_energy=float(weight.square().sum().item()),
        gradient_energy=float(gradient.square().sum().item()),
        residual_energy=float(residual.square().sum().item()),
    )

    assert actual <= bound + 1e-5


def test_cauchy_margin_bound_rejects_negative_energy() -> None:
    try:
        cauchy_margin_bound(
            weight_energy=-1.0,
            gradient_energy=1.0,
            residual_energy=1.0,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative energy must be rejected")
