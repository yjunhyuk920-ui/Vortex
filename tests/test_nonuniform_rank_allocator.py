import torch

from vortex_runtime.nonuniform_rank_allocator import (
    ModuleRankProfile,
    allocate_nonuniform_ranks,
    profile_module_rank_value,
    uniform_equivalent_byte_budget,
)


def _profile(name: str, gains: tuple[float, ...]) -> ModuleRankProfile:
    return ModuleRankProfile(
        name=name,
        input_features=8,
        output_features=8,
        numerical_rank=len(gains),
        maximum_rank=len(gains),
        bits=8,
        column_bytes=16.0,
        output_energy=sum(gains),
        marginal_output_energy=gains,
    )


def test_allocator_moves_rank_to_higher_value_module() -> None:
    profiles = {
        "high": _profile("high", (10.0, 9.0, 8.0, 0.1)),
        "low": _profile("low", (1.0, 0.5, 0.25, 0.1)),
    }
    allocation = allocate_nonuniform_ranks(
        profiles,
        byte_budget=64.0,
        minimum_rank=1,
    )

    assert allocation.ranks == {"high": 3, "low": 1}
    assert allocation.used_bytes == 64.0
    assert allocation.estimated_captured_output_fraction > 0.95


def test_allocator_preserves_contiguous_prefix_and_budget() -> None:
    profiles = {
        "a": _profile("a", (5.0, 4.0, 3.0)),
        "b": _profile("b", (6.0, 0.1, 0.1)),
        "c": _profile("c", (2.0, 2.0, 2.0)),
    }
    budget = uniform_equivalent_byte_budget(profiles, rank=2)
    allocation = allocate_nonuniform_ranks(
        profiles,
        byte_budget=budget,
        minimum_rank=1,
    )

    assert allocation.used_bytes <= budget
    assert sum(allocation.ranks.values()) == 6
    assert all(1 <= rank <= profiles[name].maximum_rank for name, rank in allocation.ranks.items())
    assert allocation.ranks["b"] == 1
    assert allocation.ranks["a"] >= 2


def test_prompt_profile_reports_output_energy_by_response_column() -> None:
    inputs = torch.eye(4, dtype=torch.float32)
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    outputs = inputs @ weight.T

    profile = profile_module_rank_value(
        name="linear",
        input_tensor=inputs,
        output_tensor=outputs,
        bias=None,
        maximum_rank=4,
        bits=8,
    )

    assert profile.numerical_rank == 4
    assert profile.maximum_rank == 4
    assert abs(sum(profile.marginal_output_energy) - profile.output_energy) < 1e-5
    assert profile.captured_fraction(4) > 0.999999
    assert profile.column_bytes > 0


def test_minimum_rank_must_fit_budget() -> None:
    profiles = {"only": _profile("only", (1.0, 0.5))}
    try:
        allocate_nonuniform_ranks(
            profiles,
            byte_budget=8.0,
            minimum_rank=1,
        )
    except ValueError as exc:
        assert "cannot fund" in str(exc)
    else:
        raise AssertionError("expected an insufficient-budget error")
