from pathlib import Path

import torch

from vortex_runtime.atlas_linear import OnlineAtlasLinear


def test_atlas_learns_low_rank_trace_and_stops_reading_weight() -> None:
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn(96, 64, generator=generator)
    raw_basis = torch.randn(64, 8, generator=generator)
    basis = torch.linalg.qr(raw_basis).Q[:, :8]
    loader_calls = 0

    def load_weight() -> torch.Tensor:
        nonlocal loader_calls
        loader_calls += 1
        return weight

    atlas = OnlineAtlasLinear(
        in_features=64,
        out_features=96,
        weight_loader=load_weight,
        max_rank=16,
        atol=1e-8,
        rtol=1e-6,
    )

    for _ in range(64):
        coefficients = torch.randn(8, generator=generator)
        x = basis @ coefficients
        actual = atlas(x)
        expected = weight @ x
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    assert atlas.rank <= 8
    assert loader_calls <= 8
    assert atlas.stats.fast_vectors >= 56
    assert atlas.stats.fast_fraction >= 0.85


def test_atlas_persistence_preserves_fast_path(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(32, 24, generator=generator)
    samples = torch.randn(6, 24, generator=generator)

    atlas = OnlineAtlasLinear(
        in_features=24,
        out_features=32,
        weight_loader=lambda: weight,
        max_rank=12,
    )
    expected = weight @ samples.T
    for row in samples:
        atlas(row)
    atlas.save(tmp_path / "atlas")

    loader_calls = 0

    def forbidden_loader() -> torch.Tensor:
        nonlocal loader_calls
        loader_calls += 1
        return weight

    restored = OnlineAtlasLinear(
        in_features=24,
        out_features=32,
        weight_loader=forbidden_loader,
        max_rank=12,
    )
    restored.load(tmp_path / "atlas")
    actual = torch.stack([restored(row) for row in samples], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert loader_calls == 0
    assert restored.stats.fast_vectors == samples.shape[0]
