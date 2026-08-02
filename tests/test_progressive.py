import torch

from vortex_runtime.progressive import ProgressiveLinear


def test_certified_argmax_matches_dense() -> None:
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn(1024, 256, generator=generator)
    op = ProgressiveLinear(weight, base_bits=4, tile_cols=32)
    for _ in range(30):
        x = torch.randn(256, generator=generator)
        result = op.certify_argmax(x)
        assert result.certified
        assert result.token_id == int((weight @ x).argmax().item())


def test_approximate_matvec_bound_is_sound() -> None:
    generator = torch.Generator().manual_seed(8)
    weight = torch.randn(128, 64, generator=generator)
    x = torch.randn(64, generator=generator)
    op = ProgressiveLinear(weight, base_bits=3, tile_cols=16)
    center, bound, _ = op.approximate_matvec(x, absolute_error=0.25)
    exact = weight @ x
    assert torch.all((exact - center).abs() <= bound + 1e-5)
    assert torch.all(bound <= 0.25 + 1e-5)
