from pathlib import Path

import torch

from vortex_runtime.hf_loader import HuggingFaceLayout, TensorLocator
from vortex_runtime.toy_model import create_tiny_llama
from vortex_runtime.vtx_linear import transcode_hf_linear, DiskProgressiveLinear


def test_disk_progressive_head_matches_exact(tmp_path: Path) -> None:
    model_dir = create_tiny_llama(tmp_path / "model", seed=33)
    locator = TensorLocator(HuggingFaceLayout(model_dir))
    out = transcode_hf_linear(
        locator,
        "lm_head.weight",
        tmp_path / "head.vtx",
        base_bits=6,
        tile_cols=16,
        row_block=64,
    )
    op = DiskProgressiveLinear(out)
    weight = locator.load("lm_head.weight")
    generator = torch.Generator().manual_seed(4)
    for _ in range(10):
        x = torch.randn(weight.shape[1], generator=generator)
        result = op.certify_argmax(x)
        assert result.certified
        assert result.token_id == int((weight @ x).argmax().item())
        assert result.residual_bytes_read <= op.manifest["residual_bytes"]
