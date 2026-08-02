from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics
import tempfile
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.atlas_linear import OnlineAtlasLinear
from vortex_runtime.hf_loader import HuggingFaceLayout, TensorLocator
from vortex_runtime.llama import StreamingLlama
from vortex_runtime.planner import llama_memory_plan
from vortex_runtime.progressive import ProgressiveLinear
from vortex_runtime.toy_model import create_tiny_llama
from vortex_runtime.vtx_linear import DiskProgressiveLinear, transcode_hf_linear


def main() -> None:
    started = time.time()
    report: dict[str, object] = {}

    gen = torch.Generator().manual_seed(11)
    weight = torch.randn(4096, 1024, generator=gen)
    progressive = {}
    for bits in (4, 5, 6):
        op = ProgressiveLinear(weight, base_bits=bits, tile_cols=128)
        rows = []
        for _ in range(16):
            x = torch.randn(1024, generator=gen)
            result = op.certify_argmax(x)
            assert result.token_id == int((weight @ x).argmax())
            rows.append(result)
        progressive[str(bits)] = {
            "exact_rate": 1.0,
            "coarse_match_rate": statistics.mean(
                r.coarse_token_id == r.exact_token_id for r in rows
            ),
            "mean_residual_fraction": statistics.mean(
                r.residual_fraction_read for r in rows
            ),
            "p95_residual_fraction": sorted(
                r.residual_fraction_read for r in rows
            )[14],
        }
    report["progressive_lm_head"] = progressive

    root = Path(tempfile.mkdtemp())
    model_dir = create_tiny_llama(root / "model", seed=44)
    locator = TensorLocator(HuggingFaceLayout(model_dir))
    exact_head = locator.load("lm_head.weight")
    disk_rows = []
    vtx_dir = transcode_hf_linear(
        locator,
        "lm_head.weight",
        root / "head.vtx",
        base_bits=6,
        tile_cols=16,
        row_block=64,
    )
    disk_op = DiskProgressiveLinear(vtx_dir)
    for seed in range(32):
        x = torch.randn(64, generator=torch.Generator().manual_seed(100 + seed))
        result = disk_op.certify_argmax(x)
        assert result.token_id == int((exact_head @ x).argmax())
        disk_rows.append(result)
    report["disk_progressive_lm_head"] = {
        "exact_rate": 1.0,
        "mean_residual_fraction": statistics.mean(
            r.residual_fraction_read for r in disk_rows
        ),
        "coarse_match_rate": statistics.mean(
            r.coarse_token_id == r.token_id for r in disk_rows
        ),
        "manifest": disk_op.manifest,
    }

    atlas_generator = torch.Generator().manual_seed(404)
    atlas_weight = torch.randn(96, 64, generator=atlas_generator)
    trace_basis = torch.linalg.qr(
        torch.randn(64, 8, generator=atlas_generator)
    ).Q[:, :8]
    atlas_loader_calls = 0

    def load_atlas_weight() -> torch.Tensor:
        nonlocal atlas_loader_calls
        atlas_loader_calls += 1
        return atlas_weight

    atlas = OnlineAtlasLinear(
        in_features=64,
        out_features=96,
        weight_loader=load_atlas_weight,
        max_rank=16,
    )
    for _ in range(128):
        x = trace_basis @ torch.randn(8, generator=atlas_generator)
        actual = atlas(x)
        torch.testing.assert_close(
            actual, atlas_weight @ x, atol=2e-5, rtol=2e-5
        )
    report["online_atlas_low_rank"] = {
        "exact_allclose": True,
        "rank": atlas.rank,
        "cold_weight_reads": atlas_loader_calls,
        "fast_fraction": atlas.stats.fast_fraction,
        "weight_bytes_read": atlas.stats.weight_bytes_read,
        "capsule_bytes": atlas.capsule_bytes,
    }

    atlas_checkpoint = create_tiny_llama(root / "atlas-replay", seed=123)
    atlas_prompt = torch.tensor([[2, 7, 11, 13]], dtype=torch.long)
    atlas_suffixes = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    atlas_builder = StreamingLlama(
        atlas_checkpoint,
        tensor_budget_bytes=2 * 1024 * 1024,
        atlas_suffixes=atlas_suffixes,
        atlas_max_rank=64,
    )
    atlas_expected = atlas_builder.generate(atlas_prompt, max_new_tokens=12)
    builder_report = atlas_builder.atlas_report()
    atlas_store = atlas_builder.save_atlas(root / "persistent-atlas")
    atlas_replay = StreamingLlama(
        atlas_checkpoint,
        tensor_budget_bytes=2 * 1024 * 1024,
        atlas_suffixes=atlas_suffixes,
        atlas_max_rank=64,
    )
    atlas_replay.load_atlas(atlas_store)
    atlas_actual = atlas_replay.generate(atlas_prompt, max_new_tokens=12)
    replay_report = atlas_replay.atlas_report()
    assert atlas_actual == atlas_expected
    assert replay_report["cold_weight_reads"] == 0
    report["internal_atlas_replay"] = {
        "exact_token_match": True,
        "managed_suffixes": list(atlas_suffixes),
        "build_cold_weight_reads": builder_report["cold_weight_reads"],
        "replay_cold_weight_reads": replay_report["cold_weight_reads"],
        "replay_fast_vectors": replay_report["fast_vectors"],
        "capsule_bytes": replay_report["capsule_bytes"],
    }

    jacobi_rows = []
    for seed in range(10, 20):
        checkpoint = create_tiny_llama(root / f"jacobi-{seed}", seed=seed)
        model = StreamingLlama(checkpoint, tensor_budget_bytes=2 * 1024 * 1024)
        prompt = torch.randint(
            1,
            model.config.vocab_size,
            (1, 8),
            generator=torch.Generator().manual_seed(seed * 31),
        )
        sequential = model.generate(prompt, 32)
        jacobi, stats = model.jacobi_generate(
            prompt,
            32,
            block_size=16,
            max_iterations=8,
            fill_token=seed % model.config.vocab_size,
        )
        assert jacobi == sequential
        jacobi_rows.append(stats)
    report["jacobi"] = {
        "exact_match_rate": 1.0,
        "mean_target_passes_per_token": statistics.mean(
            r.target_passes_per_token for r in jacobi_rows
        ),
        "mean_max_committed_block": statistics.mean(
            r.max_committed_block for r in jacobi_rows
        ),
        "mean_committed_block": statistics.mean(
            r.mean_committed_block for r in jacobi_rows
        ),
    }

    report["llama_3_1_405b_plan"] = llama_memory_plan({
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "num_hidden_layers": 126,
    })
    report["elapsed_seconds"] = time.time() - started

    output = Path(__file__).resolve().parents[1] / "validation_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
