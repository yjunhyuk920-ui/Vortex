from pathlib import Path

import torch

from vortex_runtime.hf_loader import HuggingFaceLayout
from vortex_runtime.llama import StreamingLlama
from vortex_runtime.toy_model import create_tiny_llama


def test_layout_and_streamed_generation(tmp_path: Path) -> None:
    model_dir = create_tiny_llama(tmp_path / "model")
    layout = HuggingFaceLayout(model_dir)
    assert layout.summary()["discovered_layers"] == 4
    model = StreamingLlama(model_dir, tensor_budget_bytes=2 * 1024 * 1024)
    output = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
    assert len(output) == 3
    assert all(0 <= token < model.config.vocab_size for token in output)
    assert model.cache.stats.peak_bytes <= model.cache.stats.budget_bytes


def test_jacobi_matches_sequential_greedy(tmp_path: Path) -> None:
    model_dir = create_tiny_llama(tmp_path / "jacobi-model", seed=19)
    model = StreamingLlama(model_dir, tensor_budget_bytes=2 * 1024 * 1024)
    prompt = torch.tensor([[2, 7, 11, 13]], dtype=torch.long)
    sequential = model.generate(prompt, max_new_tokens=12)
    jacobi, stats = model.jacobi_generate(
        prompt,
        max_new_tokens=12,
        block_size=8,
        max_iterations=6,
    )
    assert jacobi == sequential
    assert stats.generated_tokens == 12


def test_streaming_budget_is_respected_with_eviction(tmp_path: Path) -> None:
    model_dir = create_tiny_llama(tmp_path / "evict-model", seed=29)
    model = StreamingLlama(model_dir, tensor_budget_bytes=70 * 1024)
    output = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
    assert len(output) == 2
    assert model.cache.stats.peak_bytes <= model.cache.stats.budget_bytes
    assert model.cache.stats.evictions > 0


def test_persistent_internal_atlas_replays_without_cold_weight_reads(
    tmp_path: Path,
) -> None:
    model_dir = create_tiny_llama(tmp_path / "atlas-model", seed=123)
    prompt = torch.tensor([[2, 7, 11, 13]], dtype=torch.long)
    suffixes = ("self_attn.o_proj.weight", "mlp.down_proj.weight")

    builder = StreamingLlama(
        model_dir,
        tensor_budget_bytes=2 * 1024 * 1024,
        atlas_suffixes=suffixes,
        atlas_max_rank=64,
    )
    expected = builder.generate(prompt, max_new_tokens=12)
    first_report = builder.atlas_report()
    assert first_report["cold_weight_reads"] > 0
    atlas_dir = builder.save_atlas(tmp_path / "atlas")

    replay = StreamingLlama(
        model_dir,
        tensor_budget_bytes=2 * 1024 * 1024,
        atlas_suffixes=suffixes,
        atlas_max_rank=64,
    )
    replay.load_atlas(atlas_dir)
    actual = replay.generate(prompt, max_new_tokens=12)
    replay_report = replay.atlas_report()

    assert actual == expected
    assert replay_report["cold_weight_reads"] == 0
    assert replay_report["cold_vectors"] == 0
    assert replay_report["fast_vectors"] > 0
