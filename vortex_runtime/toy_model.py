from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from .llama import LlamaConfig


def create_tiny_llama(
    output_dir: str | Path,
    *,
    seed: int = 7,
    config: LlamaConfig | None = None,
) -> Path:
    """Create a valid tiny HF-style Llama checkpoint for runtime tests."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = config or LlamaConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int, scale: float = 0.03) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    config_dict = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": False,
    }
    (output / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    weight_map: dict[str, str] = {}
    globals_name = "model-00001-of-00006.safetensors"
    global_tensors = {
        "model.embed_tokens.weight": randn(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.ones(cfg.hidden_size),
        "lm_head.weight": randn(cfg.vocab_size, cfg.hidden_size),
    }
    save_file(global_tensors, output / globals_name)
    for key in global_tensors:
        weight_map[key] = globals_name

    for layer in range(cfg.num_hidden_layers):
        shard_name = f"model-{layer + 2:05d}-of-00006.safetensors"
        prefix = f"model.layers.{layer}"
        tensors = {
            f"{prefix}.input_layernorm.weight": torch.ones(cfg.hidden_size),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(cfg.hidden_size),
            f"{prefix}.self_attn.q_proj.weight": randn(cfg.hidden_size, cfg.hidden_size),
            f"{prefix}.self_attn.k_proj.weight": randn(
                cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
            ),
            f"{prefix}.self_attn.v_proj.weight": randn(
                cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
            ),
            f"{prefix}.self_attn.o_proj.weight": randn(cfg.hidden_size, cfg.hidden_size),
            f"{prefix}.mlp.gate_proj.weight": randn(cfg.intermediate_size, cfg.hidden_size),
            f"{prefix}.mlp.up_proj.weight": randn(cfg.intermediate_size, cfg.hidden_size),
            f"{prefix}.mlp.down_proj.weight": randn(cfg.hidden_size, cfg.intermediate_size),
        }
        save_file(tensors, output / shard_name)
        for key in tensors:
            weight_map[key] = shard_name

    index = {
        "metadata": {"total_size": sum(p.stat().st_size for p in output.glob("*.safetensors"))},
        "weight_map": weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return output
