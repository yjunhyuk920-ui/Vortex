from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, int]
    elements: int
    bf16_gib: float
    q4_gib: float
    q5_gib: float
    q6_gib: float


def _gib(elements: int, bits: int) -> float:
    return elements * bits / 8 / 1024**3


def llama_memory_plan(config: dict[str, Any]) -> dict[str, Any]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    vocab = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])

    shapes = {
        "q_proj": (hidden, hidden),
        "k_proj": (kv_dim, hidden),
        "v_proj": (kv_dim, hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
        "embedding": (vocab, hidden),
        "lm_head": (vocab, hidden),
    }
    tensors = []
    for name, shape in shapes.items():
        elements = shape[0] * shape[1]
        tensors.append(TensorPlan(
            name=name,
            shape=shape,
            elements=elements,
            bf16_gib=_gib(elements, 16),
            q4_gib=_gib(elements, 4),
            q5_gib=_gib(elements, 5),
            q6_gib=_gib(elements, 6),
        ))
    layer_elements = sum(t.elements for t in tensors if t.name not in {"embedding", "lm_head"})
    return {
        "layers": layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "largest_tensor": asdict(max(tensors, key=lambda t: t.elements)),
        "per_layer_bf16_gib": _gib(layer_elements, 16),
        "per_layer_q4_gib": _gib(layer_elements, 4),
        "full_model_parameter_estimate_billion": (
            layer_elements * layers + 2 * vocab * hidden
        ) / 1e9,
        "tensors": [asdict(t) for t in tensors],
    }
