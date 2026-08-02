from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator

from safetensors import safe_open
import torch


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: Path
    layer_index: int | None


class HuggingFaceLayout:
    """Zero-transformers inspection of a local Hugging Face model directory."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        index_candidates = sorted(self.model_dir.glob("*.safetensors.index.json"))
        if index_candidates:
            index = json.loads(index_candidates[0].read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})
        else:
            shards = sorted(self.model_dir.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError("no safetensors files found")
            weight_map: dict[str, str] = {}
            for shard in shards:
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in weight_map:
                            raise ValueError(f"duplicate tensor key: {key}")
                        weight_map[key] = shard.name
        self.weight_map = weight_map

    def locations(self) -> Iterator[TensorLocation]:
        for name, shard_name in self.weight_map.items():
            match = _LAYER_RE.search(name)
            layer = int(match.group(1)) if match else None
            yield TensorLocation(name, self.model_dir / shard_name, layer)

    def layer_tensor_names(self, layer_index: int) -> list[str]:
        return sorted(
            loc.name for loc in self.locations() if loc.layer_index == layer_index
        )

    @property
    def layer_indices(self) -> list[int]:
        return sorted(
            {loc.layer_index for loc in self.locations() if loc.layer_index is not None}
        )

    def summary(self) -> dict[str, object]:
        shard_names = {loc.shard.name for loc in self.locations()}
        return {
            "model_type": self.config.get("model_type"),
            "architectures": self.config.get("architectures", []),
            "num_hidden_layers": self.config.get("num_hidden_layers"),
            "hidden_size": self.config.get("hidden_size"),
            "intermediate_size": self.config.get("intermediate_size"),
            "num_attention_heads": self.config.get("num_attention_heads"),
            "num_key_value_heads": self.config.get("num_key_value_heads"),
            "vocab_size": self.config.get("vocab_size"),
            "tensor_count": len(self.weight_map),
            "shard_count": len(shard_names),
            "discovered_layers": len(self.layer_indices),
        }


class TensorLocator:
    """Loads individual tensors from safetensors shards without full model load."""

    def __init__(self, layout: HuggingFaceLayout) -> None:
        self.layout = layout

    def load(self, name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        try:
            shard_name = self.layout.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found: {name}") from exc
        shard = self.layout.model_dir / shard_name
        with safe_open(str(shard), framework="pt", device=str(device)) as handle:
            return handle.get_tensor(name)

    def load_layer(
        self,
        layer_index: int,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.load(name, device=device)
            for name in self.layout.layer_tensor_names(layer_index)
        }

    def shape(self, name: str) -> tuple[int, ...]:
        try:
            shard_name = self.layout.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found: {name}") from exc
        shard = self.layout.model_dir / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(name).get_shape())

    def load_slice(self, name: str, index) -> torch.Tensor:
        try:
            shard_name = self.layout.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found: {name}") from exc
        shard = self.layout.model_dir / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[index]
