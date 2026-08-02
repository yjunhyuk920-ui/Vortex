from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .hf_loader import HuggingFaceLayout, TensorLocator
from .progressive import CertificationResult
from .vtx_linear import transcode_hf_linear, DiskProgressiveLinear, DiskCertificationResult
from .tile_cache import ByteBudgetLRU


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden size must be divisible by attention heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LlamaConfig":
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
        )


@dataclass(frozen=True)
class JacobiStats:
    target_passes: int
    generated_tokens: int
    target_passes_per_token: float
    max_committed_block: int
    mean_committed_block: float


@dataclass
class LayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps).to(x.dtype)
    return normalized * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim)
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :].to(q.dtype)
    sin = emb.sin()[None, None, :, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return x
    return x.repeat_interleave(repetitions, dim=1)


class StreamingLlama:
    """Minimal Llama decoder that loads one layer at a time from safetensors."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        tensor_budget_bytes: int = 8 * 1024**3,
        lm_head_base_bits: int = 4,
    ) -> None:
        self.layout = HuggingFaceLayout(model_dir)
        if self.layout.config.get("model_type") != "llama":
            raise ValueError("prototype StreamingLlama currently lowers model_type=llama")
        self.config = LlamaConfig.from_dict(self.layout.config)
        self.locator = TensorLocator(self.layout)
        self.device = torch.device(device)
        self.cache = ByteBudgetLRU(tensor_budget_bytes)
        self.layer_cache = [LayerCache() for _ in range(self.config.num_hidden_layers)]

        self.embedding_name = "model.embed_tokens.weight"
        self.final_norm = self._load_global("model.norm.weight")
        vtx_dir = Path(model_dir) / ".vortex" / f"lm-head-{lm_head_base_bits}bit"
        transcode_hf_linear(
            self.locator,
            "lm_head.weight",
            vtx_dir,
            base_bits=lm_head_base_bits,
            tile_cols=min(128, self.config.hidden_size),
            row_block=512,
        )
        self.lm_head = DiskProgressiveLinear(vtx_dir)

    def _load_global(self, name: str) -> torch.Tensor:
        tensor = self.locator.load(name, device="cpu").to(self.device)
        self.cache.put(("global", name), tensor)
        return tensor

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.detach().to("cpu").reshape(-1)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        rows = []
        for token in unique.tolist():
            row = self.locator.load_slice(
                self.embedding_name, (slice(token, token + 1), slice(None))
            )[0]
            rows.append(row)
        table = torch.stack(rows, dim=0).to(self.device)
        embedded = table[inverse.to(table.device)]
        return embedded.reshape(*input_ids.shape, self.config.hidden_size)

    def reset_cache(self) -> None:
        self.layer_cache = [LayerCache() for _ in range(self.config.num_hidden_layers)]

    def _short(self, layer: int, suffix: str) -> str:
        return f"model.layers.{layer}.{suffix}"

    def _weight(self, layer: int, suffix: str) -> torch.Tensor:
        name = self._short(layer, suffix)
        cached = self.cache.get(name)
        if cached is None:
            cached = self.locator.load(name, device="cpu").to(self.device)
            self.cache.put(name, cached)
        return cached

    def _layer_forward(
        self,
        hidden: torch.Tensor,
        layer_index: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        residual = hidden
        input_norm = self._weight(layer_index, "input_layernorm.weight")
        x = rms_norm(hidden, input_norm, cfg.rms_norm_eps)
        del input_norm

        bsz, seq_len, _ = x.shape
        q_weight = self._weight(layer_index, "self_attn.q_proj.weight")
        q = F.linear(x, q_weight)
        del q_weight
        k_weight = self._weight(layer_index, "self_attn.k_proj.weight")
        k = F.linear(x, k_weight)
        del k_weight
        v_weight = self._weight(layer_index, "self_attn.v_proj.weight")
        v = F.linear(x, v_weight)
        del v_weight

        q = q.view(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, positions, cfg.rope_theta)

        cache = self.layer_cache[layer_index]
        past_len = 0 if cache.key is None else cache.key.shape[2]
        if cache.key is not None:
            k = torch.cat((cache.key, k), dim=2)
            v = torch.cat((cache.value, v), dim=2)
        cache.key = k.detach()
        cache.value = v.detach()

        repetition = cfg.num_attention_heads // cfg.num_key_value_heads
        k_full = repeat_kv(k, repetition)
        v_full = repeat_kv(v, repetition)
        scores = torch.matmul(q, k_full.transpose(-1, -2)) / math.sqrt(cfg.head_dim)

        total_len = past_len + seq_len
        query_positions = torch.arange(past_len, total_len, device=x.device)
        key_positions = torch.arange(total_len, device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal[None, None, :, :], -torch.inf)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(x.dtype)
        attn = torch.matmul(probs, v_full)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, cfg.hidden_size)
        o_weight = self._weight(layer_index, "self_attn.o_proj.weight")
        hidden = residual + F.linear(attn, o_weight)
        del o_weight

        residual = hidden
        post_norm = self._weight(layer_index, "post_attention_layernorm.weight")
        x = rms_norm(hidden, post_norm, cfg.rms_norm_eps)
        del post_norm
        gate_weight = self._weight(layer_index, "mlp.gate_proj.weight")
        gate = F.silu(F.linear(x, gate_weight))
        del gate_weight
        up_weight = self._weight(layer_index, "mlp.up_proj.weight")
        up = F.linear(x, up_weight)
        del up_weight
        down_weight = self._weight(layer_index, "mlp.down_proj.weight")
        hidden = residual + F.linear(gate * up, down_weight)
        del down_weight
        return hidden

    @torch.inference_mode()
    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("prototype supports input_ids shape [1, seq]")
        input_ids = input_ids.to(self.device)
        hidden = self._embed(input_ids)
        past_len = 0
        if self.layer_cache and self.layer_cache[0].key is not None:
            past_len = self.layer_cache[0].key.shape[2]
        positions = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            device=self.device,
        )
        for layer in range(self.config.num_hidden_layers):
            hidden = self._layer_forward(hidden, layer, positions)
        return rms_norm(hidden, self.final_norm, self.config.rms_norm_eps)

    @torch.inference_mode()
    def next_token(self, input_ids: torch.Tensor) -> DiskCertificationResult:
        hidden = self.forward_hidden(input_ids)
        return self.lm_head.certify_argmax(hidden[0, -1].to(torch.float32))

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
        self.reset_cache()
        result: list[int] = []
        current = input_ids
        for _ in range(max_new_tokens):
            certified = self.next_token(current)
            token = certified.token_id
            result.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=self.device)
        return result

    @torch.inference_mode()
    def _dense_logits_fresh(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.reset_cache()
        hidden = self.forward_hidden(input_ids)
        return self.lm_head.exact_matmul(hidden[0])

    @torch.inference_mode()
    def jacobi_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        block_size: int = 16,
        max_iterations: int = 8,
        fill_token: int = 0,
    ) -> tuple[list[int], JacobiStats]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, seq]")
        if block_size <= 0 or max_iterations <= 0:
            raise ValueError("block_size and max_iterations must be positive")

        prefix = input_ids.to(self.device)
        generated: list[int] = []
        passes = 0
        committed_sizes: list[int] = []

        while len(generated) < max_new_tokens:
            width = min(block_size, max_new_tokens - len(generated))
            guesses = torch.full(
                (width,), fill_token, dtype=torch.long, device=self.device
            )
            committed: list[int] = []

            for _ in range(max_iterations):
                sequence = torch.cat((prefix, guesses[None, :]), dim=1)
                logits = self._dense_logits_fresh(sequence)
                passes += 1
                start = prefix.shape[1] - 1
                proposed = logits[start : start + width].argmax(dim=-1)

                stable = proposed.eq(guesses)
                stable_prefix = 0
                for is_stable in stable.tolist():
                    if not is_stable:
                        break
                    stable_prefix += 1

                if stable_prefix > 0:
                    committed = proposed[:stable_prefix].tolist()
                    break
                guesses = proposed

            if not committed:
                committed = [int(proposed[0].item())]

            remaining = max_new_tokens - len(generated)
            committed = committed[:remaining]
            committed_sizes.append(len(committed))
            generated.extend(committed)
            prefix = torch.cat(
                (
                    prefix,
                    torch.tensor(
                        [committed], dtype=torch.long, device=self.device
                    ),
                ),
                dim=1,
            )

        stats = JacobiStats(
            target_passes=passes,
            generated_tokens=len(generated),
            target_passes_per_token=passes / max(1, len(generated)),
            max_committed_block=max(committed_sizes, default=0),
            mean_committed_block=(
                sum(committed_sizes) / len(committed_sizes)
                if committed_sizes
                else 0.0
            ),
        )
        return generated, stats
