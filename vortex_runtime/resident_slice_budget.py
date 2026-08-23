from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True)
class LlamaInventory:
    hidden_size: int = 16_384
    intermediate_size: int = 53_248
    vocab_size: int = 128_256
    layers: int = 126
    kv_heads: int = 8
    head_dim: int = 128

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def layer_parameters(self) -> int:
        h = self.hidden_size
        return (
            2 * h * h
            + 2 * h * self.kv_width
            + 3 * h * self.intermediate_size
        )

    @property
    def embedding_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def registered_parameters(self) -> int:
        return (
            self.layers * self.layer_parameters
            + self.embedding_parameters
            + self.lm_head_parameters
        )

    def target_kv_bytes(self, tokens: int, bytes_per_scalar: int = 2) -> int:
        return (
            tokens
            * self.layers
            * 2
            * self.kv_heads
            * self.head_dim
            * bytes_per_scalar
        )

    def draft_kv_bytes(self, selected_layers: int, tokens: int, bytes_per_scalar: int = 2) -> int:
        return (
            tokens
            * selected_layers
            * 2
            * self.kv_heads
            * self.head_dim
            * bytes_per_scalar
        )


@dataclass(frozen=True)
class QuantFormat:
    name: str
    payload_bits: int
    group_size: int | None = None
    scale_bytes: int = 0
    zero_bytes: int = 0

    def stored_bytes(self, parameters: int) -> int:
        payload = (parameters * self.payload_bits + 7) // 8
        if self.group_size is None:
            return payload
        groups = (parameters + self.group_size - 1) // self.group_size
        return payload + groups * (self.scale_bytes + self.zero_bytes)

    @property
    def bytes_per_parameter(self) -> Fraction:
        base = Fraction(self.payload_bits, 8)
        if self.group_size is None:
            return base
        return base + Fraction(self.scale_bytes + self.zero_bytes, self.group_size)


BF16 = QuantFormat("bf16", payload_bits=16)
Q8_G128 = QuantFormat("q8_g128_fp16_scale_zero", payload_bits=8, group_size=128, scale_bytes=2, zero_bytes=2)
Q4_IDEAL = QuantFormat("q4_ideal_no_metadata", payload_bits=4)
Q4_G128 = QuantFormat("q4_g128_fp16_scale_zero", payload_bits=4, group_size=128, scale_bytes=2, zero_bytes=2)


@dataclass(frozen=True)
class RuntimeReserve:
    target_kv_bytes: int
    target_weight_staging_bytes: int
    block_activation_bytes: int
    allocator_fragmentation_bytes: int

    @property
    def total(self) -> int:
        return (
            self.target_kv_bytes
            + self.target_weight_staging_bytes
            + self.block_activation_bytes
            + self.allocator_fragmentation_bytes
        )


def default_reserve(inventory: LlamaInventory, segment_tokens: int) -> RuntimeReserve:
    return RuntimeReserve(
        target_kv_bytes=inventory.target_kv_bytes(segment_tokens),
        target_weight_staging_bytes=512 * MIB,
        block_activation_bytes=256 * MIB,
        allocator_fragmentation_bytes=512 * MIB,
    )


def resident_total_bytes(
    inventory: LlamaInventory,
    quant: QuantFormat,
    selected_layers: int,
    segment_tokens: int,
    reserve: RuntimeReserve,
) -> int:
    return (
        quant.stored_bytes(inventory.lm_head_parameters)
        + selected_layers * quant.stored_bytes(inventory.layer_parameters)
        + inventory.draft_kv_bytes(selected_layers, segment_tokens)
        + reserve.total
    )


def maximum_resident_layers(
    inventory: LlamaInventory,
    quant: QuantFormat,
    segment_tokens: int,
    reserve: RuntimeReserve,
    vram_bytes: int = 8 * GIB,
) -> int:
    layer = 0
    while resident_total_bytes(inventory, quant, layer + 1, segment_tokens, reserve) <= vram_bytes:
        layer += 1
    return layer


def draft_weight_scan_bytes(
    inventory: LlamaInventory,
    quant: QuantFormat,
    selected_layers: int,
) -> int:
    return (
        quant.stored_bytes(inventory.lm_head_parameters)
        + selected_layers * quant.stored_bytes(inventory.layer_parameters)
    )


def maximum_full_layers_under_weight_budget(
    inventory: LlamaInventory,
    quant: QuantFormat,
    weight_budget_bytes: int,
) -> int:
    head = quant.stored_bytes(inventory.lm_head_parameters)
    if head > weight_budget_bytes:
        return -1
    return (weight_budget_bytes - head) // quant.stored_bytes(inventory.layer_parameters)
