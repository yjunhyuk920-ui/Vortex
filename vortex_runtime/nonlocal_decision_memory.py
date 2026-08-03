from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable

import torch

GIB = 1024**3


@dataclass(frozen=True)
class DecisionMemoryBudget:
    entries: int
    hidden_size: int
    key_rank: int
    block_length: int
    key_bits: int
    token_bits: int
    index_overhead_fraction: float
    keys_gib: float
    blocks_gib: float
    index_gib: float
    total_gib: float
    projection_gflop_per_query: float
    search_gflop_per_query: float
    total_lookup_gflop_per_query: float
    memory_limit_gib: float
    lookup_limit_gflop: float
    memory_pass: bool
    lookup_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass
class NonlocalDecisionMemory:
    center: torch.Tensor
    basis: torch.Tensor
    keys: torch.Tensor
    blocks: torch.Tensor
    lengths: torch.Tensor
    positions: torch.Tensor
    requested_rank: int
    effective_rank: int
    block_length: int

    @property
    def entries(self) -> int:
        return int(self.keys.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.center.numel())


@dataclass(frozen=True)
class PrefixSummary:
    count: int
    first: int
    maximum: int
    mean: float
    p95: float
    coverage_ge_1: float
    coverage_ge_4: float
    coverage_ge_16: float
    coverage_ge_64: float
    coverage_ge_247: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalFrontier:
    requested_rank: int
    effective_rank: int
    nearest: PrefixSummary
    topk_oracles: dict[int, PrefixSummary]
    global_oracle: PrefixSummary
    nearest_positions: list[int]
    budget_actual_entries: DecisionMemoryBudget
    budget_scaled_entries: DecisionMemoryBudget

    def to_dict(self) -> dict:
        return {
            "requested_rank": self.requested_rank,
            "effective_rank": self.effective_rank,
            "nearest": self.nearest.to_dict(),
            "topk_oracles": {
                str(key): value.to_dict() for key, value in self.topk_oracles.items()
            },
            "global_oracle": self.global_oracle.to_dict(),
            "nearest_positions": self.nearest_positions,
            "budget_actual_entries": self.budget_actual_entries.to_dict(),
            "budget_scaled_entries": self.budget_scaled_entries.to_dict(),
        }


def decision_memory_budget(
    *,
    entries: int,
    hidden_size: int,
    key_rank: int,
    block_length: int,
    key_bits: int = 16,
    token_bits: int = 32,
    index_overhead_fraction: float = 0.25,
    memory_limit_gib: float = 2.0,
    lookup_limit_gflop: float = 1.0,
) -> DecisionMemoryBudget:
    if min(entries, hidden_size, key_rank, block_length, key_bits, token_bits) <= 0:
        raise ValueError("budget dimensions and precisions must be positive")
    if index_overhead_fraction < 0:
        raise ValueError("index overhead must be nonnegative")
    key_bytes = entries * key_rank * key_bits / 8
    block_bytes = entries * block_length * token_bits / 8
    index_bytes = index_overhead_fraction * (key_bytes + block_bytes)
    keys_gib = key_bytes / GIB
    blocks_gib = block_bytes / GIB
    index_gib = index_bytes / GIB
    total_gib = keys_gib + blocks_gib + index_gib
    projection_gflop = 2 * hidden_size * key_rank / 1e9
    search_gflop = 2 * entries * key_rank / 1e9
    lookup_gflop = projection_gflop + search_gflop
    return DecisionMemoryBudget(
        entries=entries,
        hidden_size=hidden_size,
        key_rank=key_rank,
        block_length=block_length,
        key_bits=key_bits,
        token_bits=token_bits,
        index_overhead_fraction=index_overhead_fraction,
        keys_gib=keys_gib,
        blocks_gib=blocks_gib,
        index_gib=index_gib,
        total_gib=total_gib,
        projection_gflop_per_query=projection_gflop,
        search_gflop_per_query=search_gflop,
        total_lookup_gflop_per_query=lookup_gflop,
        memory_limit_gib=memory_limit_gib,
        lookup_limit_gflop=lookup_limit_gflop,
        memory_pass=total_gib <= memory_limit_gib,
        lookup_pass=lookup_gflop <= lookup_limit_gflop,
    )


def _normalized_rows(matrix: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    return matrix / torch.clamp(norms, min=1e-12)


def build_nonlocal_decision_memory(
    *,
    prompt_hidden_states: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    key_rank: int,
    block_length: int,
) -> NonlocalDecisionMemory:
    hidden = prompt_hidden_states.detach().to("cpu", torch.float32).contiguous()
    tokens = prompt_token_ids.detach().to("cpu", torch.long).reshape(-1).contiguous()
    if hidden.ndim != 2:
        raise ValueError("prompt_hidden_states must have shape [tokens, hidden]")
    if hidden.shape[0] != tokens.numel():
        raise ValueError("prompt hidden states and tokens must align")
    if tokens.numel() < 2:
        raise ValueError("at least two prompt tokens are required")
    if key_rank <= 0 or block_length <= 0:
        raise ValueError("key rank and block length must be positive")

    # The final prompt position is excluded because its following token belongs
    # to the held-out continuation. Every stored block is prompt-only.
    key_hidden = hidden[:-1]
    center = key_hidden.mean(dim=0)
    centered = key_hidden - center
    effective_rank = min(key_rank, centered.shape[0], centered.shape[1])
    if effective_rank <= 0:
        raise ValueError("memory has no usable key rank")
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:effective_rank].T.contiguous()
    keys = _normalized_rows(centered @ basis).contiguous()

    entries = tokens.numel() - 1
    blocks = torch.full((entries, block_length), -1, dtype=torch.long)
    lengths = torch.zeros(entries, dtype=torch.long)
    for position in range(entries):
        suffix = tokens[position + 1 : position + 1 + block_length]
        blocks[position, : suffix.numel()] = suffix
        lengths[position] = suffix.numel()

    return NonlocalDecisionMemory(
        center=center.contiguous(),
        basis=basis,
        keys=keys,
        blocks=blocks,
        lengths=lengths,
        positions=torch.arange(entries, dtype=torch.long),
        requested_rank=key_rank,
        effective_rank=effective_rank,
        block_length=block_length,
    )


def project_queries(
    memory: NonlocalDecisionMemory,
    query_hidden_states: torch.Tensor,
) -> torch.Tensor:
    queries = query_hidden_states.detach().to("cpu", torch.float32).contiguous()
    if queries.ndim != 2 or queries.shape[1] != memory.hidden_size:
        raise ValueError("query hidden states must match memory hidden size")
    return _normalized_rows((queries - memory.center) @ memory.basis).contiguous()


def prefix_lengths_for_entries(
    memory: NonlocalDecisionMemory,
    *,
    target_suffix: torch.Tensor,
    entry_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target_suffix.detach().to("cpu", torch.long).reshape(-1)
    if entry_indices is None:
        blocks = memory.blocks
        lengths = memory.lengths
    else:
        indices = entry_indices.detach().to("cpu", torch.long).reshape(-1)
        blocks = memory.blocks[indices]
        lengths = memory.lengths[indices]
    width = min(memory.block_length, target.numel())
    if width <= 0:
        return torch.zeros(blocks.shape[0], dtype=torch.long)
    candidate = blocks[:, :width]
    valid = torch.arange(width)[None, :] < lengths[:, None]
    equal = valid & (candidate == target[:width][None, :])
    return equal.to(torch.long).cumprod(dim=1).sum(dim=1)


def _summary(values: Iterable[int] | torch.Tensor) -> PrefixSummary:
    tensor = torch.as_tensor(list(values) if not isinstance(values, torch.Tensor) else values)
    tensor = tensor.to(torch.float64).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("at least one prefix value is required")
    return PrefixSummary(
        count=int(tensor.numel()),
        first=int(tensor[0].item()),
        maximum=int(tensor.max().item()),
        mean=float(tensor.mean().item()),
        p95=float(torch.quantile(tensor, 0.95).item()),
        coverage_ge_1=float((tensor >= 1).to(torch.float64).mean().item()),
        coverage_ge_4=float((tensor >= 4).to(torch.float64).mean().item()),
        coverage_ge_16=float((tensor >= 16).to(torch.float64).mean().item()),
        coverage_ge_64=float((tensor >= 64).to(torch.float64).mean().item()),
        coverage_ge_247=float((tensor >= 247).to(torch.float64).mean().item()),
    )


def evaluate_nonlocal_decision_memory(
    memory: NonlocalDecisionMemory,
    *,
    query_hidden_states: torch.Tensor,
    continuation_token_ids: torch.Tensor,
    topk_values: tuple[int, ...] = (4, 16, 64),
    scaled_entries: int = 65536,
    target_hidden_size: int = 16384,
) -> RetrievalFrontier:
    queries = project_queries(memory, query_hidden_states)
    continuation = continuation_token_ids.detach().to("cpu", torch.long).reshape(-1)
    if queries.shape[0] > continuation.numel():
        raise ValueError("not enough continuation tokens for query positions")
    similarities = queries @ memory.keys.T
    nearest_indices = torch.argmax(similarities, dim=1)
    nearest_prefixes: list[int] = []
    global_prefixes: list[int] = []
    topk_prefixes: dict[int, list[int]] = {value: [] for value in topk_values}

    for step in range(queries.shape[0]):
        target_suffix = continuation[step : step + memory.block_length]
        nearest = prefix_lengths_for_entries(
            memory,
            target_suffix=target_suffix,
            entry_indices=nearest_indices[step : step + 1],
        )
        nearest_prefixes.append(int(nearest[0].item()))

        every_prefix = prefix_lengths_for_entries(
            memory,
            target_suffix=target_suffix,
        )
        global_prefixes.append(int(every_prefix.max().item()))

        ranked = torch.argsort(similarities[step], descending=True)
        for requested_topk in topk_values:
            topk = min(requested_topk, memory.entries)
            selected = ranked[:topk]
            topk_prefixes[requested_topk].append(
                int(every_prefix[selected].max().item())
            )

    actual_budget = decision_memory_budget(
        entries=memory.entries,
        hidden_size=target_hidden_size,
        key_rank=memory.requested_rank,
        block_length=memory.block_length,
    )
    scaled_budget = decision_memory_budget(
        entries=scaled_entries,
        hidden_size=target_hidden_size,
        key_rank=memory.requested_rank,
        block_length=memory.block_length,
    )
    return RetrievalFrontier(
        requested_rank=memory.requested_rank,
        effective_rank=memory.effective_rank,
        nearest=_summary(torch.tensor(nearest_prefixes)),
        topk_oracles={
            value: _summary(torch.tensor(topk_prefixes[value]))
            for value in topk_values
        },
        global_oracle=_summary(torch.tensor(global_prefixes)),
        nearest_positions=memory.positions[nearest_indices].tolist(),
        budget_actual_entries=actual_budget,
        budget_scaled_entries=scaled_budget,
    )
