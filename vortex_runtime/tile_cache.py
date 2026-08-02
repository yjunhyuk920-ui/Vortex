from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable

import torch


@dataclass(frozen=True)
class CacheStats:
    budget_bytes: int
    resident_bytes: int
    peak_bytes: int
    hits: int
    misses: int
    evictions: int


class ByteBudgetLRU:
    """Byte-accurate LRU used to enforce a VRAM-like tensor budget."""

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.budget_bytes = budget_bytes
        self._items: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self._bytes = 0
        self._peak = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def get(self, key: Hashable) -> torch.Tensor | None:
        tensor = self._items.get(key)
        if tensor is None:
            self._misses += 1
            return None
        self._items.move_to_end(key)
        self._hits += 1
        return tensor

    def put(self, key: Hashable, tensor: torch.Tensor) -> None:
        size = self.tensor_bytes(tensor)
        if size > self.budget_bytes:
            raise MemoryError(
                f"single tensor requires {size:,} bytes, budget is {self.budget_bytes:,}"
            )
        previous = self._items.pop(key, None)
        if previous is not None:
            self._bytes -= self.tensor_bytes(previous)
        while self._items and self._bytes + size > self.budget_bytes:
            _, evicted = self._items.popitem(last=False)
            self._bytes -= self.tensor_bytes(evicted)
            self._evictions += 1
        self._items[key] = tensor
        self._bytes += size
        self._peak = max(self._peak, self._bytes)

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            budget_bytes=self.budget_bytes,
            resident_bytes=self._bytes,
            peak_bytes=self._peak,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )
