import torch

from vortex_runtime.tile_cache import ByteBudgetLRU


def test_cache_enforces_budget() -> None:
    cache = ByteBudgetLRU(1024)
    for i in range(10):
        cache.put(i, torch.zeros(128, dtype=torch.float32))
    assert cache.stats.peak_bytes <= 1024
    assert cache.stats.evictions > 0
