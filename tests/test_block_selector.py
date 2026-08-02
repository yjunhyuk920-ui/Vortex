from scripts.run_block_shared_residual_selector import (
    is_better,
    selector_probe_counts,
)


def test_selector_probe_counts_includes_zero_budget_and_boundaries() -> None:
    cumulative = [64 * 1024 * index for index in range(1, 1025)]
    counts = selector_probe_counts(
        candidate_count=1024,
        combined_count=683,
        cumulative_bytes=cumulative,
    )
    assert counts[0] == 0
    assert 128 in counts
    assert 256 in counts
    assert 683 in counts
    assert all(0 <= count <= 1024 for count in counts)


def test_selector_prefers_incremental_prefix_then_fewer_bytes() -> None:
    first = {
        "incremental_committed_tokens": 1,
        "committed_prefix_tokens": 2,
        "selected_weight_bytes": 8 * 1024 * 1024,
    }
    same_gain_more_bytes = {
        "incremental_committed_tokens": 1,
        "committed_prefix_tokens": 2,
        "selected_weight_bytes": 16 * 1024 * 1024,
    }
    larger_gain = {
        "incremental_committed_tokens": 2,
        "committed_prefix_tokens": 3,
        "selected_weight_bytes": 32 * 1024 * 1024,
    }

    assert is_better(first, None)
    assert not is_better(same_gain_more_bytes, first)
    assert is_better(larger_gain, first)
