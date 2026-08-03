from __future__ import annotations

import math

from vortex_runtime.host_indexed_cell_probe import (
    PointerTableConfig,
    adversarial_table_pair,
    balanced_resident_cache,
    benchmark_packed_pointer_chase,
    build_disjoint_pointer_table,
    cache_frontier_point,
    decode_chain,
    evaluate_adversarial_pair,
    sampled_resident_cache,
    target_cell_probe_projection,
    worst_chain_host_miss_lower_bound,
)


def test_exact_pointer_decode_is_one_serial_probe_per_token() -> None:
    config = PointerTableConfig(chains=4, steps=8)
    table = build_disjoint_pointer_table(config)
    trace = decode_chain(table, chain=2)

    assert trace.logical_probes == 8
    assert trace.serial_dependency_depth == 8
    assert trace.host_misses == 8
    assert trace.resident_hits == 0
    assert trace.addresses == tuple(range(16, 24))
    assert trace.terminal_next_address == -1
    assert trace.tokens == tuple(
        (2 * 7 + step * 11 + 3) % 16
        for step in range(8)
    )


def test_balanced_cache_attains_pigeonhole_lower_bound() -> None:
    config = PointerTableConfig(chains=16, steps=32)
    table = build_disjoint_pointer_table(config)

    for capacity in (0, 1, 15, 16, 160, 256, config.cells):
        cache = balanced_resident_cache(config, capacity)
        point = cache_frontier_point(
            table,
            cache_capacity_records=capacity,
            resident_records=cache,
        )
        expected = worst_chain_host_miss_lower_bound(
            chains=config.chains,
            steps=config.steps,
            cache_capacity_records=capacity,
        )
        assert point.theorem_host_miss_lower_bound == expected
        assert point.measured_worst_chain_misses == expected
        assert point.theorem_pass


def test_sampled_cache_never_beats_worst_chain_lower_bound() -> None:
    config = PointerTableConfig(chains=32, steps=32)
    table = build_disjoint_pointer_table(config)
    capacity = 257
    expected = worst_chain_host_miss_lower_bound(
        chains=config.chains,
        steps=config.steps,
        cache_capacity_records=capacity,
    )

    for seed in range(10):
        cache = sampled_resident_cache(
            config,
            capacity,
            seed=seed,
        )
        point = cache_frontier_point(
            table,
            cache_capacity_records=capacity,
            resident_records=cache,
        )
        assert point.measured_worst_chain_misses >= expected
        assert point.theorem_pass


def test_one_unread_record_can_change_token_and_future_address() -> None:
    config = PointerTableConfig(chains=8, steps=32)
    table = build_disjoint_pointer_table(config)

    for step in (0, 15, 31):
        result = evaluate_adversarial_pair(
            table,
            chain=3,
            step=step,
        )
        assert result.prefix_addresses_identical
        assert result.prefix_tokens_identical
        assert result.current_token_differs
        assert result.next_address_differs
        assert result.only_one_record_differs
        assert result.passes


def test_adversarial_tables_differ_at_exactly_the_addressed_record() -> None:
    config = PointerTableConfig(chains=4, steps=8)
    table = build_disjoint_pointer_table(config)
    base, alternate, address = adversarial_table_pair(
        table,
        chain=1,
        step=4,
    )
    differences = [
        index
        for index in range(config.cells)
        if (
            base.values[index] != alternate.values[index]
            or base.next_addresses[index]
            != alternate.next_addresses[index]
        )
    ]
    assert differences == [address]


def test_packed_host_prototype_executes_one_dependent_probe_per_step() -> None:
    timing = benchmark_packed_pointer_chase(
        cells=4096,
        steps_per_repeat=10_000,
        repeats=2,
        seed=7,
    )
    assert timing.cells == 4096
    assert timing.logical_probes_per_step == 1
    assert timing.record_storage_bits == 64
    assert timing.median_ns_per_probe > 0
    assert timing.minimum_ns_per_probe > 0
    assert not timing.timing_is_target_representative


def test_target_projection_charges_pointer_storage_and_cache() -> None:
    projection = target_cell_probe_projection()

    assert projection.q4_cells == 56_175_137_076
    assert projection.chain_steps == 256
    assert projection.complete_chains == 219_434_129
    assert projection.ignored_tail_cells == 52
    assert projection.address_bits == 36
    assert projection.explicit_record_bits == 40
    assert projection.explicit_record_bytes == 5.0
    assert projection.resident_raw_record_capacity == 1_717_986_918
    assert projection.cached_records_per_chain_floor == 7
    assert projection.worst_chain_host_miss_lower_bound == 249
    assert math.isclose(
        projection.worst_chain_host_miss_fraction,
        249 / 256,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        projection.q4_only_metadata_gib,
        26.158586645498872,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        projection.explicit_pointer_table_gib,
        261.5858664549887,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert projection.logical_host_bytes_per_chain_lower_bound == 1245.0
    assert math.isclose(
        projection.logical_host_bytes_per_token_lower_bound,
        4.86328125,
        rel_tol=0,
        abs_tol=1e-15,
    )
