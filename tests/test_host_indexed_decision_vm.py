from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from vortex_runtime.host_indexed_cell_probe import (
    ExplicitPointerTable,
    PointerTableConfig,
    build_disjoint_pointer_table,
)
from vortex_runtime.host_indexed_decision_vm import (
    DecisionVMFormatError,
    DecisionVMReader,
    FORMAT_ALIGNED64,
    FORMAT_COMPACT40,
    HEADER_BYTES,
    LRURecordCache,
    benchmark_format,
    build_decision_vm_file,
    target_decision_vm_projection,
)


@pytest.mark.parametrize("flags", [FORMAT_COMPACT40, FORMAT_ALIGNED64])
def test_round_trip_exact_replay_for_both_formats(
    tmp_path: Path,
    flags: int,
) -> None:
    config = PointerTableConfig(chains=8, steps=32)
    table = build_disjoint_pointer_table(config)
    path = tmp_path / f"table-{flags}.vtx"
    report = build_decision_vm_file(table, path, flags=flags)

    assert report.atomic_replace
    assert report.temporary_file_removed
    assert report.file_bytes == path.stat().st_size
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    with DecisionVMReader(path) as reader:
        assert reader.checksum_verified
        assert reader.header.record_count == config.cells
        assert reader.header.start_count == config.chains
        assert reader.header.chain_steps == config.steps
        for chain in range(config.chains):
            result = reader.replay(chain=chain)
            source_start = chain * config.steps
            assert result.tokens == tuple(
                table.values[
                    source_start : source_start + config.steps
                ]
            )
            assert result.addresses == tuple(
                range(source_start, source_start + config.steps)
            )
            assert result.terminal_next_address == -1
            assert result.logical_record_probes == config.steps
            assert result.mmap_record_reads == config.steps


def test_lru_cache_turns_second_replay_into_all_hits(
    tmp_path: Path,
) -> None:
    config = PointerTableConfig(chains=4, steps=64)
    table = build_disjoint_pointer_table(config)
    path = tmp_path / "cache.vtx"
    build_decision_vm_file(table, path, flags=FORMAT_COMPACT40)

    with DecisionVMReader(path) as reader:
        cache = LRURecordCache(config.steps)
        first = reader.replay(chain=2, cache=cache)
        second = reader.replay(chain=2, cache=cache)

    assert first.tokens == second.tokens
    assert first.cache_hits == 0
    assert first.cache_misses == config.steps
    assert first.mmap_record_reads == config.steps
    assert second.cache_hits == config.steps
    assert second.cache_misses == 0
    assert second.mmap_record_reads == 0


def test_failed_rebuild_preserves_existing_valid_destination(
    tmp_path: Path,
) -> None:
    config = PointerTableConfig(chains=2, steps=8)
    valid = build_disjoint_pointer_table(config)
    path = tmp_path / "atomic.vtx"
    build_decision_vm_file(valid, path, flags=FORMAT_COMPACT40)
    original = path.read_bytes()

    invalid_next = list(valid.next_addresses)
    invalid_next[3] = config.cells + 99
    invalid = ExplicitPointerTable(
        config=config,
        values=valid.values,
        next_addresses=tuple(invalid_next),
        starts=valid.starts,
    )
    with pytest.raises(DecisionVMFormatError):
        build_decision_vm_file(
            invalid,
            path,
            flags=FORMAT_COMPACT40,
        )

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
    with DecisionVMReader(path) as reader:
        assert reader.replay(chain=0).tokens == tuple(
            valid.values[: config.steps]
        )


def test_reader_rejects_bad_magic_truncation_and_payload_corruption(
    tmp_path: Path,
) -> None:
    table = build_disjoint_pointer_table(
        PointerTableConfig(chains=2, steps=16)
    )
    source = tmp_path / "source.vtx"
    build_decision_vm_file(table, source, flags=FORMAT_ALIGNED64)

    bad_magic = tmp_path / "bad-magic.vtx"
    shutil.copyfile(source, bad_magic)
    with bad_magic.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"BROKEN!!")
    with pytest.raises(DecisionVMFormatError, match="magic"):
        DecisionVMReader(bad_magic)

    truncated = tmp_path / "truncated.vtx"
    data = source.read_bytes()
    truncated.write_bytes(data[:-1])
    with pytest.raises(DecisionVMFormatError, match="file size"):
        DecisionVMReader(truncated)

    corrupt = tmp_path / "corrupt.vtx"
    shutil.copyfile(source, corrupt)
    with corrupt.open("r+b") as handle:
        handle.seek(HEADER_BYTES + 3)
        current = handle.read(1)
        handle.seek(HEADER_BYTES + 3)
        handle.write(bytes([current[0] ^ 0x20]))
    with pytest.raises(DecisionVMFormatError, match="payload checksum"):
        DecisionVMReader(corrupt)


def test_reader_without_payload_verification_still_validates_pointers(
    tmp_path: Path,
) -> None:
    config = PointerTableConfig(chains=2, steps=16)
    table = build_disjoint_pointer_table(config)
    path = tmp_path / "pointer.vtx"
    build_decision_vm_file(table, path, flags=FORMAT_COMPACT40)

    # Record zero is five bytes. Encode a pointer code larger than record_count.
    invalid_word = ((config.cells + 100 + 1) << 4) | 3
    with path.open("r+b") as handle:
        handle.seek(HEADER_BYTES)
        handle.write(invalid_word.to_bytes(5, "little"))

    with DecisionVMReader(
        path,
        verify_payload_checksum=False,
    ) as reader:
        with pytest.raises(DecisionVMFormatError, match="next address"):
            reader.read_record(0)


def test_small_benchmark_reports_complete_nonrepresentative_evidence(
    tmp_path: Path,
) -> None:
    table = build_disjoint_pointer_table(
        PointerTableConfig(chains=16, steps=64)
    )
    for flags in (FORMAT_COMPACT40, FORMAT_ALIGNED64):
        result = benchmark_format(
            table,
            tmp_path / f"bench-{flags}.vtx",
            flags=flags,
            address_samples=512,
            seed=9,
        )
        assert result.sequential.count == 512
        assert result.shuffled_random.count == 512
        assert result.dependent.count == 64
        assert result.sequential.p50_ns > 0
        assert result.dependent.p99_ns > 0
        assert result.first_replay_cache_hits == 0
        assert result.first_replay_cache_misses == 64
        assert result.cached_replay_cache_hits == 64
        assert result.cached_replay_cache_misses == 0
        assert result.checksum_verified
        assert not result.os_cache_state_controlled
        assert not result.timing_is_target_representative


def test_target_file_size_projection_is_explicit() -> None:
    projection = target_decision_vm_projection()
    assert projection.record_count == 56_175_137_076
    assert projection.start_count == 219_434_129
    assert projection.chain_steps == 256
    assert projection.compact_record_bytes == 5
    assert projection.aligned_record_bytes == 8
    assert projection.header_bytes == 64
    assert projection.compact_total_gib > 263.22
    assert projection.compact_total_gib < 263.23
    assert projection.aligned_total_gib > 420.17
    assert projection.aligned_total_gib < 420.18
    assert not projection.timing_projected
