from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import mmap
import os
from pathlib import Path
import random
import statistics
import struct
import time
from typing import Iterable, Sequence
import uuid
import zlib

from vortex_runtime.host_indexed_cell_probe import ExplicitPointerTable

GIB = 1024**3
MAGIC = b"VTXDVM01"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sHHHHQIIQQII8s")
HEADER_BYTES = HEADER_STRUCT.size
START_STRUCT = struct.Struct("<Q")
FORMAT_COMPACT40 = 1
FORMAT_ALIGNED64 = 2
FORMAT_RECORD_BYTES = {
    FORMAT_COMPACT40: 5,
    FORMAT_ALIGNED64: 8,
}


class DecisionVMFormatError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionVMHeader:
    magic: bytes
    version: int
    header_bytes: int
    record_bytes: int
    flags: int
    record_count: int
    start_count: int
    chain_steps: int
    records_offset: int
    starts_offset: int
    payload_crc32: int
    header_crc32: int

    @property
    def expected_file_bytes(self) -> int:
        return (
            self.starts_offset
            + self.start_count * START_STRUCT.size
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["magic"] = self.magic.decode(
            "ascii",
            errors="replace",
        )
        payload["expected_file_bytes"] = self.expected_file_bytes
        return payload


@dataclass(frozen=True)
class DecisionRecord:
    q4_value: int
    next_address: int


@dataclass(frozen=True)
class DecisionVMBuildReport:
    path: str
    format_name: str
    record_bytes: int
    record_count: int
    start_count: int
    chain_steps: int
    file_bytes: int
    payload_crc32: int
    header_crc32: int
    build_ns: int
    atomic_replace: bool
    parent_directory_fsync_attempted: bool
    temporary_file_removed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReplayResult:
    chain: int
    tokens: tuple[int, ...]
    addresses: tuple[int, ...]
    terminal_next_address: int
    logical_record_probes: int
    mmap_record_reads: int
    cache_hits: int
    cache_misses: int
    start_table_reads: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tokens"] = list(self.tokens)
        payload["addresses"] = list(self.addresses)
        return payload


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    mean_ns: float
    minimum_ns: float
    maximum_ns: float
    operations_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FormatBenchmark:
    format_name: str
    record_bytes: int
    build: DecisionVMBuildReport
    sequential: LatencySummary
    shuffled_random: LatencySummary
    dependent: LatencySummary
    first_replay_ns_per_token: float
    cached_replay_ns_per_token: float
    first_replay_cache_hits: int
    first_replay_cache_misses: int
    cached_replay_cache_hits: int
    cached_replay_cache_misses: int
    reopened_replay_ns_per_token: float
    checksum_verified: bool
    os_cache_state_controlled: bool
    timing_is_target_representative: bool

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "record_bytes": self.record_bytes,
            "build": self.build.to_dict(),
            "sequential": self.sequential.to_dict(),
            "shuffled_random": self.shuffled_random.to_dict(),
            "dependent": self.dependent.to_dict(),
            "first_replay_ns_per_token": self.first_replay_ns_per_token,
            "cached_replay_ns_per_token": self.cached_replay_ns_per_token,
            "first_replay_cache_hits": self.first_replay_cache_hits,
            "first_replay_cache_misses": self.first_replay_cache_misses,
            "cached_replay_cache_hits": self.cached_replay_cache_hits,
            "cached_replay_cache_misses": self.cached_replay_cache_misses,
            "reopened_replay_ns_per_token": self.reopened_replay_ns_per_token,
            "checksum_verified": self.checksum_verified,
            "os_cache_state_controlled": self.os_cache_state_controlled,
            "timing_is_target_representative": self.timing_is_target_representative,
        }


@dataclass(frozen=True)
class TargetDecisionVMProjection:
    record_count: int
    start_count: int
    chain_steps: int
    compact_record_bytes: int
    aligned_record_bytes: int
    start_record_bytes: int
    header_bytes: int
    compact_records_gib: float
    aligned_records_gib: float
    starts_gib: float
    compact_total_gib: float
    aligned_total_gib: float
    compact_over_q4_only_gib: float
    aligned_over_q4_only_gib: float
    q4_only_gib: float
    timing_projected: bool

    def to_dict(self) -> dict:
        return asdict(self)


class LRURecordCache:
    def __init__(self, capacity_records: int) -> None:
        if capacity_records < 0:
            raise ValueError("cache capacity must be nonnegative")
        self.capacity_records = capacity_records
        self._records: OrderedDict[int, DecisionRecord] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._records.clear()
        self.hits = 0
        self.misses = 0

    def get(
        self,
        address: int,
        loader,
    ) -> tuple[DecisionRecord, bool]:
        if address in self._records:
            record = self._records.pop(address)
            self._records[address] = record
            self.hits += 1
            return record, True

        self.misses += 1
        record = loader(address)
        if self.capacity_records > 0:
            self._records[address] = record
            while len(self._records) > self.capacity_records:
                self._records.popitem(last=False)
        return record, False


def format_name(flags: int) -> str:
    if flags == FORMAT_COMPACT40:
        return "compact40"
    if flags == FORMAT_ALIGNED64:
        return "aligned64"
    raise DecisionVMFormatError(f"unsupported record format flag {flags}")


def _pack_header(
    *,
    version: int,
    record_bytes: int,
    flags: int,
    record_count: int,
    start_count: int,
    chain_steps: int,
    records_offset: int,
    starts_offset: int,
    payload_crc32: int,
    header_crc32: int,
) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC,
        version,
        HEADER_BYTES,
        record_bytes,
        flags,
        record_count,
        start_count,
        chain_steps,
        records_offset,
        starts_offset,
        payload_crc32 & 0xFFFFFFFF,
        header_crc32 & 0xFFFFFFFF,
        b"\0" * 8,
    )


def _header_with_checksum(
    *,
    record_bytes: int,
    flags: int,
    record_count: int,
    start_count: int,
    chain_steps: int,
    records_offset: int,
    starts_offset: int,
    payload_crc32: int,
) -> tuple[bytes, int]:
    unchecked = _pack_header(
        version=VERSION,
        record_bytes=record_bytes,
        flags=flags,
        record_count=record_count,
        start_count=start_count,
        chain_steps=chain_steps,
        records_offset=records_offset,
        starts_offset=starts_offset,
        payload_crc32=payload_crc32,
        header_crc32=0,
    )
    checksum = zlib.crc32(unchecked) & 0xFFFFFFFF
    return (
        _pack_header(
            version=VERSION,
            record_bytes=record_bytes,
            flags=flags,
            record_count=record_count,
            start_count=start_count,
            chain_steps=chain_steps,
            records_offset=records_offset,
            starts_offset=starts_offset,
            payload_crc32=payload_crc32,
            header_crc32=checksum,
        ),
        checksum,
    )


def _encode_record(
    *,
    q4_value: int,
    next_address: int,
    record_count: int,
    record_bytes: int,
) -> bytes:
    if not 0 <= q4_value < 16:
        raise DecisionVMFormatError("Q4 code is outside [0, 15]")
    if next_address < -1 or next_address >= record_count:
        raise DecisionVMFormatError("next address is outside the table")
    pointer_code = 0 if next_address == -1 else next_address + 1
    word = (pointer_code << 4) | q4_value
    if word >= 1 << (record_bytes * 8):
        raise DecisionVMFormatError(
            "record does not fit the selected encoding"
        )
    return word.to_bytes(record_bytes, "little", signed=False)


def _decode_record_bytes(
    raw: bytes,
    *,
    record_count: int,
) -> DecisionRecord:
    word = int.from_bytes(raw, "little", signed=False)
    q4_value = word & 0xF
    pointer_code = word >> 4
    next_address = -1 if pointer_code == 0 else pointer_code - 1
    if not 0 <= q4_value < 16:
        raise DecisionVMFormatError("decoded Q4 code is invalid")
    if next_address < -1 or next_address >= record_count:
        raise DecisionVMFormatError("decoded next address is invalid")
    return DecisionRecord(q4_value=q4_value, next_address=next_address)


def build_decision_vm_file(
    table: ExplicitPointerTable,
    path: str | Path,
    *,
    flags: int,
) -> DecisionVMBuildReport:
    if flags not in FORMAT_RECORD_BYTES:
        raise ValueError("unsupported decision VM format")
    record_bytes = FORMAT_RECORD_BYTES[flags]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    began = time.perf_counter_ns()
    parent_fsync_attempted = False
    replaced = False

    records_offset = HEADER_BYTES
    starts_offset = (
        records_offset + table.config.cells * record_bytes
    )
    payload_crc = 0

    try:
        with temporary.open("w+b") as handle:
            handle.write(b"\0" * HEADER_BYTES)
            for value, next_address in zip(
                table.values,
                table.next_addresses,
            ):
                encoded = _encode_record(
                    q4_value=value,
                    next_address=next_address,
                    record_count=table.config.cells,
                    record_bytes=record_bytes,
                )
                handle.write(encoded)
                payload_crc = zlib.crc32(encoded, payload_crc)

            for start in table.starts:
                if not 0 <= start < table.config.cells:
                    raise DecisionVMFormatError(
                        "chain start is outside the table"
                    )
                encoded_start = START_STRUCT.pack(start)
                handle.write(encoded_start)
                payload_crc = zlib.crc32(
                    encoded_start,
                    payload_crc,
                )

            payload_crc &= 0xFFFFFFFF
            header, header_crc = _header_with_checksum(
                record_bytes=record_bytes,
                flags=flags,
                record_count=table.config.cells,
                start_count=table.config.chains,
                chain_steps=table.config.steps,
                records_offset=records_offset,
                starts_offset=starts_offset,
                payload_crc32=payload_crc,
            )
            handle.seek(0)
            handle.write(header)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, destination)
        replaced = True
        try:
            directory_fd = os.open(
                destination.parent,
                os.O_RDONLY,
            )
            try:
                parent_fsync_attempted = True
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            parent_fsync_attempted = True

        ended = time.perf_counter_ns()
        return DecisionVMBuildReport(
            path=str(destination),
            format_name=format_name(flags),
            record_bytes=record_bytes,
            record_count=table.config.cells,
            start_count=table.config.chains,
            chain_steps=table.config.steps,
            file_bytes=destination.stat().st_size,
            payload_crc32=payload_crc,
            header_crc32=header_crc,
            build_ns=ended - began,
            atomic_replace=True,
            parent_directory_fsync_attempted=parent_fsync_attempted,
            temporary_file_removed=not temporary.exists(),
        )
    finally:
        if temporary.exists():
            temporary.unlink()
        if not replaced and destination.exists():
            # Never remove an older valid destination after a failed rebuild.
            pass


class DecisionVMReader:
    def __init__(
        self,
        path: str | Path,
        *,
        verify_payload_checksum: bool = True,
    ) -> None:
        self.path = Path(path)
        self._handle = self.path.open("rb")
        try:
            self._mmap = mmap.mmap(
                self._handle.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
            self.header = self._read_and_validate_header()
            self.checksum_verified = False
            if verify_payload_checksum:
                actual = zlib.crc32(
                    self._mmap[
                        self.header.records_offset :
                        self.header.expected_file_bytes
                    ]
                ) & 0xFFFFFFFF
                if actual != self.header.payload_crc32:
                    raise DecisionVMFormatError(
                        "payload checksum mismatch"
                    )
                self.checksum_verified = True
        except Exception:
            if hasattr(self, "_mmap"):
                self._mmap.close()
            self._handle.close()
            raise

    def __enter__(self) -> "DecisionVMReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_mmap") and not self._mmap.closed:
            self._mmap.close()
        if not self._handle.closed:
            self._handle.close()

    def _read_and_validate_header(self) -> DecisionVMHeader:
        if len(self._mmap) < HEADER_BYTES:
            raise DecisionVMFormatError("file is shorter than the header")
        fields = HEADER_STRUCT.unpack_from(self._mmap, 0)
        (
            magic,
            version,
            header_bytes,
            record_bytes,
            flags,
            record_count,
            start_count,
            chain_steps,
            records_offset,
            starts_offset,
            payload_crc32,
            header_crc32,
            _reserved,
        ) = fields

        if magic != MAGIC:
            raise DecisionVMFormatError("bad decision VM magic")
        if version != VERSION:
            raise DecisionVMFormatError("unsupported decision VM version")
        if header_bytes != HEADER_BYTES:
            raise DecisionVMFormatError("unexpected header size")
        if flags not in FORMAT_RECORD_BYTES:
            raise DecisionVMFormatError("unknown record format")
        if record_bytes != FORMAT_RECORD_BYTES[flags]:
            raise DecisionVMFormatError("record size and flags disagree")
        if record_count <= 0 or start_count <= 0 or chain_steps <= 0:
            raise DecisionVMFormatError("invalid zero-sized table")
        if records_offset != HEADER_BYTES:
            raise DecisionVMFormatError("unexpected records offset")
        expected_starts = records_offset + record_count * record_bytes
        if starts_offset != expected_starts:
            raise DecisionVMFormatError("unexpected starts offset")

        unchecked = _pack_header(
            version=version,
            record_bytes=record_bytes,
            flags=flags,
            record_count=record_count,
            start_count=start_count,
            chain_steps=chain_steps,
            records_offset=records_offset,
            starts_offset=starts_offset,
            payload_crc32=payload_crc32,
            header_crc32=0,
        )
        actual_header_crc = zlib.crc32(unchecked) & 0xFFFFFFFF
        if actual_header_crc != header_crc32:
            raise DecisionVMFormatError("header checksum mismatch")

        header = DecisionVMHeader(
            magic=magic,
            version=version,
            header_bytes=header_bytes,
            record_bytes=record_bytes,
            flags=flags,
            record_count=record_count,
            start_count=start_count,
            chain_steps=chain_steps,
            records_offset=records_offset,
            starts_offset=starts_offset,
            payload_crc32=payload_crc32,
            header_crc32=header_crc32,
        )
        if len(self._mmap) != header.expected_file_bytes:
            raise DecisionVMFormatError(
                "file size does not exactly match the header"
            )
        return header

    def read_record(self, address: int) -> DecisionRecord:
        if not 0 <= address < self.header.record_count:
            raise DecisionVMFormatError("record address is out of range")
        offset = (
            self.header.records_offset
            + address * self.header.record_bytes
        )
        raw = self._mmap[
            offset : offset + self.header.record_bytes
        ]
        return _decode_record_bytes(
            raw,
            record_count=self.header.record_count,
        )

    def read_start(self, chain: int) -> int:
        if not 0 <= chain < self.header.start_count:
            raise DecisionVMFormatError("chain index is out of range")
        offset = self.header.starts_offset + chain * START_STRUCT.size
        (address,) = START_STRUCT.unpack_from(self._mmap, offset)
        if address >= self.header.record_count:
            raise DecisionVMFormatError("chain start is out of range")
        return int(address)

    def replay(
        self,
        *,
        chain: int,
        maximum_steps: int | None = None,
        cache: LRURecordCache | None = None,
    ) -> ReplayResult:
        limit = (
            self.header.chain_steps
            if maximum_steps is None
            else maximum_steps
        )
        if limit <= 0:
            raise ValueError("maximum_steps must be positive")

        address = self.read_start(chain)
        addresses: list[int] = []
        tokens: list[int] = []
        mmap_reads = 0
        local_hits = 0
        local_misses = 0

        for _ in range(limit):
            if address < 0:
                break
            addresses.append(address)
            if cache is None:
                record = self.read_record(address)
                mmap_reads += 1
                local_misses += 1
            else:
                record, hit = cache.get(address, self.read_record)
                if hit:
                    local_hits += 1
                else:
                    local_misses += 1
                    mmap_reads += 1
            tokens.append(record.q4_value)
            address = record.next_address

        return ReplayResult(
            chain=chain,
            tokens=tuple(tokens),
            addresses=tuple(addresses),
            terminal_next_address=address,
            logical_record_probes=len(addresses),
            mmap_record_reads=mmap_reads,
            cache_hits=local_hits,
            cache_misses=local_misses,
            start_table_reads=1,
        )


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("latency sample is empty")
    if not 0 <= probability <= 1:
        raise ValueError("probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(values: Sequence[float]) -> LatencySummary:
    if not values:
        raise ValueError("latency sample is empty")
    mean = float(statistics.fmean(values))
    return LatencySummary(
        count=len(values),
        p50_ns=percentile(values, 0.50),
        p95_ns=percentile(values, 0.95),
        p99_ns=percentile(values, 0.99),
        mean_ns=mean,
        minimum_ns=float(min(values)),
        maximum_ns=float(max(values)),
        operations_per_second=(1e9 / mean if mean > 0 else 0.0),
    )


def measure_record_addresses(
    reader: DecisionVMReader,
    addresses: Sequence[int],
) -> LatencySummary:
    samples: list[float] = []
    checksum = 0
    for address in addresses:
        began = time.perf_counter_ns()
        record = reader.read_record(int(address))
        ended = time.perf_counter_ns()
        checksum ^= record.q4_value ^ record.next_address
        samples.append(float(ended - began))
    if checksum == -2:
        raise RuntimeError("unreachable checksum guard")
    return summarize_latencies(samples)


def dependent_latency_samples(
    reader: DecisionVMReader,
    *,
    chain: int,
    steps: int,
) -> LatencySummary:
    address = reader.read_start(chain)
    samples: list[float] = []
    for _ in range(steps):
        if address < 0:
            break
        began = time.perf_counter_ns()
        record = reader.read_record(address)
        ended = time.perf_counter_ns()
        samples.append(float(ended - began))
        address = record.next_address
    return summarize_latencies(samples)


def target_decision_vm_projection(
    *,
    record_count: int = 56_175_137_076,
    chain_steps: int = 256,
) -> TargetDecisionVMProjection:
    if record_count <= 0 or chain_steps <= 0:
        raise ValueError("projection dimensions must be positive")
    start_count = record_count // chain_steps
    q4_only = record_count * 4 / 8 / GIB
    compact_records = record_count * 5 / GIB
    aligned_records = record_count * 8 / GIB
    starts = start_count * START_STRUCT.size / GIB
    compact_total = (HEADER_BYTES / GIB) + compact_records + starts
    aligned_total = (HEADER_BYTES / GIB) + aligned_records + starts
    return TargetDecisionVMProjection(
        record_count=record_count,
        start_count=start_count,
        chain_steps=chain_steps,
        compact_record_bytes=5,
        aligned_record_bytes=8,
        start_record_bytes=START_STRUCT.size,
        header_bytes=HEADER_BYTES,
        compact_records_gib=compact_records,
        aligned_records_gib=aligned_records,
        starts_gib=starts,
        compact_total_gib=compact_total,
        aligned_total_gib=aligned_total,
        compact_over_q4_only_gib=compact_total - q4_only,
        aligned_over_q4_only_gib=aligned_total - q4_only,
        q4_only_gib=q4_only,
        timing_projected=False,
    )


def benchmark_format(
    table: ExplicitPointerTable,
    path: str | Path,
    *,
    flags: int,
    address_samples: int = 20_000,
    seed: int = 44,
) -> FormatBenchmark:
    build = build_decision_vm_file(table, path, flags=flags)
    sample_count = min(address_samples, table.config.cells)
    sequential_addresses = list(range(sample_count))
    shuffled_addresses = list(range(table.config.cells))
    random.Random(seed).shuffle(shuffled_addresses)
    shuffled_addresses = shuffled_addresses[:sample_count]

    with DecisionVMReader(path, verify_payload_checksum=True) as reader:
        source_tokens = tuple(
            table.values[address]
            for address in range(table.config.steps)
        )
        exact = reader.replay(chain=0)
        if exact.tokens != source_tokens:
            raise RuntimeError("mmap replay differs from source table")

        sequential = measure_record_addresses(
            reader,
            sequential_addresses,
        )
        shuffled = measure_record_addresses(
            reader,
            shuffled_addresses,
        )
        dependent = dependent_latency_samples(
            reader,
            chain=table.config.chains // 2,
            steps=table.config.steps,
        )

        cache = LRURecordCache(table.config.steps)
        began = time.perf_counter_ns()
        first = reader.replay(chain=1, cache=cache)
        first_ns = time.perf_counter_ns() - began
        began = time.perf_counter_ns()
        cached = reader.replay(chain=1, cache=cache)
        cached_ns = time.perf_counter_ns() - began

    with DecisionVMReader(path, verify_payload_checksum=False) as reopened:
        began = time.perf_counter_ns()
        reopened_result = reopened.replay(chain=1)
        reopened_ns = time.perf_counter_ns() - began

    if first.tokens != cached.tokens or first.tokens != reopened_result.tokens:
        raise RuntimeError("replay paths disagree")

    token_count = max(1, len(first.tokens))
    return FormatBenchmark(
        format_name=format_name(flags),
        record_bytes=FORMAT_RECORD_BYTES[flags],
        build=build,
        sequential=sequential,
        shuffled_random=shuffled,
        dependent=dependent,
        first_replay_ns_per_token=first_ns / token_count,
        cached_replay_ns_per_token=cached_ns / token_count,
        first_replay_cache_hits=first.cache_hits,
        first_replay_cache_misses=first.cache_misses,
        cached_replay_cache_hits=cached.cache_hits,
        cached_replay_cache_misses=cached.cache_misses,
        reopened_replay_ns_per_token=(
            reopened_ns / max(1, len(reopened_result.tokens))
        ),
        checksum_verified=True,
        os_cache_state_controlled=False,
        timing_is_target_representative=False,
    )
