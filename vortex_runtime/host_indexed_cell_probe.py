from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from math import floor
import random
import statistics
import time
from typing import Iterable, Sequence

GIB = 1024**3


@dataclass(frozen=True)
class PointerTableConfig:
    chains: int = 64
    steps: int = 256
    q4_levels: int = 16

    def __post_init__(self) -> None:
        if min(self.chains, self.steps, self.q4_levels) <= 0:
            raise ValueError("pointer-table dimensions must be positive")
        if self.q4_levels != 16:
            raise ValueError("the current Gate uses 16 signed-Q4 codes")

    @property
    def cells(self) -> int:
        return self.chains * self.steps

    @property
    def address_bits(self) -> int:
        return max(1, (self.cells - 1).bit_length())

    @property
    def record_bits(self) -> int:
        return 4 + self.address_bits


@dataclass(frozen=True)
class ExplicitPointerTable:
    config: PointerTableConfig
    values: tuple[int, ...]
    next_addresses: tuple[int, ...]
    starts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.config.cells:
            raise ValueError("value table length mismatch")
        if len(self.next_addresses) != self.config.cells:
            raise ValueError("pointer table length mismatch")
        if len(self.starts) != self.config.chains:
            raise ValueError("chain-start count mismatch")


@dataclass(frozen=True)
class ProbeTrace:
    chain: int
    addresses: tuple[int, ...]
    tokens: tuple[int, ...]
    terminal_next_address: int
    logical_probes: int
    serial_dependency_depth: int
    resident_hits: int
    host_misses: int
    host_record_bits: int
    logical_host_bytes: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CacheFrontierPoint:
    cache_capacity_records: int
    theorem_host_miss_lower_bound: int
    worst_chain: int
    measured_worst_chain_misses: int
    measured_worst_chain_hits: int
    theorem_pass: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdversarialPairResult:
    chain: int
    step: int
    changed_address: int
    prefix_addresses_identical: bool
    prefix_tokens_identical: bool
    current_token_differs: bool
    next_address_differs: bool
    only_one_record_differs: bool
    passes: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HostPrototypeTiming:
    cells: int
    steps_per_repeat: int
    repeats: int
    record_storage_bits: int
    logical_probes_per_step: int
    median_ns_per_probe: float
    minimum_ns_per_probe: float
    maximum_ns_per_probe: float
    checksum: int
    timing_is_target_representative: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetCellProbeProjection:
    q4_cells: int
    chain_steps: int
    complete_chains: int
    ignored_tail_cells: int
    address_bits: int
    q4_bits: int
    explicit_record_bits: int
    explicit_record_bytes: float
    q4_only_metadata_gib: float
    explicit_pointer_table_gib: float
    pointer_overhead_gib: float
    resident_cache_gib: float
    resident_raw_record_capacity: int
    cached_records_per_chain_floor: int
    worst_chain_host_miss_lower_bound: int
    worst_chain_host_miss_fraction: float
    logical_host_bytes_per_chain_lower_bound: float
    logical_host_bytes_per_token_lower_bound: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_disjoint_pointer_table(
    config: PointerTableConfig,
) -> ExplicitPointerTable:
    values: list[int] = []
    next_addresses: list[int] = []
    starts: list[int] = []

    for chain in range(config.chains):
        start = chain * config.steps
        starts.append(start)
        for step in range(config.steps):
            address = start + step
            values.append((chain * 7 + step * 11 + 3) % 16)
            next_addresses.append(
                address + 1 if step + 1 < config.steps else -1
            )

    return ExplicitPointerTable(
        config=config,
        values=tuple(values),
        next_addresses=tuple(next_addresses),
        starts=tuple(starts),
    )


def decode_chain(
    table: ExplicitPointerTable,
    *,
    chain: int,
    resident_records: Iterable[int] = (),
    maximum_steps: int | None = None,
) -> ProbeTrace:
    config = table.config
    if not 0 <= chain < config.chains:
        raise ValueError("chain is outside the table")
    limit = config.steps if maximum_steps is None else maximum_steps
    if limit <= 0:
        raise ValueError("maximum_steps must be positive")

    resident = frozenset(int(value) for value in resident_records)
    address = table.starts[chain]
    addresses: list[int] = []
    tokens: list[int] = []
    hits = 0
    misses = 0

    for _ in range(limit):
        if address < 0:
            break
        addresses.append(address)
        if address in resident:
            hits += 1
        else:
            misses += 1
        tokens.append(table.values[address])
        address = table.next_addresses[address]

    host_bits = misses * config.record_bits
    return ProbeTrace(
        chain=chain,
        addresses=tuple(addresses),
        tokens=tuple(tokens),
        terminal_next_address=address,
        logical_probes=len(addresses),
        serial_dependency_depth=len(addresses),
        resident_hits=hits,
        host_misses=misses,
        host_record_bits=host_bits,
        logical_host_bytes=host_bits / 8,
    )


def worst_chain_host_miss_lower_bound(
    *,
    chains: int,
    steps: int,
    cache_capacity_records: int,
) -> int:
    if min(chains, steps) <= 0:
        raise ValueError("chains and steps must be positive")
    if cache_capacity_records < 0:
        raise ValueError("cache capacity must be nonnegative")
    return max(0, steps - floor(cache_capacity_records / chains))


def balanced_resident_cache(
    config: PointerTableConfig,
    capacity_records: int,
) -> frozenset[int]:
    if capacity_records < 0:
        raise ValueError("cache capacity must be nonnegative")
    capacity = min(capacity_records, config.cells)
    base = capacity // config.chains
    remainder = capacity % config.chains
    cached: set[int] = set()

    for chain in range(config.chains):
        count = min(
            config.steps,
            base + (1 if chain < remainder else 0),
        )
        start = chain * config.steps
        cached.update(range(start, start + count))
    return frozenset(cached)


def sampled_resident_cache(
    config: PointerTableConfig,
    capacity_records: int,
    *,
    seed: int,
) -> frozenset[int]:
    if capacity_records < 0:
        raise ValueError("cache capacity must be nonnegative")
    capacity = min(capacity_records, config.cells)
    generator = random.Random(seed)
    return frozenset(generator.sample(range(config.cells), capacity))


def worst_chain_for_cache(
    table: ExplicitPointerTable,
    resident_records: Iterable[int],
) -> ProbeTrace:
    resident = frozenset(int(value) for value in resident_records)
    traces = [
        decode_chain(
            table,
            chain=chain,
            resident_records=resident,
        )
        for chain in range(table.config.chains)
    ]
    return max(
        traces,
        key=lambda trace: (trace.host_misses, -trace.chain),
    )


def cache_frontier_point(
    table: ExplicitPointerTable,
    *,
    cache_capacity_records: int,
    resident_records: Iterable[int],
) -> CacheFrontierPoint:
    resident = frozenset(int(value) for value in resident_records)
    if len(resident) > cache_capacity_records:
        raise ValueError("resident set exceeds declared capacity")
    trace = worst_chain_for_cache(table, resident)
    lower_bound = worst_chain_host_miss_lower_bound(
        chains=table.config.chains,
        steps=table.config.steps,
        cache_capacity_records=cache_capacity_records,
    )
    return CacheFrontierPoint(
        cache_capacity_records=cache_capacity_records,
        theorem_host_miss_lower_bound=lower_bound,
        worst_chain=trace.chain,
        measured_worst_chain_misses=trace.host_misses,
        measured_worst_chain_hits=trace.resident_hits,
        theorem_pass=trace.host_misses >= lower_bound,
    )


def adversarial_table_pair(
    table: ExplicitPointerTable,
    *,
    chain: int,
    step: int,
) -> tuple[ExplicitPointerTable, ExplicitPointerTable, int]:
    config = table.config
    if not 0 <= chain < config.chains:
        raise ValueError("chain is outside the table")
    if not 0 <= step < config.steps:
        raise ValueError("step is outside the chain")
    if config.chains < 2:
        raise ValueError("the adversary requires at least two chains")

    changed_address = table.starts[chain] + step
    alternate_values = list(table.values)
    alternate_next = list(table.next_addresses)
    alternate_values[changed_address] = (
        alternate_values[changed_address] + 1
    ) % 16
    alternate_chain = (chain + 1) % config.chains
    alternate_pointer = table.starts[alternate_chain]
    if alternate_pointer == alternate_next[changed_address]:
        alternate_chain = (alternate_chain + 1) % config.chains
        alternate_pointer = table.starts[alternate_chain]
    alternate_next[changed_address] = alternate_pointer

    alternate = ExplicitPointerTable(
        config=config,
        values=tuple(alternate_values),
        next_addresses=tuple(alternate_next),
        starts=table.starts,
    )
    return table, alternate, changed_address


def evaluate_adversarial_pair(
    table: ExplicitPointerTable,
    *,
    chain: int,
    step: int,
) -> AdversarialPairResult:
    base, alternate, changed_address = adversarial_table_pair(
        table,
        chain=chain,
        step=step,
    )
    base_trace = decode_chain(base, chain=chain)
    alternate_trace = decode_chain(
        alternate,
        chain=chain,
        maximum_steps=step + 1,
    )

    prefix_addresses_identical = (
        base_trace.addresses[:step]
        == alternate_trace.addresses[:step]
    )
    prefix_tokens_identical = (
        base_trace.tokens[:step]
        == alternate_trace.tokens[:step]
    )
    current_token_differs = (
        base_trace.tokens[step]
        != alternate_trace.tokens[step]
    )
    base_next = base.next_addresses[changed_address]
    alternate_next = alternate.next_addresses[changed_address]
    next_address_differs = base_next != alternate_next

    record_differences = sum(
        1
        for address in range(table.config.cells)
        if (
            base.values[address] != alternate.values[address]
            or base.next_addresses[address]
            != alternate.next_addresses[address]
        )
    )
    only_one_record_differs = record_differences == 1
    passes = bool(
        prefix_addresses_identical
        and prefix_tokens_identical
        and current_token_differs
        and next_address_differs
        and only_one_record_differs
    )
    return AdversarialPairResult(
        chain=chain,
        step=step,
        changed_address=changed_address,
        prefix_addresses_identical=prefix_addresses_identical,
        prefix_tokens_identical=prefix_tokens_identical,
        current_token_differs=current_token_differs,
        next_address_differs=next_address_differs,
        only_one_record_differs=only_one_record_differs,
        passes=passes,
    )


def build_packed_pointer_cycle(
    *,
    cells: int,
    seed: int,
) -> tuple[array, int]:
    if cells <= 1:
        raise ValueError("packed cycle requires at least two cells")
    order = list(range(cells))
    random.Random(seed).shuffle(order)
    records = array("Q", [0]) * cells
    for index, address in enumerate(order):
        next_address = order[(index + 1) % cells]
        value = (address * 7 + index * 11 + 3) & 0xF
        records[address] = (next_address << 4) | value
    return records, order[0]


def benchmark_packed_pointer_chase(
    *,
    cells: int = 262_144,
    steps_per_repeat: int = 200_000,
    repeats: int = 5,
    seed: int = 43,
) -> HostPrototypeTiming:
    if min(cells, steps_per_repeat, repeats) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    records, start = build_packed_pointer_cycle(
        cells=cells,
        seed=seed,
    )
    durations: list[float] = []
    checksum = 0

    address = start
    for _ in range(min(steps_per_repeat, 10_000)):
        record = records[address]
        checksum ^= int(record & 0xF)
        address = int(record >> 4)

    for _ in range(repeats):
        address = start
        local_checksum = 0
        began = time.perf_counter_ns()
        for _step in range(steps_per_repeat):
            record = records[address]
            local_checksum ^= int(record & 0xF)
            address = int(record >> 4)
        ended = time.perf_counter_ns()
        durations.append((ended - began) / steps_per_repeat)
        checksum ^= local_checksum ^ address

    return HostPrototypeTiming(
        cells=cells,
        steps_per_repeat=steps_per_repeat,
        repeats=repeats,
        record_storage_bits=64,
        logical_probes_per_step=1,
        median_ns_per_probe=float(statistics.median(durations)),
        minimum_ns_per_probe=float(min(durations)),
        maximum_ns_per_probe=float(max(durations)),
        checksum=int(checksum),
        timing_is_target_representative=False,
    )


def target_cell_probe_projection(
    *,
    q4_cells: int = 56_175_137_076,
    chain_steps: int = 256,
    resident_cache_gib: float = 8.0,
) -> TargetCellProbeProjection:
    if q4_cells <= 0 or chain_steps <= 0:
        raise ValueError("target dimensions must be positive")
    if resident_cache_gib < 0:
        raise ValueError("resident cache must be nonnegative")

    complete_chains = q4_cells // chain_steps
    if complete_chains <= 0:
        raise ValueError("not enough cells for one complete chain")
    used_cells = complete_chains * chain_steps
    ignored_tail = q4_cells - used_cells
    address_bits = max(1, (q4_cells - 1).bit_length())
    record_bits = 4 + address_bits
    cache_bits = int(resident_cache_gib * GIB * 8)
    cache_records = cache_bits // record_bits
    cached_per_chain = cache_records // complete_chains
    miss_lower_bound = worst_chain_host_miss_lower_bound(
        chains=complete_chains,
        steps=chain_steps,
        cache_capacity_records=cache_records,
    )
    explicit_table_gib = q4_cells * record_bits / 8 / GIB
    q4_only_gib = q4_cells * 4 / 8 / GIB
    bytes_per_record = record_bits / 8
    logical_chain_bytes = miss_lower_bound * bytes_per_record

    return TargetCellProbeProjection(
        q4_cells=q4_cells,
        chain_steps=chain_steps,
        complete_chains=complete_chains,
        ignored_tail_cells=ignored_tail,
        address_bits=address_bits,
        q4_bits=4,
        explicit_record_bits=record_bits,
        explicit_record_bytes=bytes_per_record,
        q4_only_metadata_gib=q4_only_gib,
        explicit_pointer_table_gib=explicit_table_gib,
        pointer_overhead_gib=explicit_table_gib - q4_only_gib,
        resident_cache_gib=resident_cache_gib,
        resident_raw_record_capacity=cache_records,
        cached_records_per_chain_floor=cached_per_chain,
        worst_chain_host_miss_lower_bound=miss_lower_bound,
        worst_chain_host_miss_fraction=(
            miss_lower_bound / chain_steps
        ),
        logical_host_bytes_per_chain_lower_bound=logical_chain_bytes,
        logical_host_bytes_per_token_lower_bound=(
            logical_chain_bytes / chain_steps
        ),
    )
