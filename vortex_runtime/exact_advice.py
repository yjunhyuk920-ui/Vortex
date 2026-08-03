"""Exact target-specific advice reference contracts for EXP-052.

The module models exact full-prefix and exact-state advice. Every entry has an
explicit target namespace, exact collision witness, value, build-call charge,
and serialized storage estimate. A miss is never extrapolated: it requires an
exact target fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Iterable, Mapping, Sequence


class ExactAdviceError(ValueError):
    """Raised when exact advice integrity or accounting fails."""


def _namespace_bytes(target_revision: str, decode_contract: str) -> bytes:
    if not target_revision or not decode_contract:
        raise ExactAdviceError("target revision and decode contract are required")
    return hashlib.sha256(
        (target_revision + "\0" + decode_contract).encode("utf-8")
    ).digest()


def encode_prefix(tokens: Sequence[int]) -> bytes:
    values = tuple(int(token) for token in tokens)
    if not values:
        raise ExactAdviceError("exact prefix must not be empty")
    if any(token < 0 or token > 0xFFFFFFFF for token in values):
        raise ExactAdviceError("prefix token outside uint32 range")
    return struct.pack(">I", len(values)) + b"".join(
        struct.pack(">I", token) for token in values
    )


def prefix_digest(
    *, target_revision: str, decode_contract: str, tokens: Sequence[int]
) -> bytes:
    namespace = _namespace_bytes(target_revision, decode_contract)
    return hashlib.sha256(namespace + encode_prefix(tokens)).digest()


@dataclass(frozen=True)
class PrefixAdviceEntry:
    target_revision: str
    decode_contract: str
    prefix_tokens: tuple[int, ...]
    next_token: int
    build_target_calls: int = 1

    def validate(self) -> None:
        _namespace_bytes(self.target_revision, self.decode_contract)
        encode_prefix(self.prefix_tokens)
        if self.next_token < 0 or self.next_token > 0xFFFFFFFF:
            raise ExactAdviceError("next token outside uint32 range")
        if self.build_target_calls <= 0:
            raise ExactAdviceError("build target calls must be positive")

    @property
    def exact_key(self) -> bytes:
        self.validate()
        return _namespace_bytes(self.target_revision, self.decode_contract) + encode_prefix(
            self.prefix_tokens
        )

    @property
    def serialized_bytes(self) -> int:
        # Exact namespace + prefix + value + build-call metadata + hash-index slot.
        return len(self.exact_key) + 4 + 8 + 16


@dataclass(frozen=True)
class StateAdviceEntry:
    target_revision: str
    decode_contract: str
    state_sha256: bytes
    state_sha512: bytes
    state_raw_bytes: int
    current_token: int
    position: int
    exact_prefix_tokens: tuple[int, ...]
    next_token: int
    build_target_calls: int = 1

    def validate(self) -> None:
        _namespace_bytes(self.target_revision, self.decode_contract)
        if len(self.state_sha256) != 32 or len(self.state_sha512) != 64:
            raise ExactAdviceError("state digests have invalid length")
        if self.state_raw_bytes < 0:
            raise ExactAdviceError("state_raw_bytes must be non-negative")
        if self.current_token < 0 or self.current_token > 0xFFFFFFFF:
            raise ExactAdviceError("current token outside uint32 range")
        if self.position < 0:
            raise ExactAdviceError("position must be non-negative")
        encode_prefix(self.exact_prefix_tokens)
        if self.next_token < 0 or self.next_token > 0xFFFFFFFF:
            raise ExactAdviceError("next token outside uint32 range")
        if self.build_target_calls <= 0:
            raise ExactAdviceError("build target calls must be positive")

    @property
    def digest_bucket(self) -> bytes:
        self.validate()
        namespace = _namespace_bytes(self.target_revision, self.decode_contract)
        return hashlib.sha256(namespace + self.state_sha256).digest()

    @property
    def exact_witness(self) -> tuple[bytes, int, int, tuple[int, ...]]:
        self.validate()
        return (
            self.state_sha512,
            self.current_token,
            self.position,
            self.exact_prefix_tokens,
        )

    @property
    def serialized_bytes_with_raw_collision_witness(self) -> int:
        # Namespace/index digest + SHA-256 + SHA-512 + token/position/value/build
        # metadata + exact prefix + raw-state bytes retained for bitwise checking.
        return (
            32
            + 32
            + 64
            + 4
            + 8
            + 4
            + 8
            + 16
            + len(encode_prefix(self.exact_prefix_tokens))
            + self.state_raw_bytes
        )

    @property
    def serialized_bytes_prefix_collision_witness(self) -> int:
        # A deterministic pinned target can use the complete exact prefix as the
        # collision witness and reconstruct the state. This is smaller than raw
        # KV storage but still stores the exact causal state identity.
        return (
            32
            + 32
            + 64
            + 4
            + 8
            + 4
            + 8
            + 16
            + len(encode_prefix(self.exact_prefix_tokens))
        )


@dataclass(frozen=True)
class AdviceQueryResult:
    hit: bool
    next_token: int | None
    probes: int
    corruption_detected: bool = False


class ExactPrefixAdviceTable:
    def __init__(self) -> None:
        self._entries: dict[bytes, PrefixAdviceEntry] = {}
        self.build_target_calls = 0

    def add(self, entry: PrefixAdviceEntry) -> None:
        entry.validate()
        key = entry.exact_key
        previous = self._entries.get(key)
        if previous is not None and previous.next_token != entry.next_token:
            raise ExactAdviceError("conflicting exact prefix advice value")
        if previous is None:
            self._entries[key] = entry
            self.build_target_calls += entry.build_target_calls

    def query(
        self,
        *,
        target_revision: str,
        decode_contract: str,
        prefix_tokens: Sequence[int],
    ) -> AdviceQueryResult:
        key = _namespace_bytes(target_revision, decode_contract) + encode_prefix(
            prefix_tokens
        )
        entry = self._entries.get(key)
        return AdviceQueryResult(
            hit=entry is not None,
            next_token=None if entry is None else entry.next_token,
            probes=1,
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def serialized_bytes(self) -> int:
        return sum(entry.serialized_bytes for entry in self._entries.values())

    def manifest(self) -> dict[str, object]:
        rows = [
            {
                "key_sha256": hashlib.sha256(key).hexdigest(),
                "next_token": entry.next_token,
                "serialized_bytes": entry.serialized_bytes,
            }
            for key, entry in sorted(self._entries.items())
        ]
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return {
            "entry_count": len(rows),
            "serialized_bytes": self.serialized_bytes,
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }


class ExactStateAdviceTable:
    def __init__(self) -> None:
        self._buckets: dict[bytes, list[StateAdviceEntry]] = {}
        self.build_target_calls = 0

    def add(self, entry: StateAdviceEntry) -> None:
        entry.validate()
        bucket = self._buckets.setdefault(entry.digest_bucket, [])
        for previous in bucket:
            if previous.exact_witness == entry.exact_witness:
                if previous.next_token != entry.next_token:
                    raise ExactAdviceError("conflicting exact state advice value")
                return
        bucket.append(entry)
        self.build_target_calls += entry.build_target_calls

    def query(
        self,
        *,
        target_revision: str,
        decode_contract: str,
        state_sha256: bytes,
        state_sha512: bytes,
        current_token: int,
        position: int,
        exact_prefix_tokens: Sequence[int],
    ) -> AdviceQueryResult:
        if len(state_sha256) != 32 or len(state_sha512) != 64:
            raise ExactAdviceError("query state digests have invalid length")
        namespace = _namespace_bytes(target_revision, decode_contract)
        bucket_key = hashlib.sha256(namespace + state_sha256).digest()
        bucket = self._buckets.get(bucket_key, [])
        witness = (
            state_sha512,
            int(current_token),
            int(position),
            tuple(int(token) for token in exact_prefix_tokens),
        )
        probes = 1
        for entry in bucket:
            probes += 1
            if entry.exact_witness == witness:
                return AdviceQueryResult(True, entry.next_token, probes)
        return AdviceQueryResult(False, None, probes)

    @property
    def entry_count(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    @property
    def serialized_bytes_with_raw_collision_witness(self) -> int:
        return sum(
            entry.serialized_bytes_with_raw_collision_witness
            for bucket in self._buckets.values()
            for entry in bucket
        )

    @property
    def serialized_bytes_prefix_collision_witness(self) -> int:
        return sum(
            entry.serialized_bytes_prefix_collision_witness
            for bucket in self._buckets.values()
            for entry in bucket
        )


@dataclass(frozen=True)
class AdviceAccounting:
    query_count: int
    advice_hits: int
    target_fallback_calls: int
    build_target_calls: int
    advice_bytes: int
    lookup_probes: int

    def validate(self) -> None:
        if self.query_count <= 0:
            raise ExactAdviceError("query_count must be positive")
        for value in (
            self.advice_hits,
            self.target_fallback_calls,
            self.build_target_calls,
            self.advice_bytes,
            self.lookup_probes,
        ):
            if value < 0:
                raise ExactAdviceError("accounting values must be non-negative")
        if self.advice_hits + self.target_fallback_calls != self.query_count:
            raise ExactAdviceError("hits plus fallbacks must equal queries")

    @property
    def hit_rate(self) -> float:
        self.validate()
        return self.advice_hits / self.query_count

    @property
    def target_forward_component_per_query(self) -> float:
        self.validate()
        return (self.build_target_calls + self.target_fallback_calls) / self.query_count

    @property
    def online_target_fallback_fraction(self) -> float:
        self.validate()
        return self.target_fallback_calls / self.query_count


def minimum_reuse_for_fraction(allowed_fraction: float) -> int:
    if not math.isfinite(allowed_fraction) or not 0.0 < allowed_fraction < 1.0:
        raise ExactAdviceError("allowed fraction must lie in (0,1)")
    return math.ceil(1.0 / allowed_fraction)


def amortized_build_streams_per_query(build_calls: int, exact_reuses: int) -> float:
    if build_calls < 0 or exact_reuses <= 0:
        raise ExactAdviceError("invalid build/reuse accounting")
    return build_calls / exact_reuses


def prefix_universe_size(vocabulary_size: int, maximum_length: int) -> int:
    if vocabulary_size <= 1 or maximum_length < 0:
        raise ExactAdviceError("invalid vocabulary or maximum length")
    return sum(pow(vocabulary_size, length) for length in range(maximum_length + 1))


@dataclass(frozen=True)
class IndependentStateAudit:
    state_count: int
    advice_entries: int
    vocabulary_size: int
    measured_hits: int
    measured_fallbacks: int
    expected_hit_fraction: float
    measured_hit_fraction: float
    wrong_hits: int


def independent_state_audit(
    *,
    state_count: int,
    coverage_fraction: float,
    vocabulary_size: int,
    seed: int,
) -> IndependentStateAudit:
    if state_count <= 0 or vocabulary_size <= 1:
        raise ExactAdviceError("invalid independent-state configuration")
    if not math.isfinite(coverage_fraction) or not 0.0 <= coverage_fraction <= 1.0:
        raise ExactAdviceError("coverage fraction outside [0,1]")
    entries = min(state_count, max(0, int(round(state_count * coverage_fraction))))

    def exact_token(state: int) -> int:
        payload = struct.pack(">QQ", seed & 0xFFFFFFFFFFFFFFFF, state)
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % vocabulary_size

    table = {state: exact_token(state) for state in range(entries)}
    hits = 0
    fallbacks = 0
    wrong = 0
    for state in range(state_count):
        expected = exact_token(state)
        if state in table:
            hits += 1
            wrong += table[state] != expected
        else:
            fallbacks += 1
    expected_fraction = entries / state_count
    measured_fraction = hits / state_count
    if wrong:
        raise ExactAdviceError("independent-state advice produced wrong hit")
    return IndependentStateAudit(
        state_count=state_count,
        advice_entries=entries,
        vocabulary_size=vocabulary_size,
        measured_hits=hits,
        measured_fallbacks=fallbacks,
        expected_hit_fraction=expected_fraction,
        measured_hit_fraction=measured_fraction,
        wrong_hits=wrong,
    )


def entries_within_budget(entry_bytes: int, budget_bytes: int) -> int:
    if entry_bytes <= 0 or budget_bytes < 0:
        raise ExactAdviceError("invalid storage budget")
    return budget_bytes // entry_bytes


def reuse_histogram(keys: Iterable[bytes]) -> Mapping[int, int]:
    counts: dict[bytes, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    histogram: dict[int, int] = {}
    for count in counts.values():
        histogram[count] = histogram.get(count, 0) + 1
    return dict(sorted(histogram.items()))
