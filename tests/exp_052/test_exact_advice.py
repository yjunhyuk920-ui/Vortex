from __future__ import annotations

import hashlib

import pytest

from vortex_runtime.exact_advice import (
    AdviceAccounting,
    ExactAdviceError,
    ExactPrefixAdviceTable,
    ExactStateAdviceTable,
    PrefixAdviceEntry,
    StateAdviceEntry,
    amortized_build_streams_per_query,
    entries_within_budget,
    independent_state_audit,
    minimum_reuse_for_fraction,
    prefix_digest,
    prefix_universe_size,
    reuse_histogram,
)

REVISION = "a" * 40
CONTRACT = "greedy-temp0"


def state_entry(
    *,
    sha256: bytes | None = None,
    sha512: bytes | None = None,
    prefix: tuple[int, ...] = (1, 2, 3),
    token: int = 4,
) -> StateAdviceEntry:
    raw = b"state-bytes"
    return StateAdviceEntry(
        target_revision=REVISION,
        decode_contract=CONTRACT,
        state_sha256=sha256 or hashlib.sha256(raw).digest(),
        state_sha512=sha512 or hashlib.sha512(raw).digest(),
        state_raw_bytes=len(raw),
        current_token=prefix[-1],
        position=len(prefix) - 1,
        exact_prefix_tokens=prefix,
        next_token=token,
    )


def test_exact_prefix_hit_and_revision_namespace() -> None:
    table = ExactPrefixAdviceTable()
    table.add(
        PrefixAdviceEntry(
            target_revision=REVISION,
            decode_contract=CONTRACT,
            prefix_tokens=(1, 2, 3),
            next_token=4,
        )
    )
    hit = table.query(
        target_revision=REVISION,
        decode_contract=CONTRACT,
        prefix_tokens=(1, 2, 3),
    )
    miss = table.query(
        target_revision="b" * 40,
        decode_contract=CONTRACT,
        prefix_tokens=(1, 2, 3),
    )
    assert hit.hit and hit.next_token == 4
    assert not miss.hit
    assert table.entry_count == 1
    assert table.build_target_calls == 1
    assert table.serialized_bytes > 0


def test_conflicting_prefix_value_fails_closed() -> None:
    table = ExactPrefixAdviceTable()
    first = PrefixAdviceEntry(REVISION, CONTRACT, (1, 2), 3)
    table.add(first)
    with pytest.raises(ExactAdviceError):
        table.add(PrefixAdviceEntry(REVISION, CONTRACT, (1, 2), 9))


def test_state_digest_collision_requires_exact_witness() -> None:
    table = ExactStateAdviceTable()
    shared_sha256 = b"x" * 32
    first = state_entry(
        sha256=shared_sha256,
        sha512=b"a" * 64,
        prefix=(1, 2, 3),
        token=4,
    )
    second = state_entry(
        sha256=shared_sha256,
        sha512=b"b" * 64,
        prefix=(9, 8, 7),
        token=6,
    )
    table.add(first)
    table.add(second)
    first_hit = table.query(
        target_revision=REVISION,
        decode_contract=CONTRACT,
        state_sha256=shared_sha256,
        state_sha512=b"a" * 64,
        current_token=3,
        position=2,
        exact_prefix_tokens=(1, 2, 3),
    )
    collision_miss = table.query(
        target_revision=REVISION,
        decode_contract=CONTRACT,
        state_sha256=shared_sha256,
        state_sha512=b"c" * 64,
        current_token=3,
        position=2,
        exact_prefix_tokens=(1, 2, 3),
    )
    assert first_hit.hit and first_hit.next_token == 4
    assert not collision_miss.hit
    assert first_hit.probes >= 2
    assert table.entry_count == 2


def test_conflicting_exact_state_value_fails_closed() -> None:
    table = ExactStateAdviceTable()
    table.add(state_entry(token=4))
    with pytest.raises(ExactAdviceError):
        table.add(state_entry(token=5))


def test_accounting_charges_build_and_fallback() -> None:
    accounting = AdviceAccounting(
        query_count=100,
        advice_hits=10,
        target_fallback_calls=90,
        build_target_calls=50,
        advice_bytes=1000,
        lookup_probes=120,
    )
    assert accounting.hit_rate == pytest.approx(0.1)
    assert accounting.online_target_fallback_fraction == pytest.approx(0.9)
    assert accounting.target_forward_component_per_query == pytest.approx(1.4)


def test_minimum_reuse_is_85_for_target_fraction() -> None:
    allowed = 0.011851851851851851
    assert minimum_reuse_for_fraction(allowed) == 85
    assert amortized_build_streams_per_query(1, 85) == pytest.approx(1 / 85)


def test_independent_state_coverage_matches_exact_formula() -> None:
    audit = independent_state_audit(
        state_count=2**14,
        coverage_fraction=0.10,
        vocabulary_size=50257,
        seed=202608036,
    )
    assert audit.wrong_hits == 0
    assert audit.measured_hits + audit.measured_fallbacks == 2**14
    assert audit.measured_hit_fraction == audit.expected_hit_fraction


def test_prefix_universe_uses_arbitrary_precision() -> None:
    value = prefix_universe_size(50257, 16)
    assert value > 2**128
    assert prefix_universe_size(256, 4) == sum(256**i for i in range(5))


def test_storage_budget_and_reuse_histogram() -> None:
    assert entries_within_budget(128, 1024) == 8
    histogram = reuse_histogram([b"a", b"b", b"a", b"c", b"a", b"b"])
    assert histogram == {1: 1, 2: 1, 3: 1}


def test_invalid_contracts_fail_closed() -> None:
    with pytest.raises(ExactAdviceError):
        prefix_digest(
            target_revision="",
            decode_contract=CONTRACT,
            tokens=(1,),
        )
    with pytest.raises(ExactAdviceError):
        AdviceAccounting(
            query_count=10,
            advice_hits=3,
            target_fallback_calls=6,
            build_target_calls=1,
            advice_bytes=0,
            lookup_probes=1,
        ).validate()
    with pytest.raises(ExactAdviceError):
        minimum_reuse_for_fraction(1.0)
    with pytest.raises(ExactAdviceError):
        entries_within_budget(0, 100)
