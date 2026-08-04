from fractions import Fraction

import numpy as np
import pytest

from vortex_runtime.temporal_span_replay import (
    IncrementalModularSpan,
    TemporalSpanError,
    certify_temporal_span,
    exact_float32_fraction,
    favorable_basis_cache_bytes,
    float32_to_field,
    q4_matrix_bytes,
    verify_fraction_witness,
)


PRIMES = (65521, 65519, 65497)


def fraction_mod(value: Fraction, prime: int) -> int:
    return (value.numerator % prime) * pow(value.denominator % prime, -1, prime) % prime


def test_float32_field_map_matches_exact_ratios() -> None:
    values = np.asarray(
        [0.0, -0.0, 1.0, -2.5, np.float32(2**-149), np.float32(2**-126)],
        dtype=np.float32,
    )
    for prime in PRIMES:
        observed = float32_to_field(values, prime)
        expected = np.asarray(
            [fraction_mod(exact_float32_fraction(value), prime) for value in values],
            dtype=np.int64,
        )
        assert np.array_equal(observed, expected)


def test_incremental_modular_span_forms_reduced_basis() -> None:
    span = IncrementalModularSpan(width=4, prime=PRIMES[0])
    assert span.add_float32([0.0, 1.0, 0.0, 0.0])
    assert span.add_float32([1.0, 0.0, 0.0, 0.0])
    assert not span.add_float32([1.0, 1.0, 0.0, 0.0])
    assert span.rank == 2
    span.validate()


def test_low_dimensional_recurrence_and_duplicates() -> None:
    vectors = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    ]
    result = certify_temporal_span(vectors, primes=PRIMES)
    assert result.independent_flags == (True, True, False, False)
    assert result.certified_independent_count == 2
    assert result.maximum_rank_lower_bound == 2
    assert result.exact_duplicate_hits == 1
    assert result.first_exact_duplicate_position == 3
    assert result.rank_disagreement_count == 0


def test_triangular_sequence_forces_new_direction() -> None:
    vectors = [np.eye(8, dtype=np.float32)[index] for index in range(8)]
    result = certify_temporal_span(vectors, primes=PRIMES)
    assert all(result.independent_flags)
    assert result.maximum_rank_lower_bound == 8
    assert all(trajectory[-1] == 8 for trajectory in result.rank_trajectories.values())


def test_exact_fraction_witness() -> None:
    first = np.asarray([1.0, 0.5, -2.0], dtype=np.float32)
    second = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    target = np.asarray([2.0, 0.0, -7.0], dtype=np.float32)
    assert verify_fraction_witness(
        [first, second], target, [Fraction(2), Fraction(-1)]
    )
    assert not verify_fraction_witness(
        [first, second], target, [Fraction(1), Fraction(1)]
    )


def test_random_integer_vectors_reach_dimension_rank() -> None:
    rng = np.random.default_rng(690069)
    matrix = rng.integers(-8, 9, size=(12, 7), dtype=np.int16).astype(np.float32)
    result = certify_temporal_span(list(matrix), primes=PRIMES)
    assert result.maximum_rank_lower_bound <= 7
    assert result.certified_independent_count == result.maximum_rank_lower_bound
    assert result.maximum_rank_lower_bound == 7


def test_nonfinite_fails_closed() -> None:
    with pytest.raises(TemporalSpanError):
        float32_to_field(np.asarray([1.0, np.nan], dtype=np.float32), PRIMES[0])
    with pytest.raises(TemporalSpanError):
        certify_temporal_span(
            [np.asarray([1.0, np.inf], dtype=np.float32)], primes=PRIMES
        )


def test_favorable_cache_accounting_excludes_metadata_but_not_vectors() -> None:
    assert q4_matrix_bytes(64, 256) == 8192
    assert favorable_basis_cache_bytes(
        input_width=64, output_width=256, rank_lower_bound=64
    ) == 81920
