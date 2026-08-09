from __future__ import annotations

from fractions import Fraction
import itertools

from vortex_runtime.cross_matrix_advice_locality import (
    COVER_RADIUS_FRACTION_WITNESS,
    IDEAL_CACHE_LINE_BITS,
    IDEAL_WORD_BITS,
    certified_covering_radius_cells,
    covering_entropy_witness_holds,
    derive_frontier,
    ideal_packed_probe_count,
    joint_rank_one_probe_lower_bound,
    maximum_rank_one_span_length,
    outside_cancellation_weight,
    registered_non_embedding_specs,
    shortened_local_dimension_upper_bound,
)
from vortex_runtime.query_adaptive_cold_equation import (
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)


def _span(basis: tuple[int, ...] | list[int]) -> set[int]:
    words = {0}
    for row in basis:
        words |= {word ^ row for word in tuple(words)}
    return words


def _gf2_rank(words: set[int] | list[int]) -> int:
    pivots: dict[int, int] = {}
    for word in words:
        current = word
        while current:
            pivot = current.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = current
                break
            current ^= pivots[pivot]
    return len(pivots)


def _distance_to_subspace(word: int, subspace: set[int]) -> int:
    return min((word ^ codeword).bit_count() for codeword in subspace)


def _rank_one_words(rows: int, columns: int, offset: int) -> set[int]:
    words: set[int] = set()
    for left in range(1 << rows):
        for right in range(1 << columns):
            word = 0
            for row in range(rows):
                for column in range(columns):
                    if (left >> row) & 1 and (right >> column) & 1:
                        word |= 1 << (offset + row * columns + column)
            words.add(word)
    return words


def test_global_linear_query_identity_charges_every_outside_bit() -> None:
    width = 5
    local_mask = 0b00011
    advice_space = _span((0b00101, 0b11010))
    for query in range(1 << 2):
        for advice in advice_space:
            residual = query ^ advice
            assert outside_cancellation_weight(
                query, advice, local_mask, width
            ) == (advice & ~local_mask).bit_count()
            for checkpoint in range(1 << width):
                native = (query & checkpoint).bit_count() & 1
                reconstructed = (
                    ((advice & checkpoint).bit_count() & 1)
                    ^ ((residual & checkpoint).bit_count() & 1)
                )
                assert reconstructed == native


def test_zero_outside_probe_uses_exactly_the_shortened_local_space() -> None:
    width = 6
    local_mask = 0b000111
    advice_space = _span((0b001001, 0b010010, 0b000100))
    for advice in advice_space:
        outside = outside_cancellation_weight(
            0b000101, advice, local_mask, width
        )
        assert (outside == 0) == ((advice & ~local_mask) == 0)


def test_fixed_outside_support_buys_at_most_one_dimension_per_probe() -> None:
    width = 6
    local_mask = 0b000111
    outside_coordinates = (1 << 3, 1 << 4, 1 << 5)
    # Exhaust every two-row presentation.  Duplicated presentations are useful
    # here: the reference derives each actual row space independently.
    for first, second in itertools.product(range(1 << width), repeat=2):
        advice_space = _span((first, second))
        shortened = {
            word for word in advice_space if word & ~local_mask == 0
        }
        local_dimension = _gf2_rank(shortened)
        for selector in range(1 << len(outside_coordinates)):
            outside_mask = 0
            for index, coordinate in enumerate(outside_coordinates):
                if (selector >> index) & 1:
                    outside_mask |= coordinate
            fixed_domain = {
                word
                for word in advice_space
                if word & ~(local_mask | outside_mask) == 0
            }
            projected = {word & local_mask for word in fixed_domain}
            assert _gf2_rank(projected) <= shortened_local_dimension_upper_bound(
                local_dimension, outside_mask.bit_count()
            )


def test_shortened_dimensions_direct_sum_but_projection_dimensions_do_not() -> None:
    # U={(x,x)} across two three-bit blocks.  Both projections have dimension
    # three even though dim(U)=three; neither block has a nonzero shortened
    # vector.  This is the smallest useful counterexample to naive projection
    # accounting.
    basis = tuple((1 << index) | (1 << (index + 3)) for index in range(3))
    advice_space = _span(basis)
    left_mask = 0b000111
    right_mask = 0b111000
    assert _gf2_rank({word & left_mask for word in advice_space}) == 3
    assert _gf2_rank({word & right_mask for word in advice_space}) == 3
    assert _gf2_rank(
        {word for word in advice_space if word & ~left_mask == 0}
    ) == 0
    assert _gf2_rank(
        {word for word in advice_space if word & ~right_mask == 0}
    ) == 0
    assert _gf2_rank(advice_space) == 3
    assert outside_cancellation_weight(
        0b000111, 0b111111, left_mask, 6
    ) == 3


def test_rank_one_span_to_cover_lift_holds_exhaustively_on_small_blocks() -> None:
    # Blocks are 2x2 and 1x2, hence rank-one span lengths two and one.  For
    # every two-row global advice presentation, independently compute the
    # worst rank-one distances and the complete ambient covering radius.
    block_queries = (
        _rank_one_words(2, 2, 0),
        _rank_one_words(1, 2, 4),
    )
    for first, second in itertools.product(range(1 << 6), repeat=2):
        advice_space = _span((first, second))
        times = [
            max(_distance_to_subspace(query, advice_space) for query in queries)
            for queries in block_queries
        ]
        covering_radius = max(
            _distance_to_subspace(word, advice_space) for word in range(1 << 6)
        )
        assert covering_radius <= 2 * times[0] + times[1]
        assert covering_radius <= 2 * sum(times)


def test_registered_entropy_witness_is_finite_and_strict() -> None:
    assert COVER_RADIUS_FRACTION_WITNESS == Fraction(13, 50)
    assert covering_entropy_witness_holds()
    assert certified_covering_radius_cells() == 104_974_453_310


def test_registered_population_and_span_length_are_frozen() -> None:
    specs = registered_non_embedding_specs()
    assert sum(spec.count for spec in specs) == 883
    assert sum(spec.parameters for spec in specs) == (
        REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    assert maximum_rank_one_span_length() == 16_384


def test_registered_bound_is_far_below_the_target_and_cannot_reject_it() -> None:
    coefficient_uses = joint_rank_one_probe_lower_bound()
    assert coefficient_uses == 6_407_133
    result = derive_frontier()
    lower = result["registered_lower_bound"]
    assert lower["coefficient_uses"] == coefficient_uses
    assert lower["coefficient_fraction"] == "2135711/134582632448"
    assert lower["bound_over_p50_target"] < 0.00134
    assert lower["p50_target_over_bound"] > 746
    assert result["claim_boundary"][
        "target_sized_direct_sum_for_rank_one_queries"
    ] == "NOT PROVED"
    assert result["claim_boundary"]["core_candidate"] == "NONE"


def test_ideal_word_and_line_mapping_is_only_a_favorable_payload_floor() -> None:
    coefficient_uses = joint_rank_one_probe_lower_bound()
    assert IDEAL_WORD_BITS == 64
    assert IDEAL_CACHE_LINE_BITS == 512
    assert ideal_packed_probe_count(coefficient_uses, 64) == 100_112
    assert ideal_packed_probe_count(coefficient_uses, 512) == 12_514
    result = derive_frontier()
    assert result["registered_lower_bound"]["ideal_packed_payload_bytes"] == (
        800_896
    )
    assert result["claim_boundary"][
        "word_or_address_granular_hardware_lower_bound"
    ] == "NOT PROVED"
