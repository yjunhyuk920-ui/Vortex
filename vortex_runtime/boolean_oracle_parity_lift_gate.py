"""Exact black-box lift gate from Boolean-semiring MatVec to GF(2) MatVec.

The gate is intentionally narrow.  A caller may adaptively query the complete
Boolean-semiring product ``B_M(s)`` for arbitrary binary vectors ``s`` and wants
the GF(2) product ``M v``.  All matrix-dependent information is required to
enter through those oracle responses.

For one designated row and target support T=supp(v), compare

    R0 = 1_T
    Ri = indicator(T with coordinate i removed)

for every i in T.  The requested parity flips between R0 and Ri.  A Boolean OR
query S distinguishes the pair exactly when ``S intersect T == {i}``.  Hence
one query can distinguish at most one deletion witness.  Along R0's transcript,
every exact deterministic lift must therefore issue at least |T| isolating
queries.  For full support these are exactly the n singleton queries.

This does *not* constrain an algorithm that directly probes or reinterprets the
Boolean data structure's matrix-dependent internal representation; that is a
direct GF(2) data-structure problem, outside this black-box gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Iterable, Sequence


def _validate_width(width: int) -> None:
    if width <= 0:
        raise ValueError("width must be positive")


def support_mask(width: int, support: Iterable[int]) -> int:
    """Encode a coordinate support as one nonnegative integer bit mask."""

    _validate_width(width)
    mask = 0
    for coordinate in support:
        if coordinate < 0 or coordinate >= width:
            raise ValueError("support coordinate outside width")
        mask |= 1 << coordinate
    return mask


def boolean_row_or(row_mask: int, query_mask: int) -> int:
    if row_mask < 0 or query_mask < 0:
        raise ValueError("masks must be nonnegative")
    return int(bool(row_mask & query_mask))


def gf2_row_parity(row_mask: int, target_mask: int) -> int:
    if row_mask < 0 or target_mask < 0:
        raise ValueError("masks must be nonnegative")
    return (row_mask & target_mask).bit_count() & 1


def deletion_witness_rows(target_mask: int) -> tuple[int, tuple[int, ...]]:
    """Return R0 and all coordinates i for Ri=R0 without i."""

    if target_mask <= 0:
        raise ValueError("target support must be nonempty")
    coordinates = tuple(
        index for index in range(target_mask.bit_length()) if target_mask & (1 << index)
    )
    return target_mask, coordinates


def query_isolates_coordinate(
    query_mask: int,
    target_mask: int,
    coordinate: int,
) -> bool:
    """Whether this OR query distinguishes R0 from deletion witness Ri."""

    if query_mask < 0 or target_mask <= 0:
        raise ValueError("invalid masks")
    bit = 1 << coordinate
    if not target_mask & bit:
        raise ValueError("coordinate is outside target support")
    return (query_mask & target_mask) == bit


def isolated_coordinates(
    query_masks: Sequence[int],
    target_mask: int,
) -> set[int]:
    """Return deletion witnesses distinguished by the R0-path query sequence."""

    _, coordinates = deletion_witness_rows(target_mask)
    isolated: set[int] = set()
    for query in query_masks:
        for coordinate in coordinates:
            if query_isolates_coordinate(query, target_mask, coordinate):
                isolated.add(coordinate)
                # One query intersects T in one fixed set and therefore can
                # isolate at most one coordinate.
                break
    return isolated


def indistinguishable_deletions(
    query_masks: Sequence[int],
    target_mask: int,
) -> tuple[int, ...]:
    """Deletion witnesses still sharing R0's complete Boolean transcript."""

    row0, coordinates = deletion_witness_rows(target_mask)
    survivors: list[int] = []
    transcript0 = tuple(boolean_row_or(row0, query) for query in query_masks)
    for coordinate in coordinates:
        row_i = row0 & ~(1 << coordinate)
        transcript_i = tuple(boolean_row_or(row_i, query) for query in query_masks)
        if transcript_i == transcript0:
            survivors.append(coordinate)
    return tuple(survivors)


def adversarial_certificate(
    query_masks: Sequence[int],
    target_mask: int,
) -> dict[str, object]:
    """Certify whether the observed R0 path can already determine target parity."""

    row0, coordinates = deletion_witness_rows(target_mask)
    survivors = indistinguishable_deletions(query_masks, target_mask)
    parity0 = gf2_row_parity(row0, target_mask)
    witness = None
    if survivors:
        coordinate = survivors[0]
        row_i = row0 & ~(1 << coordinate)
        witness = {
            "coordinate": coordinate,
            "row0_parity": parity0,
            "deletion_parity": gf2_row_parity(row_i, target_mask),
            "transcript_equal": tuple(
                boolean_row_or(row0, query) for query in query_masks
            )
            == tuple(boolean_row_or(row_i, query) for query in query_masks),
        }
    return {
        "target_support_size": len(coordinates),
        "queries": len(query_masks),
        "isolated_coordinates": sorted(isolated_coordinates(query_masks, target_mask)),
        "indistinguishable_deletions": list(survivors),
        "exact_parity_determined_on_adversary_family": not survivors,
        "opposite_parity_witness": witness,
    }


def minimum_black_box_boolean_calls(target_support_size: int) -> int:
    """Exact worst-case lower bound for the frozen deletion adversary."""

    if target_support_size <= 0:
        raise ValueError("target support size must be positive")
    return target_support_size


def inner_product_one_entries(width: int) -> int:
    """Number of ones in the complete GF(2) inner-product truth matrix."""

    _validate_width(width)
    return ((1 << width) - 1) * (1 << (width - 1))


def maximum_inner_product_one_rectangle_entries(width: int) -> int:
    """Exact largest all-one rectangle in IP_width(a,b)=a dot b mod 2.

    Let A x B be a nonempty all-one rectangle and choose a0 in A.  The span U
    of all differences a+a0 is orthogonal to every b in B.  Since a0 dot b=1,
    a0 is not in U, so the affine condition a0 dot b=1 cuts U^perp in half:

        |A| <= 2^dim(U)
        |B| <= 2^(width-dim(U)-1).

    Thus |A||B| <= 2^(width-1).  A singleton nonzero row attains equality.
    """

    _validate_width(width)
    return 1 << (width - 1)


def exact_inner_product_boolean_rank(width: int) -> int:
    """Exact Boolean rank / one-rectangle cover number of GF(2) inner product."""

    _validate_width(width)
    # Lower bound: total ones divided by the exact maximum rectangle size.
    lower = (
        inner_product_one_entries(width)
        // maximum_inner_product_one_rectangle_entries(width)
    )
    # Upper bound: one rectangle for each nonzero left vector, containing that
    # single row and all right vectors on which its inner product is one.
    upper = (1 << width) - 1
    if lower != upper:  # pragma: no cover - algebraic identity guard.
        raise AssertionError("Boolean-rank lower/upper bounds unexpectedly differ")
    return upper


def exhaustive_maximum_one_rectangle(width: int) -> int:
    """Enumerate every nonempty left subset for width <=4 as a control."""

    _validate_width(width)
    if width > 4:
        raise ValueError("rectangle exhaustive control is limited to width <= 4")
    vectors = tuple(range(1 << width))
    best = 0
    for left_subset_mask in range(1, 1 << len(vectors)):
        left = [
            vector
            for index, vector in enumerate(vectors)
            if left_subset_mask & (1 << index)
        ]
        right = [
            vector
            for vector in vectors
            if all(gf2_row_parity(a, vector) == 1 for a in left)
        ]
        best = max(best, len(left) * len(right))
    return best


def one_shot_boolean_feature_lift_projection(width: int, output_rows: int) -> dict[str, object]:
    """Storage lower bound for one Boolean-product feature lift.

    A representation

        parity(a,v) = OR_k (E(a)[k] AND Phi(v)[k])

    is precisely a Boolean factorization of the inner-product truth matrix, even
    when E and Phi are arbitrary nonlinear functions.  Its feature dimension is
    therefore at least the exact Boolean rank ``2^width-1``.
    """

    _validate_width(width)
    if output_rows <= 0:
        raise ValueError("output_rows must be positive")
    features = exact_inner_product_boolean_rank(width)
    stored_row_feature_bits = output_rows * features
    decimal_digits = len(str(features))
    return {
        "right_width": width,
        "output_rows": output_rows,
        "minimum_boolean_features": features,
        "minimum_feature_decimal_digits": decimal_digits,
        "minimum_row_feature_bits": stored_row_feature_bits,
        "source_binary_matrix_bits": output_rows * width,
        "feature_storage_over_source": stored_row_feature_bits / (output_rows * width),
    }


def canonical_isolating_queries(width: int, target_mask: int) -> tuple[int, ...]:
    """One matching upper sequence: isolate each target-support coordinate."""

    _validate_width(width)
    if target_mask <= 0 or target_mask >= (1 << width):
        raise ValueError("target mask outside width")
    return tuple(
        1 << coordinate
        for coordinate in range(width)
        if target_mask & (1 << coordinate)
    )


def exhaustive_short_transcript_check(width: int) -> dict[str, int | bool]:
    """Exhaustively verify that <n distinct subset queries never suffice.

    This control is intentionally small.  Adaptivity does not add another case
    split on the R0 path: every nonempty query gets answer 1 when T is full, so
    a deterministic adaptive algorithm has one fixed sequence on that path.
    """

    _validate_width(width)
    if width > 5:
        raise ValueError("exhaustive control is limited to width <= 5")
    target = (1 << width) - 1
    subsets = tuple(range(1, 1 << width))
    checked = 0
    false_determinations = 0
    maximum_queries = width - 1
    for count in range(maximum_queries + 1):
        for queries in combinations(subsets, count):
            checked += 1
            certificate = adversarial_certificate(queries, target)
            if certificate["exact_parity_determined_on_adversary_family"]:
                false_determinations += 1
    canonical = canonical_isolating_queries(width, target)
    canonical_certificate = adversarial_certificate(canonical, target)
    return {
        "width": width,
        "short_query_sets_checked": checked,
        "false_short_determinations": false_determinations,
        "canonical_queries": len(canonical),
        "canonical_determines_family": bool(
            canonical_certificate["exact_parity_determined_on_adversary_family"]
        ),
    }


@dataclass(frozen=True)
class WidthProjection:
    width: int
    word_bits: int = 64

    @property
    def required_boolean_calls(self) -> int:
        return minimum_black_box_boolean_calls(self.width)

    @property
    def boolean_response_bits(self) -> int:
        # The frozen oracle returns a complete n-bit Boolean product per call.
        return self.width * self.required_boolean_calls

    @property
    def source_matrix_bits(self) -> int:
        return self.width * self.width

    @property
    def response_over_source_bits(self) -> float:
        return self.boolean_response_bits / self.source_matrix_bits

    @property
    def response_words(self) -> int:
        return self.required_boolean_calls * ceil(self.width / self.word_bits)

    @property
    def larsen_williams_composed_exponent(self) -> float:
        # n black-box calls times the O(n^(7/4)/sqrt(w)) deterministic Boolean
        # query implementation.  This records composition scaling, not a lower
        # bound on all possible direct GF(2) data structures.
        return 11 / 4

    @property
    def direct_dense_exponent(self) -> float:
        return 2.0


def audit_payload(
    widths: Sequence[int] = (4, 8, 25, 108, 216, 16384),
    word_bits: int = 64,
) -> dict[str, object]:
    if word_bits <= 0:
        raise ValueError("word_bits must be positive")
    projections = [WidthProjection(width, word_bits) for width in widths]
    return {
        "classification": "E0_BOOLEAN_ORACLE_PARITY_LIFT_GATE",
        "theorem": {
            "statement": (
                "A deterministic exact black-box lift from complete Boolean-semiring "
                "MatVec responses to GF(2) Mv requires at least |supp(v)| Boolean "
                "product queries on the deletion-witness family."
            ),
            "full_support_minimum_calls": "n",
            "proof_core": (
                "For R0=1_T and Ri=1_(T\\{i}), a Boolean subset query S "
                "distinguishes the pair iff S intersect T={i}; each query isolates "
                "at most one i while target parity flips for every i."
            ),
        },
        "small_exhaustive_controls": [
            exhaustive_short_transcript_check(width) for width in range(1, 6)
        ],
        "one_shot_feature_lift_theorem": {
            "statement": (
                "Any arbitrary nonlinear row/query feature maps E,Phi satisfying "
                "GF2_inner_product(a,v)=OR_k(E(a)[k] AND Phi(v)[k]) for every "
                "a,v in F2^d require at least 2^d-1 Boolean features."
            ),
            "exact_boolean_rank": "2^d - 1",
            "maximum_one_rectangle_entries": "2^(d-1)",
            "total_one_entries": "(2^d-1)*2^(d-1)",
            "upper_construction": "one rectangle for every nonzero left vector",
            "small_exhaustive_rectangle_controls": [
                {
                    "width": width,
                    "exhaustive_maximum_one_rectangle_entries": exhaustive_maximum_one_rectangle(width),
                    "closed_form_maximum_one_rectangle_entries": maximum_inner_product_one_rectangle_entries(width),
                    "exact_boolean_rank": exact_inner_product_boolean_rank(width),
                }
                for width in range(1, 5)
            ],
            "storage_projections": [
                one_shot_boolean_feature_lift_projection(width, rows)
                for width, rows in ((8, 4), (25, 25), (108, 25), (216, 25))
            ],
        },
        "width_projections": [
            {
                "width": item.width,
                "word_bits": item.word_bits,
                "required_boolean_product_calls": item.required_boolean_calls,
                "complete_boolean_response_bits": item.boolean_response_bits,
                "binary_source_matrix_bits": item.source_matrix_bits,
                "complete_response_over_source_bits": item.response_over_source_bits,
                "complete_response_words": item.response_words,
                "lw_black_box_composed_exponent": item.larsen_williams_composed_exponent,
                "direct_binary_dense_exponent": item.direct_dense_exponent,
            }
            for item in projections
        ],
        "literature_context": {
            "larsen_williams": (
                "Deterministic systematic Boolean-semiring MatVec cell-probe query "
                "O(n^(7/4)/sqrt(w)); SODA 2017 / arXiv:1605.01695."
            ),
            "chakraborty_kamma_larsen": (
                "STOC 2018 / arXiv:1711.04467 gives a faster randomized succinct "
                "Boolean upper bound and Boolean/F2 systematic lower bounds."
            ),
        },
        "decision": "REJECT_BLACK_BOX_BOOLEAN_TO_F2_LIFT_AS_SUBDENSE_CORE",
        "claim_boundary": {
            "adaptive_deterministic_boolean_oracle_lift": "REJECTED",
            "single_transformed_boolean_product_feature_lift": "REJECTED_EXPONENTIAL_FEATURES",
            "direct_probe_of_boolean_index_internals": "OPEN",
            "direct_gf2_data_structure": "OPEN",
            "native_q4_bf16_fp32_lift": "OPEN",
            "global_cross_matrix_advice": "OPEN",
            "whole_mission": "OPEN",
            "target_hardware": "NOT_TESTED",
        },
    }
