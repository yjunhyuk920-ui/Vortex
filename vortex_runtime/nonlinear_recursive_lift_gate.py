"""Gate for recursively lifting nonlinear local query encodings.

Suppose one stored parent block is formed entrywise from ``k`` child blocks::

    Y[p] = f(X_1[p], ..., X_k[p]).

A tempting recursion is to answer a parity query on ``Y`` using only the same
query answers on the child blocks.  The two-site identity below shows that
this works for every source and every rank-one query only when ``f`` is
affine.  Any genuinely nonlinear coordinate therefore needs additional
cross-child correlation sources; a small nonlinear probe code cannot be
tensorized for free.

The theorem is deliberately narrow.  It does not reject a non-entrywise
global encoding that explicitly stores and charges those correlations.
"""

from __future__ import annotations


DECISION = (
    "REJECT_FREE_ENTRYWISE_RECURSION_OF_NONLINEAR_PROBE_SEEDS_"
    "REQUIRE_CHARGED_CORRELATION_SOURCES_OR_GLOBAL_ENCODING"
)


def _validate_truth_table(*, arity: int, truth_table: int) -> None:
    if arity <= 0:
        raise ValueError("arity must be positive")
    table_size = 1 << arity
    if truth_table < 0 or truth_table >= (1 << table_size):
        raise ValueError("truth table does not fit the declared arity")


def function_value(*, arity: int, truth_table: int, point: int) -> int:
    _validate_truth_table(arity=arity, truth_table=truth_table)
    if point < 0 or point >= (1 << arity):
        raise ValueError("point does not fit the declared arity")
    return (truth_table >> point) & 1


def _unchecked_value(truth_table: int, point: int) -> int:
    return (truth_table >> point) & 1


def affine_coefficients(
    *, arity: int, truth_table: int
) -> tuple[int, tuple[int, ...]] | None:
    """Return ``(constant, slopes)`` exactly when ``f`` is affine over F2."""

    _validate_truth_table(arity=arity, truth_table=truth_table)
    constant = _unchecked_value(truth_table, 0)
    slopes = tuple(
        _unchecked_value(truth_table, 1 << coordinate)
        ^ constant
        for coordinate in range(arity)
    )
    for point in range(1 << arity):
        expected = constant
        for coordinate, slope in enumerate(slopes):
            expected ^= slope & ((point >> coordinate) & 1)
        if _unchecked_value(truth_table, point) != expected:
            return None
    return constant, slopes


def two_site_factorization_witness(
    *, arity: int, truth_table: int
) -> dict[str, int] | None:
    """Find equal child parities with different lifted output parities.

    For two selected matrix positions, the child query answers are the bits
    of ``x XOR y``.  If ``f(x) XOR f(y)`` is not determined by that difference,
    no decoder receiving only the child answers can recover the parent query.
    """

    _validate_truth_table(arity=arity, truth_table=truth_table)
    first_by_difference: dict[int, tuple[int, int, int]] = {}
    for left in range(1 << arity):
        for right in range(1 << arity):
            difference = left ^ right
            parity = _unchecked_value(
                truth_table, left
            ) ^ _unchecked_value(truth_table, right)
            previous = first_by_difference.get(difference)
            if previous is None:
                first_by_difference[difference] = (left, right, parity)
                continue
            old_left, old_right, old_parity = previous
            if old_parity != parity:
                return {
                    "child_parity_vector": difference,
                    "first_left": old_left,
                    "first_right": old_right,
                    "first_parent_parity": old_parity,
                    "second_left": left,
                    "second_right": right,
                    "second_parent_parity": parity,
                }
    return None


def two_site_parity_factors(
    *, arity: int, truth_table: int
) -> bool:
    """Whether parent parity is a function only of child parities."""

    return (
        two_site_factorization_witness(
            arity=arity, truth_table=truth_table
        )
        is None
    )


def exhaustive_arity_audit(*, maximum_arity: int = 4) -> list[dict[str, int]]:
    """Exhaustively verify factorization iff affinity at small arities."""

    if maximum_arity <= 0 or maximum_arity > 4:
        raise ValueError("registered exhaustive range is arity 1 through 4")
    rows: list[dict[str, int]] = []
    for arity in range(1, maximum_arity + 1):
        table_size = 1 << arity
        function_count = 1 << table_size
        affine_count = 0
        factoring_count = 0
        mismatch_count = 0
        for truth_table in range(function_count):
            affine = (
                affine_coefficients(
                    arity=arity, truth_table=truth_table
                )
                is not None
            )
            factors = two_site_parity_factors(
                arity=arity, truth_table=truth_table
            )
            affine_count += int(affine)
            factoring_count += int(factors)
            mismatch_count += int(affine != factors)
        rows.append(
            {
                "arity": arity,
                "boolean_function_count": function_count,
                "affine_function_count": affine_count,
                "two_site_factoring_function_count": factoring_count,
                "classification_mismatch_count": mismatch_count,
            }
        )
    return rows


def derive_audit() -> dict[str, object]:
    """Return the exact theorem boundary and registered finite controls."""

    rows = exhaustive_arity_audit()
    if any(row["classification_mismatch_count"] for row in rows):
        raise AssertionError("two-site factorization/affinity equivalence drifted")
    for row in rows:
        expected_affine = 1 << (row["arity"] + 1)
        if row["affine_function_count"] != expected_affine:
            raise AssertionError("registered affine function count drifted")

    # f(x_0,x_1)=x_0 AND x_1, encoded in little-endian truth-table order.
    and_table = 1 << 3
    and_witness = two_site_factorization_witness(
        arity=2, truth_table=and_table
    )
    if and_witness is None:
        raise AssertionError("registered nonlinear AND witness disappeared")

    return {
        "classification": "E0_NONLINEAR_RECURSIVE_LIFT_GATE",
        "contract": {
            "parent_cell_is_entrywise_boolean_function": True,
            "child_blocks_are_arbitrary_and_independent": True,
            "query_contains_two_sites_in_one_row_or_column": True,
            "parent_decoder_receives_only_matching_child_query_answers": True,
            "decoder_computation_is_free_and_arbitrary": True,
        },
        "theorem": {
            "factorization_condition": (
                "f(x) XOR f(y) depends only on x XOR y"
            ),
            "equivalent_form": (
                "h(x)=f(x) XOR f(0); h(x XOR y)=h(x) XOR h(y)"
            ),
            "conclusion": "f is affine over F2",
            "converse": "every affine f factors through child parities",
        },
        "exhaustive_controls": rows,
        "nonlinear_and_counterexample": and_witness,
        "consequence": {
            "free_recursive_lift_of_nonlinear_seed": False,
            "additional_correlation_sources_required": True,
            "toy_capacity_survivor_promoted": False,
        },
        "claim_boundary": {
            "entrywise_black_box_recursion_covered": True,
            "non_entrywise_global_encodings_covered": False,
            "charged_hadamard_correlation_blocks_covered": False,
            "native_numerical_semantics_covered": False,
            "surviving_runtime_candidate": False,
        },
        "decision": DECISION,
    }
