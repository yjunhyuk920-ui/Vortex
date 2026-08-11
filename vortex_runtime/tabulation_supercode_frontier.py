"""Finite Gate for nonadaptive XOR tabulation and Segre-aligned supercodes.

The candidate compiler may store arbitrary Boolean functions of a binary
checkpoint.  A query chooses a fixed, nonadaptive subset of stored bits and
XORs them.  Algebraic-normal-form uniqueness shows that every such exact
scheme can be replaced, without changing its recovery sets, by stored linear
forms.  Nonlinear preprocessing therefore gives no advantage in this scoped
decoder model.

After that reduction, the remaining construction problem is a structured
sparse functional dictionary: near-source-many atoms whose low-weight subset
sums contain every binary rank-one mask.  This module records an exact
capacity diagnostic for local square blocks.  Capacity is necessary only; it
does not construct an aligned dictionary, a decoder, or a native numerical
runtime.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Mapping, Sequence


REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
GLOBAL_ADVICE_BITS = 8 * 8 * (1 << 30)
REGISTERED_P50_FRACTION = Fraction(8, 675)

DECISION = (
    "REDUCE_NONADAPTIVE_XOR_TABULATION_TO_SPARSE_FUNCTIONAL_DICTIONARY_"
    "KEEP_SEGRE_ALIGNED_SUPERCODE_OPEN_UNCONSTRUCTED"
)


def parity(value: int) -> int:
    """Return the parity of a non-negative integer."""

    if value < 0:
        raise ValueError("value must be non-negative")
    return value.bit_count() & 1


def linear_truth_table(query_mask: int, dimension: int) -> tuple[int, ...]:
    """Truth table of ``x -> <query_mask, x>`` over GF(2)."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if query_mask < 0 or query_mask >= (1 << dimension):
        raise ValueError("query mask is outside the ambient dimension")
    return tuple(parity(query_mask & point) for point in range(1 << dimension))


def anf_coefficients(
    truth_table: Sequence[int], dimension: int
) -> tuple[int, ...]:
    """Return the unique algebraic-normal-form coefficients over GF(2).

    Index ``mask`` is the coefficient of the monomial containing exactly the
    variables selected by ``mask``.  The in-place subset Mobius transform is
    its own inverse over GF(2).
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    expected = 1 << dimension
    if len(truth_table) != expected:
        raise ValueError("truth table length must be 2**dimension")
    if any(value not in (0, 1) for value in truth_table):
        raise ValueError("truth table values must be bits")
    coefficients = list(truth_table)
    for bit in range(dimension):
        flag = 1 << bit
        for mask in range(expected):
            if mask & flag:
                coefficients[mask] ^= coefficients[mask ^ flag]
    return tuple(coefficients)


def _xor_selected_truth_tables(
    cell_truth_tables: Sequence[Sequence[int]],
    recovery: Sequence[int],
) -> tuple[int, ...]:
    width = len(cell_truth_tables[0])
    answer = [0] * width
    for cell_index in recovery:
        for point, value in enumerate(cell_truth_tables[cell_index]):
            answer[point] ^= value
    return tuple(answer)


def verify_and_linearize_nonadaptive_xor(
    *,
    cell_truth_tables: Sequence[Sequence[int]],
    recovery_sets: Mapping[int, Sequence[int]],
    dimension: int,
) -> dict[str, object]:
    """Verify a tiny exact scheme and extract its equivalent linear atoms.

    ``cell_truth_tables[j]`` may be an arbitrary Boolean function of the data.
    For query mask ``q``, ``recovery_sets[q]`` is a fixed set of cells whose
    XOR must equal ``<q,x>`` for every database ``x``.  ANF uniqueness then
    forces the selected degree-one coefficients to XOR to ``q`` and all
    selected higher-degree coefficients to cancel.

    The explicit truth-table control is intentionally tiny.  The theorem it
    witnesses is algebraic and applies to every finite dimension; this helper
    is not an exhaustive search over target-sized data.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not cell_truth_tables:
        raise ValueError("at least one cell is required")
    expected = 1 << dimension
    normalized = [tuple(table) for table in cell_truth_tables]
    if any(len(table) != expected for table in normalized):
        raise ValueError("every cell truth table must have length 2**dimension")
    if any(value not in (0, 1) for table in normalized for value in table):
        raise ValueError("cell truth tables must contain bits")
    if not recovery_sets:
        raise ValueError("at least one recovery set is required")

    cell_anf = [anf_coefficients(table, dimension) for table in normalized]
    linear_atoms: list[int] = []
    for coefficients in cell_anf:
        atom = 0
        for variable in range(dimension):
            if coefficients[1 << variable]:
                atom |= 1 << variable
        linear_atoms.append(atom)

    query_certificates: list[dict[str, object]] = []
    for query_mask in sorted(recovery_sets):
        if query_mask < 0 or query_mask >= (1 << dimension):
            raise ValueError("query mask is outside the ambient dimension")
        recovery = tuple(recovery_sets[query_mask])
        if len(recovery) != len(set(recovery)):
            raise ValueError("a recovery set may not repeat a cell")
        if any(index < 0 or index >= len(normalized) for index in recovery):
            raise ValueError("recovery set contains an invalid cell index")
        if _xor_selected_truth_tables(normalized, recovery) != linear_truth_table(
            query_mask, dimension
        ):
            raise ValueError("recovery set is not exact on every database")

        selected_coefficients = [0] * expected
        selected_atom = 0
        for index in recovery:
            selected_atom ^= linear_atoms[index]
            for monomial, coefficient in enumerate(cell_anf[index]):
                selected_coefficients[monomial] ^= coefficient
        higher_degree_nonzero = any(
            selected_coefficients[monomial]
            for monomial in range(expected)
            if monomial.bit_count() >= 2
        )
        if selected_coefficients[0] != 0:
            raise AssertionError("exact linear answer retained a constant term")
        if higher_degree_nonzero:
            raise AssertionError("exact linear answer retained nonlinear ANF terms")
        if selected_atom != query_mask:
            raise AssertionError("degree-one coefficients do not recover the query")
        query_certificates.append(
            {
                "query_mask": query_mask,
                "recovery_set": list(recovery),
                "recovered_linear_atom": selected_atom,
                "constant_terms_cancel": True,
                "higher_degree_terms_cancel": True,
                "same_recovery_set_works_after_linearization": True,
            }
        )

    nonlinear_cells = sum(
        any(
            coefficient
            for monomial, coefficient in enumerate(coefficients)
            if monomial.bit_count() >= 2
        )
        for coefficients in cell_anf
    )
    return {
        "dimension": dimension,
        "cell_count": len(normalized),
        "nonlinear_input_cell_count": nonlinear_cells,
        "linearized_atom_masks": linear_atoms,
        "query_certificates": query_certificates,
        "all_queries_exact_before_linearization": True,
        "all_queries_exact_after_linearization": True,
    }


def distinct_binary_rank_one_masks(block_side: int) -> int:
    """Number of distinct rank-at-most-one masks in a square GF(2) block."""

    if block_side <= 0:
        raise ValueError("block side must be positive")
    return 1 + ((1 << block_side) - 1) ** 2


def selection_ball(items: int, maximum_weight: int) -> int:
    """Number of subsets of at most ``maximum_weight`` among ``items``."""

    if items <= 0:
        raise ValueError("items must be positive")
    if maximum_weight < 0 or maximum_weight > items:
        raise ValueError("maximum weight must lie in [0, items]")
    return sum(comb(items, weight) for weight in range(maximum_weight + 1))


def minimum_capacity_weight(*, items: int, query_count: int) -> int:
    """Smallest subset weight whose raw selection count reaches query count."""

    if items <= 0 or query_count <= 0:
        raise ValueError("items and query count must be positive")
    if query_count > (1 << items):
        raise ValueError("even all subsets cannot name every query")
    running = 0
    for weight in range(items + 1):
        running += comb(items, weight)
        if running >= query_count:
            return weight
    raise AssertionError("capacity loop failed despite a feasible query count")


def proportional_local_cell_grant(local_dimension: int) -> int:
    """Favorably allocate all global binary advice uniformly to one block."""

    if local_dimension <= 0:
        raise ValueError("local dimension must be positive")
    return (
        local_dimension
        * (REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_ADVICE_BITS)
        // REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )


def block_capacity_case(block_side: int) -> dict[str, object]:
    """Necessary sparse-sum capacity diagnostic for one square block.

    A favorable implicit dictionary gets ``S=floor((D+H)/D*b^2)`` one-bit
    atoms and pays no atom-description or decoding cost.  To cover all local
    rank-one masks with at most ``t`` atoms it is necessary that

        sum(k=0..t, C(S,k)) >= 1 + (2**b-1)**2.

    Meeting this inequality is not sufficient: subset sums may collide or
    miss the structured rank-one set.
    """

    if block_side <= 0:
        raise ValueError("block side must be positive")
    dimension = block_side * block_side
    cells = proportional_local_cell_grant(dimension)
    queries = distinct_binary_rank_one_masks(block_side)
    weight = minimum_capacity_weight(items=cells, query_count=queries)
    ball = selection_ball(cells, weight)
    nonempty_ball = ball - 1
    nonzero_queries = queries - 1
    target_met_by_counting = Fraction(weight, dimension) <= (
        REGISTERED_P50_FRACTION
    )

    # For independent uniform atoms, every fixed nonempty subset sum is
    # uniform in F_2^dimension.  A union bound therefore puts the probability
    # of hitting one fixed nonzero query below nonempty_ball / 2**dimension.
    # Multiplying by the query count upper-bounds the expected number of
    # nonzero rank-one queries hit.  Integer floor exponents avoid float drift.
    fixed_hit_log2_floor = nonempty_ball.bit_length() - 1 - dimension
    expected_hits_numerator = nonzero_queries * nonempty_ball
    expected_hits_log2_floor = (
        expected_hits_numerator.bit_length() - 1 - dimension
    )
    return {
        "block_side": block_side,
        "ambient_dimension": dimension,
        "favorable_dictionary_cells": cells,
        "distinct_rank_at_most_one_queries": str(queries),
        "minimum_weight_not_rejected_by_capacity": weight,
        "selection_ball_at_minimum_weight": str(ball),
        "selection_ball_floor_log2": ball.bit_length() - 1,
        "probe_fraction": str(Fraction(weight, dimension)),
        "probe_fraction_decimal": float(Fraction(weight, dimension)),
        "registered_target_fraction": str(REGISTERED_P50_FRACTION),
        "capacity_threshold_meets_registered_fraction": target_met_by_counting,
        "random_atoms_fixed_query_hit_probability_log2_floor_upper": (
            fixed_hit_log2_floor
        ),
        "random_atoms_expected_nonzero_rank_one_hits_log2_floor_upper": (
            expected_hits_log2_floor
        ),
        "capacity_is_a_construction": False,
        "rank_one_alignment_established": False,
    }


def first_capacity_case_meeting_target(maximum_side: int = 128) -> dict[str, object]:
    """Return the first side whose necessary counting threshold fits target."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
    for side in range(1, maximum_side + 1):
        case = block_capacity_case(side)
        if case["capacity_threshold_meets_registered_fraction"]:
            return case
    raise ValueError("no capacity case met the target in the requested range")


def _nonlinear_control() -> dict[str, object]:
    dimension = 3
    tables: list[tuple[int, ...]] = []
    for cell in range(4):
        values: list[int] = []
        for point in range(1 << dimension):
            x0 = (point >> 0) & 1
            x1 = (point >> 1) & 1
            x2 = (point >> 2) & 1
            if cell == 0:
                value = x0 ^ (x1 & x2)
            elif cell == 1:
                value = x1 & x2
            elif cell == 2:
                value = x1 ^ (x0 & x2)
            else:
                value = x0 & x2
            values.append(value)
        tables.append(tuple(values))
    return verify_and_linearize_nonadaptive_xor(
        cell_truth_tables=tables,
        recovery_sets={1: (0, 1), 2: (2, 3), 3: (0, 1, 2, 3)},
        dimension=dimension,
    )


def derive_audit() -> dict[str, object]:
    cases = [block_capacity_case(side) for side in (16, 22, 23, 24, 32)]
    first = first_capacity_case_meeting_target()
    if first["block_side"] != 23:
        raise AssertionError("registered first capacity witness drifted")
    return {
        "classification": "E0_TABULATION_SUPERCODE_FRONTIER",
        "contract": {
            "registered_non_embedding_binary_bits": (
                REGISTERED_NON_EMBEDDING_COEFFICIENTS
            ),
            "global_advice_bits": GLOBAL_ADVICE_BITS,
            "favorable_total_one_bit_cells": (
                REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_ADVICE_BITS
            ),
            "favorable_cell_ratio": str(
                Fraction(
                    REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_ADVICE_BITS,
                    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
                )
            ),
            "registered_p50_fraction": str(REGISTERED_P50_FRACTION),
            "binary_single_query_screen_only": True,
        },
        "linearization_lemma": {
            "preprocessing_cells": "arbitrary Boolean functions z_j(W)",
            "addresses": "fixed nonadaptive recovery set S(q)",
            "decoder": "XOR of selected one-bit cells",
            "conclusion": (
                "degree-one ANF atoms g_j preserve every recovery set and "
                "satisfy q=XOR(j in S(q),g_j)"
            ),
            "proof": (
                "ANF is unique; exact equality to a linear character forces "
                "selected constants and higher-degree monomials to cancel"
            ),
            "finite_nonlinear_control": _nonlinear_control(),
        },
        "segre_aligned_sparse_supercode": {
            "definition": (
                "near-source-many implicit atoms whose weight-at-most-t XOR "
                "sums contain every binary rank-one mask"
            ),
            "necessary_capacity_equation": (
                "sum(k=0..t,C(S,k)) >= 1+(2^b-1)^2"
            ),
            "first_case_meeting_target_by_counting_only": first,
            "diagnostic_cases": cases,
            "random_atoms_are_alignment_constructor": False,
            "explicit_aligned_atoms": False,
            "succinct_decomposer": False,
        },
        "claim_boundary": {
            "nonadaptive_fixed_addresses_only": True,
            "xor_only_decoder_only": True,
            "arbitrary_boolean_preprocessing_within_scope": True,
            "adaptive_data_dependent_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_bf16_fp32_query_covered": False,
            "joint_32_query_physical_union_covered": False,
            "constructor_supplied": False,
            "general_lower_bound_supplied": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_NONLINEAR_PREPROCESSING_AS_DISTINCT_ADVANTAGE_FOR_NONADAPTIVE_XOR_TABULATION",
            "REDUCE_SCOPED_TABULATION_TO_SPARSE_FUNCTIONAL_DICTIONARY",
            "REJECT_RANDOM_ATOMS_AS_ALIGNMENT_ARGUMENT",
            "KEEP_SEGRE_ALIGNED_SPARSE_SUPERCODE_OPEN_UNCONSTRUCTED",
            "KEEP_ADAPTIVE_NONLINEAR_FINITE_WORD_DECODER_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
