"""Constructive gates for implicit direct-query representations.

This module implements three preregistered candidate principles:

* a query-side binary pattern-image frame with checkpoint-independent addresses;
* a value-class inverted finite-word executor preserving row reduction order;
* a coordinate-gauge control for Hadamard/elementwise nonlinearities.

The implementations are deliberately finite and explicit.  None is promoted to
the VORTEX core unless its complete paid route survives the registered gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import ceil
from typing import Callable, Iterable, Sequence

from vortex_runtime.lower_bound_audit import llama_405b_tensor_plan


REGISTERED_TARGET_FRACTION = Fraction(8, 675)
WORD_BITS = 64
GIB = 1 << 30


def registered_non_embedding_specs():
    return tuple(s for s in llama_405b_tensor_plan() if s.name != "embedding")


def ten_eleven_partition(width: int) -> tuple[int, ...]:
    """Partition a registered-width vector into 10/11-bit direct-address blocks.

    For ``width=10q+r`` use ``r`` eleven-bit and ``q-r`` ten-bit blocks.  This
    keeps the number of nonzero image reads at ``floor(width/10)``.  Registered
    Transformer widths are large enough that ``q >= r``.
    """

    if width <= 0:
        raise ValueError("width must be positive")
    q, r = divmod(width, 10)
    if r == 0:
        return (10,) * q
    if q < r:
        raise ValueError("10/11 partition is only registered for sufficiently wide vectors")
    result = (11,) * r + (10,) * (q - r)
    if sum(result) != width:
        raise AssertionError("partition drift")
    return result


def pattern_image_storage_atoms(block_widths: Sequence[int]) -> int:
    if not block_widths or any(width <= 0 for width in block_widths):
        raise ValueError("positive block widths required")
    return sum((1 << width) - 1 for width in block_widths)


def registered_sparse_image_frame_audit() -> dict[str, object]:
    """Exact model-wide payload/storage accounting for Principle I."""

    raw_bits = 0
    transformed_bits = 0
    worst_query_bits = 0
    rows = []
    for spec in registered_non_embedding_specs():
        widths = ten_eleven_partition(spec.columns)
        atoms = pattern_image_storage_atoms(widths)
        blocks = len(widths)
        source = spec.count * spec.rows * spec.columns
        transformed = spec.count * spec.rows * atoms
        query = spec.count * spec.rows * blocks
        raw_bits += source
        transformed_bits += transformed
        worst_query_bits += query
        rows.append(
            {
                "name": spec.name,
                "count": spec.count,
                "rows": spec.rows,
                "columns": spec.columns,
                "blocks": blocks,
                "patterns": atoms,
                "storage_over_binary_source": atoms / spec.columns,
                "worst_query_fraction": blocks / spec.columns,
            }
        )

    query_fraction = Fraction(worst_query_bits, raw_bits)
    return {
        "raw_binary_bits": raw_bits,
        "transformed_image_bits": transformed_bits,
        "transformed_image_gib": transformed_bits / 8 / GIB,
        "storage_over_binary_source": transformed_bits / raw_bits,
        "worst_query_bits": worst_query_bits,
        "worst_query_gib": worst_query_bits / 8 / GIB,
        "worst_query_fraction": str(query_fraction),
        "worst_query_fraction_decimal": float(query_fraction),
        "removes_at_least_90_percent_binary_source": query_fraction <= Fraction(1, 10),
        "target_multiple": float(query_fraction / REGISTERED_TARGET_FRACTION),
        "native_universal_fast_path_proved": False,
        "matrix_rows": rows,
    }


def equal_block_pattern_tradeoff(block_width: int) -> dict[str, object]:
    """Continuous full-pattern image tradeoff for one equal-width block scheme."""

    if block_width <= 0:
        raise ValueError("block_width must be positive")
    query_fraction = Fraction(1, block_width)
    storage_ratio = Fraction((1 << block_width) - 1, block_width)
    return {
        "block_width": block_width,
        "query_fraction": str(query_fraction),
        "query_fraction_decimal": float(query_fraction),
        "storage_ratio": str(storage_ratio),
        "storage_ratio_decimal": float(storage_ratio),
        "passes_registered_target_fraction": query_fraction <= REGISTERED_TARGET_FRACTION,
    }


def minimum_equal_block_width_for_target() -> int:
    """Smallest b with 1/b <= 8/675."""

    return ceil(REGISTERED_TARGET_FRACTION.denominator / REGISTERED_TARGET_FRACTION.numerator)


def three_bound_minimum_frame_atoms(dimension: int, radius: int) -> int:
    """Proof-safe atom lower bound from the sparse-sum counting inequality.

    A frame with ``S`` binary atoms and representations using at most ``R``
    distinct atoms can name at most ``sum_{j<=R} C(S,j)`` vectors.  For
    ``R<=S/2`` the standard bound is

        sum C(S,j) < (3S/R)^R.

    We return the least integer S whose favorable ``(3S/R)^R`` upper bound is
    *not already below* ``2^dimension``.  Therefore every smaller S is proved
    insufficient.  This is a lower bound, not an existence construction.
    """

    if dimension <= 0 or radius <= 0 or radius > dimension:
        raise ValueError("require 0 < radius <= dimension")

    def count_upper_is_too_small(atoms: int) -> bool:
        if atoms < 2 * radius:
            return True
        return pow(3 * atoms, radius) < (1 << dimension) * pow(radius, radius)

    low = radius
    high = max(2 * radius, dimension)
    while count_upper_is_too_small(high):
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if count_upper_is_too_small(mid):
            low = mid + 1
        else:
            high = mid
    return low


def registered_one_sided_frame_lower_bounds() -> dict[str, object]:
    """Lower-bound complete-image storage for 10% and 8/675 right-factor frames."""

    rows = []
    for width in sorted({spec.columns for spec in registered_non_embedding_specs()}):
        radius_90 = width // 10
        radius_target = (REGISTERED_TARGET_FRACTION.numerator * width) // REGISTERED_TARGET_FRACTION.denominator
        lower_90 = three_bound_minimum_frame_atoms(width, radius_90)
        lower_target = three_bound_minimum_frame_atoms(width, radius_target)
        rows.append(
            {
                "input_width": width,
                "radius_90": radius_90,
                "minimum_atoms_90_three_bound": lower_90,
                "minimum_complete_image_storage_ratio_90": lower_90 / width,
                "radius_target": radius_target,
                "minimum_atoms_target_three_bound": lower_target,
                "minimum_complete_image_storage_ratio_target": lower_target / width,
            }
        )
    return {
        "model": "ONE_SIDED_LINEAR_COMPLETE_IMAGE_FRAME_ONLY",
        "rows": rows,
        "does_not_cover_general_nonlinear_cells": True,
    }


@dataclass(frozen=True)
class PatternImageFrame:
    rows: int
    columns: int
    block_widths: tuple[int, ...]
    # Each block maps a nonzero local binary pattern to an m-bit output vector.
    images: tuple[dict[int, int], ...]


def _matrix_rows_to_masks(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if not matrix or not matrix[0]:
        raise ValueError("nonempty matrix required")
    columns = len(matrix[0])
    masks = []
    for row in matrix:
        if len(row) != columns or any(bit not in (0, 1) for bit in row):
            raise ValueError("matrix must be rectangular and binary")
        mask = 0
        for j, bit in enumerate(row):
            mask |= bit << j
        masks.append(mask)
    return tuple(masks)


def compile_pattern_image_frame(
    matrix: Sequence[Sequence[int]], block_widths: Sequence[int]
) -> PatternImageFrame:
    """Compile exact GF(2) block images; finite for every declared block list."""

    row_masks = _matrix_rows_to_masks(matrix)
    rows = len(row_masks)
    columns = len(matrix[0])
    widths = tuple(block_widths)
    if not widths or any(width <= 0 for width in widths) or sum(widths) != columns:
        raise ValueError("block widths must be positive and sum to matrix columns")

    images: list[dict[int, int]] = []
    offset = 0
    for width in widths:
        local: dict[int, int] = {}
        mask = (1 << width) - 1
        for pattern in range(1, 1 << width):
            out = 0
            for i, row_mask in enumerate(row_masks):
                row_chunk = (row_mask >> offset) & mask
                out |= ((row_chunk & pattern).bit_count() & 1) << i
            local[pattern] = out
        images.append(local)
        offset += width
    return PatternImageFrame(rows, columns, widths, tuple(images))


def query_pattern_image_frame(frame: PatternImageFrame, vector_mask: int) -> tuple[int, int]:
    """Return ``(Wv, nonzero image reads)`` from direct current-query addresses."""

    if vector_mask < 0 or vector_mask >= (1 << frame.columns):
        raise ValueError("query is outside declared vector width")
    out = 0
    reads = 0
    offset = 0
    for width, images in zip(frame.block_widths, frame.images, strict=True):
        pattern = (vector_mask >> offset) & ((1 << width) - 1)
        if pattern:
            out ^= images[pattern]
            reads += 1
        offset += width
    return out, reads


def direct_gf2_matvec_mask(matrix: Sequence[Sequence[int]], vector_mask: int) -> int:
    row_masks = _matrix_rows_to_masks(matrix)
    columns = len(matrix[0])
    if vector_mask < 0 or vector_mask >= (1 << columns):
        raise ValueError("query is outside declared vector width")
    out = 0
    for i, row_mask in enumerate(row_masks):
        out |= ((row_mask & vector_mask).bit_count() & 1) << i
    return out


@dataclass(frozen=True)
class ValueClassColumn:
    # (weight_word, sorted row indices)
    classes: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ValueClassIndex:
    rows: int
    columns: int
    column_classes: tuple[ValueClassColumn, ...]
    memberships: int
    distinct_products_per_query: int


def compile_value_class_index(matrix: Sequence[Sequence[int]]) -> ValueClassIndex:
    """Invert a finite-word matrix by column and exact weight-word value."""

    if not matrix or not matrix[0]:
        raise ValueError("nonempty matrix required")
    rows = len(matrix)
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("rectangular matrix required")
    compiled = []
    distinct = 0
    for j in range(columns):
        groups: dict[int, list[int]] = {}
        for i in range(rows):
            groups.setdefault(int(matrix[i][j]), []).append(i)
        classes = tuple((word, tuple(indices)) for word, indices in sorted(groups.items()))
        distinct += len(classes)
        compiled.append(ValueClassColumn(classes))
    return ValueClassIndex(rows, columns, tuple(compiled), rows * columns, distinct)


def query_value_class_index(
    index: ValueClassIndex,
    vector: Sequence[int],
    *,
    multiply: Callable[[int, int], int],
    add: Callable[[int, int], int],
    zero: int,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Execute shared leaf products while preserving left-to-right row adds."""

    if len(vector) != index.columns:
        raise ValueError("query width mismatch")
    acc = [zero for _ in range(index.rows)]
    products_done = 0
    additions = 0
    memberships = 0
    for j, column in enumerate(index.column_classes):
        for weight_word, row_indices in column.classes:
            product_word = multiply(weight_word, int(vector[j]))
            products_done += 1
            for row in row_indices:
                # Columns are visited in reference order, so each row observes
                # exactly the same ordered sequence of add operations.
                acc[row] = add(acc[row], product_word)
                additions += 1
                memberships += 1
    return tuple(acc), {
        "products": products_done,
        "additions": additions,
        "memberships": memberships,
    }


def query_value_state_coalesced(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
    *,
    multiply: Callable[[int, int], int],
    add: Callable[[int, int], int],
    zero: int,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Strengthened Principle II: share identical (accumulator, weight) updates.

    This is an online nonlinear grouping because the accumulator word is part of
    the group key.  The implementation remains exact: rows in one group have
    identical incoming accumulator bits and identical leaf weight bits, so their
    next accumulator word is identical under deterministic finite-word mul/add.
    """

    if not matrix or len(vector) != len(matrix[0]):
        raise ValueError("shape mismatch")
    rows = len(matrix)
    columns = len(vector)
    if any(len(row) != columns for row in matrix):
        raise ValueError("rectangular matrix required")
    acc = [zero for _ in range(rows)]
    pair_updates = 0
    products = 0
    for j in range(columns):
        # Product values are still shared by exact weight word.
        product_cache: dict[int, int] = {}
        groups: dict[tuple[int, int], list[int]] = {}
        for i in range(rows):
            weight = int(matrix[i][j])
            groups.setdefault((acc[i], weight), []).append(i)
        for (incoming, weight), members in groups.items():
            if weight not in product_cache:
                product_cache[weight] = multiply(weight, int(vector[j]))
                products += 1
            updated = add(incoming, product_cache[weight])
            pair_updates += 1
            for row in members:
                acc[row] = updated
    return tuple(acc), {
        "state_weight_pair_updates": pair_updates,
        "products": products,
        "row_memberships": rows * columns,
    }


def fp32_exact_integer_state_coalescing_adversary(rows: int, columns: int) -> dict[str, object]:
    """Exact-in-FP32 dense family keeping every row accumulator distinct.

    First-column weights are 1..m, later weights are all 1, and all query values
    are 1.  For the registered dimensions all running values stay below 2**24,
    so IEEE FP32 multiplication/addition of these integers is exact.  After
    column j, row i holds i+j+1 (with zero-based indices); all rows therefore
    occupy distinct accumulator states at every step.
    """

    if rows <= 0 or columns <= 0 or rows + columns >= (1 << 24):
        raise ValueError("adversary requires positive dimensions and exact FP32 integer range")
    pair_updates = rows * columns
    baseline_scalar_leaf_updates = rows * columns
    return {
        "rows": rows,
        "columns": columns,
        "all_coefficients_nonzero": True,
        "all_accumulator_states_distinct_after_every_column": True,
        "fp32_integer_operations_exact": True,
        "state_weight_pair_updates": pair_updates,
        "baseline_scalar_leaf_updates": baseline_scalar_leaf_updates,
        "state_coalescing_update_fraction": 1.0,
        "removes_at_least_90_percent_state_updates": False,
    }


def direct_ordered_finite_word_matvec(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
    *,
    multiply: Callable[[int, int], int],
    add: Callable[[int, int], int],
    zero: int,
) -> tuple[int, ...]:
    if not matrix or len(vector) != len(matrix[0]):
        raise ValueError("shape mismatch")
    out = []
    for row in matrix:
        acc = zero
        for weight, value in zip(row, vector, strict=True):
            acc = add(acc, multiply(int(weight), int(value)))
        out.append(acc)
    return tuple(out)


def value_class_operation_bounds(rows: int, columns: int, distinct_per_column: int) -> dict[str, object]:
    """Favorable scalar-operation accounting for Principle II."""

    if rows <= 0 or columns <= 0 or not 1 <= distinct_per_column <= rows:
        raise ValueError("invalid value-class dimensions")
    baseline_multiplies = rows * columns
    baseline_adds = rows * columns
    candidate_multiplies = distinct_per_column * columns
    candidate_adds = rows * columns
    baseline = baseline_multiplies + baseline_adds
    candidate = candidate_multiplies + candidate_adds
    fraction = Fraction(candidate, baseline)
    return {
        "baseline_multiplies": baseline_multiplies,
        "baseline_adds_favorable": baseline_adds,
        "candidate_multiplies": candidate_multiplies,
        "candidate_adds": candidate_adds,
        "candidate_operation_fraction": str(fraction),
        "candidate_operation_fraction_decimal": float(fraction),
        "removes_at_least_90_percent_operations": fraction <= Fraction(1, 10),
        "membership_visits": rows * columns,
        "membership_fraction_of_dense_coefficients": 1.0,
    }


def gf2_apply_linear_map(columns: Sequence[int], x: int, dimension: int) -> int:
    """Apply a GF(2) linear map represented by output-column bitmasks."""

    if len(columns) != dimension or x < 0 or x >= (1 << dimension):
        raise ValueError("invalid GF2 map/vector")
    out = 0
    for j, column in enumerate(columns):
        if (x >> j) & 1:
            out ^= column
    return out


def gf2_coordinate_product(x: int, y: int) -> int:
    return x & y


def gf2_matrix_rank(columns: Sequence[int], dimension: int) -> int:
    rows = [0] * dimension
    for j, column in enumerate(columns):
        if column < 0 or column >= (1 << dimension):
            raise ValueError("column outside dimension")
        for i in range(dimension):
            rows[i] |= ((column >> i) & 1) << j
    rank = 0
    col = 0
    while col < dimension and rank < dimension:
        pivot = next((r for r in range(rank, dimension) if (rows[r] >> col) & 1), None)
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(dimension):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1
        col += 1
    return rank


def preserves_coordinate_product(columns: Sequence[int], dimension: int) -> bool:
    if gf2_matrix_rank(columns, dimension) != dimension:
        return False
    # Bilinearity means basis-pair checks suffice.
    basis = [1 << i for i in range(dimension)]
    for x in basis:
        for y in basis:
            lhs = gf2_apply_linear_map(columns, gf2_coordinate_product(x, y), dimension)
            ex = gf2_apply_linear_map(columns, x, dimension)
            ey = gf2_apply_linear_map(columns, y, dimension)
            if lhs != gf2_coordinate_product(ex, ey):
                return False
    return True


def preserves_coordinate_product_isotopy(
    left: Sequence[int], right: Sequence[int], out: Sequence[int], dimension: int
) -> bool:
    """Check C(x*y)=A(x)*B(y) for three invertible GF(2) linear gauges."""

    if any(gf2_matrix_rank(columns, dimension) != dimension for columns in (left, right, out)):
        return False
    basis = [1 << i for i in range(dimension)]
    for x in basis:
        for y in basis:
            lhs = gf2_apply_linear_map(out, x & y, dimension)
            rhs = gf2_apply_linear_map(left, x, dimension) & gf2_apply_linear_map(right, y, dimension)
            if lhs != rhs:
                return False
    return True


def exhaustive_linear_hadamard_isotopy_control(dimension: int) -> dict[str, int]:
    """Small exact control for independent input/output gauges at a product node."""

    if not 1 <= dimension <= 2:
        raise ValueError("registered exhaustive isotopy control supports dimensions 1..2")
    invertible_maps = []
    for columns in product(range(1 << dimension), repeat=dimension):
        if gf2_matrix_rank(columns, dimension) == dimension:
            invertible_maps.append(columns)
    triples = 0
    preserving = 0
    nonpermutation = 0
    for left in invertible_maps:
        for right in invertible_maps:
            for out in invertible_maps:
                triples += 1
                if preserves_coordinate_product_isotopy(left, right, out, dimension):
                    preserving += 1
                    if not (
                        is_permutation_linear_map(left, dimension)
                        and is_permutation_linear_map(right, dimension)
                        and is_permutation_linear_map(out, dimension)
                        and tuple(left) == tuple(right) == tuple(out)
                    ):
                        nonpermutation += 1
    return {
        "dimension": dimension,
        "invertible_map_count": len(invertible_maps),
        "triples_checked": triples,
        "product_isotopies": preserving,
        "non_aligned_permutation_isotopies": nonpermutation,
    }


def is_permutation_linear_map(columns: Sequence[int], dimension: int) -> bool:
    return sorted(columns) == sorted(1 << i for i in range(dimension))


def exhaustive_hadamard_automorphism_control(dimension: int) -> dict[str, int]:
    """Exhaust all binary matrices for n<=3 and count multiplicative bijections."""

    if not 1 <= dimension <= 3:
        raise ValueError("registered exhaustive control supports dimensions 1..3")
    invertible = 0
    preserving = 0
    preserving_nonpermutation = 0
    for columns in product(range(1 << dimension), repeat=dimension):
        if gf2_matrix_rank(columns, dimension) != dimension:
            continue
        invertible += 1
        if preserves_coordinate_product(columns, dimension):
            preserving += 1
            if not is_permutation_linear_map(columns, dimension):
                preserving_nonpermutation += 1
    return {
        "dimension": dimension,
        "invertible_maps": invertible,
        "product_preserving_maps": preserving,
        "product_preserving_nonpermutations": preserving_nonpermutation,
    }


def preserves_boolean_meet_bijection(mapping: Sequence[int], dimension: int) -> bool:
    """Check arbitrary (not necessarily linear) bijection against coordinatewise AND."""

    size = 1 << dimension
    if len(mapping) != size or sorted(mapping) != list(range(size)):
        return False
    for x in range(size):
        for y in range(size):
            if mapping[x & y] != (mapping[x] & mapping[y]):
                return False
    return True


def is_coordinate_permutation_bijection(mapping: Sequence[int], dimension: int) -> bool:
    """Recognize a Boolean-cube coordinate permutation from its complete table."""

    images = [mapping[1 << i] for i in range(dimension)]
    if sorted(images) != sorted(1 << i for i in range(dimension)):
        return False
    for x in range(1 << dimension):
        out = 0
        for i, image in enumerate(images):
            if (x >> i) & 1:
                out |= image
        if mapping[x] != out:
            return False
    return True


def exhaustive_boolean_meet_automorphism_control(dimension: int) -> dict[str, int]:
    """Exhaust arbitrary state bijections for n<=3; only coordinate permutations survive."""

    if not 1 <= dimension <= 3:
        raise ValueError("registered exhaustive meet control supports dimensions 1..3")
    size = 1 << dimension
    preserving = 0
    non_coordinate = 0
    checked = 0
    for mapping in permutations(range(size)):
        checked += 1
        if preserves_boolean_meet_bijection(mapping, dimension):
            preserving += 1
            if not is_coordinate_permutation_bijection(mapping, dimension):
                non_coordinate += 1
    return {
        "dimension": dimension,
        "bijections_checked": checked,
        "meet_preserving_bijections": preserving,
        "meet_preserving_non_coordinate_permutations": non_coordinate,
    }


def gauge_dense_support_invariance(rows: int, columns: int) -> dict[str, object]:
    """Worst-case dense support under safe coordinate permutations."""

    if rows <= 0 or columns <= 0:
        raise ValueError("positive dimensions required")
    dense = rows * columns
    return {
        "rows": rows,
        "columns": columns,
        "dense_nonzeros": dense,
        "nonzeros_after_any_row_column_permutation": dense,
        "source_read_fraction": 1.0,
        "arithmetic_fraction": 1.0,
        "removes_at_least_90_percent": False,
    }


def build_audit() -> dict[str, object]:
    p1 = registered_sparse_image_frame_audit()
    target_block = minimum_equal_block_width_for_target()
    p2_best = value_class_operation_bounds(16_384, 16_384, 1)
    p2_worst = value_class_operation_bounds(16_384, 16_384, 16_384)
    gauge_controls = [exhaustive_hadamard_automorphism_control(n) for n in (1, 2, 3)]
    nonlinear_gauge_controls = [exhaustive_boolean_meet_automorphism_control(n) for n in (1, 2, 3)]
    isotopy_controls = [exhaustive_linear_hadamard_isotopy_control(n) for n in (1, 2)]
    return {
        "classification": "E0_IMPLICIT_NONLINEAR_DIRECT_QUERY_FRONTIER",
        "decision": (
            "REJECT_EXPLICIT_PATTERN_IMAGE_FRAME_AS_NATIVE_CORE;"
            "REJECT_VALUE_CLASS_INVERTED_EXECUTOR_AT_ARITHMETIC_AND_INFORMATION_GATE;"
            "REJECT_SAFE_GAUGE_AS_DENSE_EFFECT_REMOVAL;"
            "KEEP_GENERAL_IMPLICIT_NONLINEAR_DIRECT_QUERY_REPRESENTATION_OPEN"
        ),
        "principle_I_sparse_image_frame": p1,
        "principle_I_general_linear_frame_lower_bounds": registered_one_sided_frame_lower_bounds(),
        "principle_I_equal_block_target_tradeoff": equal_block_pattern_tradeoff(target_block),
        "principle_II_value_class_best_case": p2_best,
        "principle_II_value_class_high_distinctness": p2_worst,
        "principle_II_state_coalescing_fp32_adversary": fp32_exact_integer_state_coalescing_adversary(16_384, 16_384),
        "principle_III_hadamard_automorphism_controls": gauge_controls,
        "principle_III_arbitrary_bijection_meet_controls": nonlinear_gauge_controls,
        "principle_III_independent_linear_gauge_isotopy_controls": isotopy_controls,
        "principle_III_dense_support_adversary": gauge_dense_support_invariance(16_384, 16_384),
        "claim_boundary": {
            "arbitrary_gf2_direct_query_producer_constructed": True,
            "arbitrary_native_fast_producer_constructed": False,
            "value_class_exact_for_declared_ordered_leaf_abi": True,
            "full_hf_backend_abi_proved": False,
            "general_nonlinear_cell_probe_structure_rejected": False,
            "target_405b_executed": False,
            "target_hardware_executed": False,
        },
        "obligations": {
            "O1": "OPEN",
            "O2": "OPEN",
            "O3": "OPEN",
            "O4": "OPEN",
            "O5": "OPEN",
            "O6": "PARTIAL_E0_REPRODUCIBLE",
        },
    }
