"""Constructive E0 gates for implicit checkpoint-program carriers.

Three preregistered principles are implemented here:

* an address-only alias router whose checkpoint bits live in a routing relation;
* a query-time Patricia pattern synthesizer over exact GF(2) row blocks;
* rank-normal encoded linear state with an explicitly paid transformed Hadamard.

The module deliberately keeps the favorable GF(2) controls separate from the
unproved arbitrary native Q4/BF16/FP32 lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from vortex_runtime.implicit_direct_query_gate import (
    registered_non_embedding_specs,
    ten_eleven_partition,
)


WORD_BITS = 64
GIB = 1 << 30


def _validate_binary_matrix(matrix: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("nonempty binary matrix required")
    rows = len(matrix)
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("rectangular matrix required")
    if any(bit not in (0, 1) for row in matrix for bit in row):
        raise ValueError("matrix must be binary")
    return rows, columns


def _matrix_row_masks(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    rows, columns = _validate_binary_matrix(matrix)
    del rows
    result = []
    for row in matrix:
        mask = 0
        for j in range(columns):
            mask |= int(row[j]) << j
        result.append(mask)
    return tuple(result)


def direct_gf2_matvec_mask(matrix: Sequence[Sequence[int]], vector_mask: int) -> int:
    row_masks = _matrix_row_masks(matrix)
    columns = len(matrix[0])
    if vector_mask < 0 or vector_mask >= (1 << columns):
        raise ValueError("query outside matrix width")
    out = 0
    for i, row_mask in enumerate(row_masks):
        out |= ((row_mask & vector_mask).bit_count() & 1) << i
    return out


# ---------------------------------------------------------------------------
# Principle I: address-only alias router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasRouter:
    rows: int
    columns: int
    block_widths: tuple[int, ...]
    # aliases[row][block] is the local query-table target pattern.  Runtime's
    # logical source address is only (row, block); this target is compiled state.
    aliases: tuple[tuple[int, ...], ...]


def compile_alias_router(
    matrix: Sequence[Sequence[int]], block_widths: Sequence[int]
) -> AliasRouter:
    """Compile the checkpoint solely into a row/block -> local-pattern relation."""

    rows, columns = _validate_binary_matrix(matrix)
    widths = tuple(int(w) for w in block_widths)
    if not widths or any(w <= 0 for w in widths) or sum(widths) != columns:
        raise ValueError("block widths must be positive and sum to matrix width")

    aliases: list[tuple[int, ...]] = []
    for row in matrix:
        offset = 0
        row_aliases = []
        for width in widths:
            pattern = 0
            for local in range(width):
                pattern |= int(row[offset + local]) << local
            row_aliases.append(pattern)
            offset += width
        aliases.append(tuple(row_aliases))
    return AliasRouter(rows, columns, widths, tuple(aliases))


def _query_subset_parity_table(vector_mask: int, offset: int, width: int) -> tuple[int, ...]:
    table = [0] * (1 << width)
    for pattern in range(1, 1 << width):
        lsb = pattern & -pattern
        local_bit = lsb.bit_length() - 1
        table[pattern] = table[pattern ^ lsb] ^ ((vector_mask >> (offset + local_bit)) & 1)
    return tuple(table)


def query_alias_router(router: AliasRouter, vector_mask: int) -> tuple[int, dict[str, int]]:
    """Evaluate exact GF(2) MatVec through fixed logical row/block alias reads."""

    if vector_mask < 0 or vector_mask >= (1 << router.columns):
        raise ValueError("query outside router width")
    tables = []
    offset = 0
    table_ops = 0
    for width in router.block_widths:
        tables.append(_query_subset_parity_table(vector_mask, offset, width))
        table_ops += (1 << width) - 1
        offset += width

    out = 0
    alias_reads = 0
    row_xors = 0
    for i, row_aliases in enumerate(router.aliases):
        value = 0
        for block_id, pattern in enumerate(row_aliases):
            # The logical request is (i, block_id).  The compiled relation
            # resolves it to ``pattern``; physical translation cost is audited
            # separately rather than being called free.
            value ^= tables[block_id][pattern]
            alias_reads += 1
            row_xors += 1
        out |= value << i
    return out, {
        "query_table_ops": table_ops,
        "logical_alias_reads": alias_reads,
        "row_xors": row_xors,
    }


def alias_router_square_audit(rows: int = 16_384, columns: int = 16_384) -> dict[str, object]:
    if rows <= 0 or columns <= 0:
        raise ValueError("positive shape required")
    widths = ten_eleven_partition(columns)
    blocks = len(widths)
    table_ops = sum((1 << width) - 1 for width in widths)
    aliases = rows * blocks
    dense_coefficients = rows * columns
    baseline_slots = 2 * dense_coefficients
    # Count one row reduction per alias in the favorable arithmetic screen.
    candidate_ops = table_ops + aliases
    information_lower_bits = dense_coefficients
    descriptor_bits_64 = aliases * WORD_BITS
    return {
        "rows": rows,
        "columns": columns,
        "blocks": blocks,
        "query_table_ops": table_ops,
        "logical_alias_reads": aliases,
        "candidate_table_plus_row_xor_ops": candidate_ops,
        "baseline_leaf_plus_add_slots": baseline_slots,
        "candidate_operation_fraction": candidate_ops / baseline_slots,
        "removes_at_least_90_percent_favorable_arithmetic": candidate_ops <= baseline_slots // 10,
        "alias_information_lower_bits": information_lower_bits,
        "alias_information_lower_over_source": 1.0,
        "descriptor_bits_if_64_per_alias": descriptor_bits_64,
        "descriptor_gib_if_64_per_alias": descriptor_bits_64 / 8 / GIB,
        "descriptor_over_binary_source_if_64_per_alias": descriptor_bits_64 / dense_coefficients,
        "native_fast_path_proved": False,
    }


def registered_alias_router_audit() -> dict[str, object]:
    source_bits = 0
    aliases = 0
    table_ops = 0
    matrix_count = 0
    for spec in registered_non_embedding_specs():
        widths = ten_eleven_partition(spec.columns)
        source_bits += spec.count * spec.rows * spec.columns
        aliases += spec.count * spec.rows * len(widths)
        table_ops += spec.count * sum((1 << width) - 1 for width in widths)
        matrix_count += spec.count
    descriptor_bits = aliases * WORD_BITS
    return {
        "matrix_count": matrix_count,
        "source_bits": source_bits,
        "source_gib": source_bits / 8 / GIB,
        "alias_count": aliases,
        "query_table_ops": table_ops,
        "information_lower_bits": source_bits,
        "descriptor_bits_if_64_per_alias": descriptor_bits,
        "descriptor_gib_if_64_per_alias": descriptor_bits / 8 / GIB,
        "descriptor_over_source_if_64_per_alias": descriptor_bits / source_bits,
        "fits_8gib_if_all_descriptors_hot": descriptor_bits <= 8 * GIB * 8,
        "claim_boundary": "EXPLICIT_64_BIT_ALIAS_DESCRIPTOR_REALIZATION_ONLY_FOR_RUNTIME_TRAFFIC",
    }


# ---------------------------------------------------------------------------
# Principle II: query-time Patricia pattern synthesizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatriciaNode:
    start: int
    common_end: int
    common_one_mask: int
    branch_bit: int | None
    zero_child: "PatriciaNode | None"
    one_child: "PatriciaNode | None"
    rows: tuple[int, ...]


@dataclass(frozen=True)
class PatriciaBlock:
    width: int
    root: PatriciaNode


@dataclass(frozen=True)
class PatriciaProgram:
    rows: int
    columns: int
    block_widths: tuple[int, ...]
    blocks: tuple[PatriciaBlock, ...]


def _build_patricia_node(pattern_rows: dict[int, tuple[int, ...]], width: int, start: int) -> PatriciaNode:
    patterns = tuple(sorted(pattern_rows))
    if not patterns:
        raise ValueError("empty Patricia subset")

    if len(patterns) == 1:
        pattern = patterns[0]
        segment_mask = pattern & ~((1 << start) - 1)
        segment_mask &= (1 << width) - 1
        return PatriciaNode(
            start=start,
            common_end=width,
            common_one_mask=segment_mask,
            branch_bit=None,
            zero_child=None,
            one_child=None,
            rows=tuple(sorted(pattern_rows[pattern])),
        )

    branch = None
    for bit in range(start, width):
        values = {(pattern >> bit) & 1 for pattern in patterns}
        if len(values) == 2:
            branch = bit
            break
    if branch is None:
        raise AssertionError("distinct patterns must diverge before width")

    first = patterns[0]
    prefix_width = branch - start
    if prefix_width:
        prefix_mask = ((1 << prefix_width) - 1) << start
        common_one_mask = first & prefix_mask
    else:
        common_one_mask = 0

    zero: dict[int, tuple[int, ...]] = {}
    one: dict[int, tuple[int, ...]] = {}
    for pattern in patterns:
        if (pattern >> branch) & 1:
            one[pattern] = pattern_rows[pattern]
        else:
            zero[pattern] = pattern_rows[pattern]
    return PatriciaNode(
        start=start,
        common_end=branch,
        common_one_mask=common_one_mask,
        branch_bit=branch,
        zero_child=_build_patricia_node(zero, width, branch + 1),
        one_child=_build_patricia_node(one, width, branch + 1),
        rows=(),
    )


def compile_patricia_program(
    matrix: Sequence[Sequence[int]], block_width: int = 64
) -> PatriciaProgram:
    rows, columns = _validate_binary_matrix(matrix)
    if block_width <= 0 or block_width > 64:
        raise ValueError("registered Patricia implementation requires 1..64 bits")
    widths = []
    remaining = columns
    while remaining:
        width = min(block_width, remaining)
        widths.append(width)
        remaining -= width

    blocks = []
    offset = 0
    for width in widths:
        groups: dict[int, list[int]] = {}
        for i, row in enumerate(matrix):
            pattern = 0
            for local in range(width):
                pattern |= int(row[offset + local]) << local
            groups.setdefault(pattern, []).append(i)
        frozen_groups = {pattern: tuple(indices) for pattern, indices in groups.items()}
        blocks.append(PatriciaBlock(width, _build_patricia_node(frozen_groups, width, 0)))
        offset += width
    return PatriciaProgram(rows, columns, tuple(widths), tuple(blocks))


def _walk_patricia(
    node: PatriciaNode,
    query_word: int,
    incoming: int,
    output: list[int],
    stats: dict[str, int],
) -> None:
    stats["node_visits"] += 1
    stats["label_word_reads"] += 1
    value = incoming ^ ((node.common_one_mask & query_word).bit_count() & 1)
    stats["parity_events"] += 1
    if node.branch_bit is None:
        for row in node.rows:
            output[row] ^= value
            stats["leaf_memberships"] += 1
        return
    if node.zero_child is None or node.one_child is None:
        raise AssertionError("internal Patricia node missing child")
    _walk_patricia(node.zero_child, query_word, value, output, stats)
    one_value = value ^ ((query_word >> node.branch_bit) & 1)
    stats["branch_one_xors"] += 1
    _walk_patricia(node.one_child, query_word, one_value, output, stats)


def query_patricia_program(program: PatriciaProgram, vector_mask: int) -> tuple[int, dict[str, int]]:
    if vector_mask < 0 or vector_mask >= (1 << program.columns):
        raise ValueError("query outside program width")
    row_values = [0] * program.rows
    stats = {
        "node_visits": 0,
        "label_word_reads": 0,
        "parity_events": 0,
        "branch_one_xors": 0,
        "leaf_memberships": 0,
    }
    offset = 0
    for block in program.blocks:
        query_word = (vector_mask >> offset) & ((1 << block.width) - 1)
        _walk_patricia(block.root, query_word, 0, row_values, stats)
        offset += block.width
    out = 0
    for i, value in enumerate(row_values):
        out |= (value & 1) << i
    return out, stats


def patricia_square_audit(rows: int = 16_384, columns: int = 16_384, block_width: int = 64) -> dict[str, object]:
    if rows <= 0 or columns <= 0 or block_width <= 0 or columns % block_width:
        raise ValueError("positive shape with exactly dividing block width required")
    blocks = columns // block_width
    max_edges_per_block = 2 * rows - 2
    max_edge_evaluations = blocks * max_edges_per_block
    leaf_contributions = rows * blocks
    favorable_events = max_edge_evaluations + leaf_contributions
    baseline_slots = 2 * rows * columns
    edge_label_bits = max_edge_evaluations * WORD_BITS
    source_bits = rows * columns
    return {
        "rows": rows,
        "columns": columns,
        "block_width": block_width,
        "blocks": blocks,
        "max_edges_per_block": max_edges_per_block,
        "max_edge_evaluations": max_edge_evaluations,
        "leaf_contributions": leaf_contributions,
        "favorable_edge_plus_leaf_events": favorable_events,
        "baseline_leaf_plus_add_slots": baseline_slots,
        "favorable_event_fraction": favorable_events / baseline_slots,
        "removes_at_least_90_percent_favorable_arithmetic": favorable_events <= baseline_slots // 10,
        "edge_label_bits_one_64bit_word_each": edge_label_bits,
        "edge_label_mib_one_64bit_word_each": edge_label_bits / 8 / (1 << 20),
        "edge_label_over_binary_source": edge_label_bits / source_bits,
        "topology_and_membership_bits_included": False,
        "native_fast_path_proved": False,
    }


# ---------------------------------------------------------------------------
# Principle III: rank-normal encoded state with paid nonlinearity
# ---------------------------------------------------------------------------


def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def gf2_matmul(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> list[list[int]]:
    if not left or not right or not right[0]:
        raise ValueError("nonempty matrices required")
    l_rows = len(left)
    inner = len(left[0])
    if any(len(row) != inner for row in left) or len(right) != inner:
        raise ValueError("matrix multiply shape mismatch")
    r_cols = len(right[0])
    if any(len(row) != r_cols for row in right):
        raise ValueError("rectangular right matrix required")
    out = [[0] * r_cols for _ in range(l_rows)]
    for i in range(l_rows):
        for k in range(inner):
            if int(left[i][k]) & 1:
                for j in range(r_cols):
                    out[i][j] ^= int(right[k][j]) & 1
    return out


def gf2_matvec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError("matvec shape mismatch")
    return [sum((int(a) & int(b)) for a, b in zip(row, vector, strict=True)) & 1 for row in matrix]


def gf2_inverse(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    n, columns = _validate_binary_matrix(matrix)
    if n != columns:
        raise ValueError("inverse requires square matrix")
    left = [list(map(int, row)) for row in matrix]
    right = _identity(n)
    for col in range(n):
        pivot = next((r for r in range(col, n) if left[r][col]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        left[col], left[pivot] = left[pivot], left[col]
        right[col], right[pivot] = right[pivot], right[col]
        for r in range(n):
            if r != col and left[r][col]:
                left[r] = [a ^ b for a, b in zip(left[r], left[col], strict=True)]
                right[r] = [a ^ b for a, b in zip(right[r], right[col], strict=True)]
    return right


@dataclass(frozen=True)
class RankNormalGauge:
    rows: int
    columns: int
    rank: int
    left: tuple[tuple[int, ...], ...]
    right: tuple[tuple[int, ...], ...]
    normal: tuple[tuple[int, ...], ...]


def compile_rank_normal_gauge(matrix: Sequence[Sequence[int]]) -> RankNormalGauge:
    rows, columns = _validate_binary_matrix(matrix)
    work = [list(map(int, row)) for row in matrix]
    left = _identity(rows)
    right = _identity(columns)
    rank = 0
    limit = min(rows, columns)
    while rank < limit:
        pivot = None
        for i in range(rank, rows):
            for j in range(rank, columns):
                if work[i][j]:
                    pivot = (i, j)
                    break
            if pivot is not None:
                break
        if pivot is None:
            break
        pi, pj = pivot
        if pi != rank:
            work[rank], work[pi] = work[pi], work[rank]
            left[rank], left[pi] = left[pi], left[rank]
        if pj != rank:
            for row in work:
                row[rank], row[pj] = row[pj], row[rank]
            for row in right:
                row[rank], row[pj] = row[pj], row[rank]

        for i in range(rows):
            if i != rank and work[i][rank]:
                work[i] = [a ^ b for a, b in zip(work[i], work[rank], strict=True)]
                left[i] = [a ^ b for a, b in zip(left[i], left[rank], strict=True)]
        for j in range(columns):
            if j != rank and work[rank][j]:
                for i in range(rows):
                    work[i][j] ^= work[i][rank]
                for i in range(columns):
                    right[i][j] ^= right[i][rank]
        rank += 1

    return RankNormalGauge(
        rows=rows,
        columns=columns,
        rank=rank,
        left=tuple(tuple(row) for row in left),
        right=tuple(tuple(row) for row in right),
        normal=tuple(tuple(row) for row in work),
    )


def verify_rank_normal_gauge(matrix: Sequence[Sequence[int]], gauge: RankNormalGauge) -> bool:
    computed = gf2_matmul(gf2_matmul(gauge.left, matrix), gauge.right)
    return computed == [list(row) for row in gauge.normal]


def query_rank_normal_gauge(
    matrix: Sequence[Sequence[int]], gauge: RankNormalGauge, vector: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Return (encoded_output, direct_encoded_output) for an exact control.

    x = B z with z=B^-1 x.  Then A W x = (A W B) z = J z.
    """

    right_inverse = gf2_inverse(gauge.right)
    encoded_input = gf2_matvec(right_inverse, vector)
    encoded_from_normal = gf2_matvec(gauge.normal, encoded_input)
    direct = gf2_matvec(matrix, vector)
    direct_encoded = gf2_matvec(gauge.left, direct)
    return encoded_from_normal, direct_encoded


def literal_transformed_hadamard(
    w1: Sequence[Sequence[int]],
    w2: Sequence[Sequence[int]],
    z1: Sequence[int],
    z2: Sequence[int],
) -> list[int]:
    """Paid transformed product for gauges z1=W1^-1 a, z2=W2^-1 b."""

    a = gf2_matvec(w1, z1)
    b = gf2_matvec(w2, z2)
    return [x & y for x, y in zip(a, b, strict=True)]


def rank_normal_hadamard_cost_audit(n: int = 16_384) -> dict[str, object]:
    if n <= 0:
        raise ValueError("positive dimension required")
    baseline_two_dense_slots = 4 * n * n
    baseline_and = n
    baseline = baseline_two_dense_slots + baseline_and
    encoded_projection_copies = 2 * n
    transformed_nonlinearity_dense_slots = 4 * n * n
    transformed_and = n
    candidate = encoded_projection_copies + transformed_nonlinearity_dense_slots + transformed_and
    return {
        "dimension": n,
        "isolated_encoded_projection_effects": 2 * n,
        "isolated_original_projection_coefficients": 2 * n * n,
        "isolated_linear_effect_fraction": Fraction(1, n).__str__(),
        "isolated_removes_at_least_90_percent": n >= 10,
        "baseline_two_dense_plus_and_slots": baseline,
        "literal_transformed_hadamard_slots": candidate,
        "whole_micrograph_candidate_fraction": candidate / baseline,
        "whole_micrograph_removes_at_least_90_percent": candidate <= baseline // 10,
        "dense_maps_restored_inside_transformed_nonlinearity": 2,
        "claim_boundary": "RANK_NORMAL_PER_OPERATOR_PLUS_LITERAL_FACTORIZED_TRANSFORMED_HADAMARD",
    }


def build_audit() -> dict[str, object]:
    return {
        "classification": "E0_IMPLICIT_PROGRAM_CARRIER_FRONTIER",
        "decision": (
            "REJECT_ADDRESS_ONLY_ALIAS_ROUTER_IF_ROUTING_METADATA_IS_PAID;"
            "REJECT_PATRICIA_PATTERN_SYNTHESIZER_IF_EDGE_PROGRAM_TRAFFIC_IS_PAID;"
            "REJECT_PER_OPERATOR_RANK_NORMAL_GAUGE_WITH_LITERAL_TRANSFORMED_HADAMARD;"
            "KEEP_GENERAL_NONLINEAR_IMPLICIT_PROGRAM_SOURCE_OPEN"
        ),
        "principle_I_square": alias_router_square_audit(),
        "principle_I_registered": registered_alias_router_audit(),
        "principle_II_square": patricia_square_audit(),
        "principle_III_rank_normal_hadamard": rank_normal_hadamard_cost_audit(),
        "claim_boundary": {
            "arbitrary_gf2_alias_router_constructed": True,
            "arbitrary_gf2_patricia_program_constructed": True,
            "arbitrary_gf2_rank_normal_compiler_constructed": True,
            "arbitrary_native_fast_producer_constructed": False,
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
