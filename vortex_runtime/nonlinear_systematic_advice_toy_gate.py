"""Exact 2x3 screen for one arbitrary nonlinear systematic advice bit.

Six raw matrix bits are stored systematically together with one completely
arbitrary Boolean function of the source.  Every depth-two adaptive decoder is
enumerated symbolically.  For a fixed decoder tree and linear query, equality
at each of the 64 sources either fixes the advice bit there, leaves it free, or
is impossible.  The resulting Boolean cubes are intersected across all
rank-one queries.
"""

from __future__ import annotations


ROWS = 2
COLUMNS = 3
SOURCE_BITS = ROWS * COLUMNS
SOURCE_VALUES = 1 << SOURCE_BITS
ADVICE_CELL = SOURCE_BITS

DECISION = (
    "REJECT_SYSTEMATIC_2X3_PLUS_ONE_ARBITRARY_ADVICE_BIT_AT_TWO_PROBES_"
    "KEEP_FULLY_NONSYSTEMATIC_SEVEN_BIT_ENCODING_OPEN"
)


def rank_one_masks() -> list[int]:
    masks: list[int] = []
    for left in range(1, 1 << ROWS):
        for right in range(1, 1 << COLUMNS):
            mask = 0
            for row in range(ROWS):
                for column in range(COLUMNS):
                    if ((left >> row) & 1) and ((right >> column) & 1):
                        mask |= 1 << (row * COLUMNS + column)
            if mask not in masks:
                masks.append(mask)
    return masks


def _branch_functions() -> list[tuple[str, int, int]]:
    branches = [("constant", 0, 0), ("constant", 1, 0)]
    for cell in range(SOURCE_BITS + 1):
        branches.append(("cell", cell, 0))
        branches.append(("cell", cell, 1))
    return branches


def _cell_value(*, cell: int, source: int, advice: int) -> int:
    if cell == ADVICE_CELL:
        return advice
    return (source >> cell) & 1


def _branch_value(
    branch: tuple[str, int, int], *, source: int, advice: int
) -> int:
    kind, value, negate = branch
    if kind == "constant":
        return value
    return _cell_value(cell=value, source=source, advice=advice) ^ negate


def _tree_value(
    *,
    root: int,
    zero_branch: tuple[str, int, int],
    one_branch: tuple[str, int, int],
    source: int,
    advice: int,
) -> int:
    first = _cell_value(cell=root, source=source, advice=advice)
    return _branch_value(
        one_branch if first else zero_branch,
        source=source,
        advice=advice,
    )


def query_advice_cubes(mask: int) -> list[tuple[int, int]]:
    """Return all distinct ``(must_be_one, must_be_zero)`` cubes."""

    if mask <= 0 or mask >= (1 << SOURCE_BITS):
        raise ValueError("query mask must be a nonzero six-bit value")
    branches = _branch_functions()
    cubes: set[tuple[int, int]] = set()
    for root in range(SOURCE_BITS + 1):
        for zero_branch in branches:
            for one_branch in branches:
                must_be_one = 0
                must_be_zero = 0
                possible = True
                for source in range(SOURCE_VALUES):
                    target = (source & mask).bit_count() & 1
                    answer_zero = _tree_value(
                        root=root,
                        zero_branch=zero_branch,
                        one_branch=one_branch,
                        source=source,
                        advice=0,
                    )
                    answer_one = _tree_value(
                        root=root,
                        zero_branch=zero_branch,
                        one_branch=one_branch,
                        source=source,
                        advice=1,
                    )
                    zero_works = answer_zero == target
                    one_works = answer_one == target
                    if not zero_works and not one_works:
                        possible = False
                        break
                    if zero_works and not one_works:
                        must_be_zero |= 1 << source
                    elif one_works and not zero_works:
                        must_be_one |= 1 << source
                if possible:
                    cubes.add((must_be_one, must_be_zero))
    return sorted(cubes)


def exact_systematic_advice_screen() -> dict[str, object]:
    masks = rank_one_masks()
    cube_lists = [(mask, query_advice_cubes(mask)) for mask in masks]
    cube_lists.sort(key=lambda item: (len(item[1]), item[0]))
    visited = 0
    witness: tuple[int, int] | None = None

    def visit(index: int, ones: int, zeros: int) -> bool:
        nonlocal visited, witness
        visited += 1
        if index == len(cube_lists):
            witness = (ones, zeros)
            return True
        for cube_ones, cube_zeros in cube_lists[index][1]:
            if cube_ones & zeros or cube_zeros & ones:
                continue
            if visit(
                index + 1,
                ones | cube_ones,
                zeros | cube_zeros,
            ):
                return True
        return False

    exists = visit(0, 0, 0)
    hard_counts = {
        f"{mask:06b}": len(cubes)
        for mask, cubes in cube_lists
        if mask.bit_count() > 2
    }
    return {
        "classification": "E0_NONLINEAR_SYSTEMATIC_ADVICE_TOY_GATE",
        "rows": ROWS,
        "columns": COLUMNS,
        "source_bits": SOURCE_BITS,
        "stored_bits": SOURCE_BITS + 1,
        "rank_one_query_count": len(masks),
        "adaptive_probe_limit": 2,
        "raw_source_bits_are_systematic": True,
        "seventh_bit_is_arbitrary_boolean_function": True,
        "all_depth_two_decoder_trees_enumerated": True,
        "hard_query_cube_counts": hard_counts,
        "backtracking_nodes": visited,
        "scheme_exists": exists,
        "advice_constraint_witness": (
            None
            if witness is None
            else {
                "must_be_one": f"{witness[0]:016x}",
                "must_be_zero": f"{witness[1]:016x}",
            }
        ),
        "decision": (
            "FOUND_SYSTEMATIC_NONLINEAR_TWO_PROBE_SEED"
            if exists
            else DECISION
        ),
        "claim_boundary": {
            "arbitrary_systematic_advice_bit_covered": True,
            "arbitrary_depth_two_adaptive_decoder_covered": True,
            "fully_nonsystematic_seven_bit_encoding_covered": False,
            "large_word_encoding_covered": False,
            "surviving_runtime_candidate": False,
        },
    }


def derive_audit() -> dict[str, object]:
    result = exact_systematic_advice_screen()
    if result["scheme_exists"]:
        raise AssertionError("registered systematic nonlinear rejection drifted")
    if set(result["hard_query_cube_counts"].values()) != {14}:
        raise AssertionError("registered symbolic cube count drifted")
    return result
