"""Pure EXP-080A arithmetic model.

This module answers one deliberately narrow question: if a perfect future
activation block already existed, could constructive exact rectangular matrix
multiplication close the VORTEX arithmetic and traffic fractions?  It performs
no I/O and owns no model/runtime state so that the logic can be lifted out of
the throwaway terminal shell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence


class HyperblockError(ValueError):
    pass


@dataclass(frozen=True)
class TensorFamily:
    name: str
    rows: int
    columns: int
    count: int

    def __post_init__(self) -> None:
        if not self.name or min(self.rows, self.columns, self.count) <= 0:
            raise HyperblockError("invalid tensor family")

    @property
    def parameters(self) -> int:
        return self.rows * self.columns * self.count


@dataclass(frozen=True)
class ArithmeticCount:
    multiplications: float
    additions: float

    @property
    def total(self) -> float:
        return self.multiplications + self.additions

    def scaled(self, factor: int) -> "ArithmeticCount":
        return ArithmeticCount(
            multiplications=self.multiplications * factor,
            additions=self.additions * factor,
        )


@dataclass(frozen=True)
class SquarePlan:
    size: int
    leaf_size: int
    count: ArithmeticCount


@dataclass(frozen=True)
class FamilyBlockPlan:
    family: str
    block_length: int
    tile_size: int
    row_tiles: int
    inner_tiles: int
    column_tiles: int
    square_products: int
    leaf_size: int | None
    count: ArithmeticCount
    accumulation_additions: int
    favorable_workspace_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BlockEvaluation:
    block_length: int
    target_sweeps: float
    proposal_billions: float
    classical_operations: float
    constructive_operations: float
    omega_oracle_operations: float
    constructive_ratio: float
    omega_oracle_ratio: float
    traffic_fraction: float
    constructive_total_fraction: float
    omega_oracle_total_fraction: float
    required_constructive_packing_factor: float
    favorable_workspace_bytes: int
    constructive_traffic_pass: bool
    constructive_arithmetic_pass: bool
    workspace_pass: bool
    constructive_joint_pass: bool
    omega_oracle_joint_pass: bool

    def as_dict(self) -> dict:
        return asdict(self)


def target_equivalent_fraction(
    *, baseline_billions: float, target_billions: float, latency_multiple: float
) -> float:
    if min(baseline_billions, target_billions, latency_multiple) <= 0:
        raise HyperblockError("target fraction inputs must be positive")
    return latency_multiple * baseline_billions / target_billions


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise HyperblockError("invalid ceil_div inputs")
    return (value + divisor - 1) // divisor


def largest_power_of_two_at_most(value: int) -> int:
    if value <= 0:
        raise HyperblockError("power-of-two ceiling must be positive")
    return 1 << (value.bit_length() - 1)


def classical_square_count(size: int) -> ArithmeticCount:
    if size <= 0:
        raise HyperblockError("square size must be positive")
    return ArithmeticCount(
        multiplications=size**3,
        additions=size * size * (size - 1),
    )


def strassen_square_count(size: int, leaf_size: int) -> ArithmeticCount:
    if size <= 0 or leaf_size <= 0:
        raise HyperblockError("Strassen sizes must be positive")
    if size & (size - 1) or leaf_size & (leaf_size - 1):
        raise HyperblockError("Strassen sizes must be powers of two")
    if leaf_size > size or size % leaf_size:
        raise HyperblockError("invalid Strassen leaf")
    if size == leaf_size:
        return classical_square_count(size)
    child = strassen_square_count(size // 2, leaf_size)
    return ArithmeticCount(
        multiplications=7 * child.multiplications,
        additions=7 * child.additions + 18 * (size // 2) ** 2,
    )


def best_strassen_square_plan(size: int) -> SquarePlan:
    candidates: list[SquarePlan] = []
    leaf = 1
    while leaf <= size:
        candidates.append(
            SquarePlan(
                size=size,
                leaf_size=leaf,
                count=strassen_square_count(size, leaf),
            )
        )
        leaf *= 2
    return min(candidates, key=lambda row: (row.count.total, row.leaf_size))


def classical_rectangular_count(
    family: TensorFamily, block_length: int
) -> ArithmeticCount:
    if block_length <= 0:
        raise HyperblockError("block length must be positive")
    one = ArithmeticCount(
        multiplications=family.rows * family.columns * block_length,
        additions=family.rows * block_length * (family.columns - 1),
    )
    return one.scaled(family.count)


def _workspace_bytes(
    family: TensorFamily,
    *,
    block_length: int,
    tile_size: int,
    q4_weight_bytes_per_scalar: float,
    activation_bytes_per_scalar: int,
    scratch_bytes_per_scalar: int,
    scratch_square_tile_count: int,
) -> int:
    values = (
        family.columns * block_length * activation_bytes_per_scalar
        + family.rows * block_length * activation_bytes_per_scalar
        + tile_size * tile_size * q4_weight_bytes_per_scalar
        + scratch_square_tile_count
        * tile_size
        * tile_size
        * scratch_bytes_per_scalar
    )
    return math.ceil(values)


def rectangular_strassen_plan(
    family: TensorFamily,
    block_length: int,
    *,
    q4_weight_bytes_per_scalar: float = 0.5,
    activation_bytes_per_scalar: int = 2,
    scratch_bytes_per_scalar: int = 4,
    scratch_square_tile_count: int = 3,
) -> FamilyBlockPlan:
    tile = largest_power_of_two_at_most(
        min(family.rows, family.columns, block_length)
    )
    row_tiles = ceil_div(family.rows, tile)
    inner_tiles = ceil_div(family.columns, tile)
    column_tiles = ceil_div(block_length, tile)
    square_products = row_tiles * inner_tiles * column_tiles
    square = best_strassen_square_plan(tile)
    accumulation = (
        row_tiles * column_tiles * max(0, inner_tiles - 1) * tile * tile
    )
    count = ArithmeticCount(
        multiplications=square_products * square.count.multiplications,
        additions=square_products * square.count.additions + accumulation,
    ).scaled(family.count)
    return FamilyBlockPlan(
        family=family.name,
        block_length=block_length,
        tile_size=tile,
        row_tiles=row_tiles,
        inner_tiles=inner_tiles,
        column_tiles=column_tiles,
        square_products=square_products,
        leaf_size=square.leaf_size,
        count=count,
        accumulation_additions=accumulation * family.count,
        favorable_workspace_bytes=_workspace_bytes(
            family,
            block_length=block_length,
            tile_size=tile,
            q4_weight_bytes_per_scalar=q4_weight_bytes_per_scalar,
            activation_bytes_per_scalar=activation_bytes_per_scalar,
            scratch_bytes_per_scalar=scratch_bytes_per_scalar,
            scratch_square_tile_count=scratch_square_tile_count,
        ),
    )


def rectangular_omega_oracle_plan(
    family: TensorFamily,
    block_length: int,
    *,
    omega: float,
    q4_weight_bytes_per_scalar: float = 0.5,
    activation_bytes_per_scalar: int = 2,
    scratch_bytes_per_scalar: int = 4,
    scratch_square_tile_count: int = 3,
) -> FamilyBlockPlan:
    if not 2.0 <= omega < 3.0:
        raise HyperblockError("omega must be in [2, 3)")
    tile = largest_power_of_two_at_most(
        min(family.rows, family.columns, block_length)
    )
    row_tiles = ceil_div(family.rows, tile)
    inner_tiles = ceil_div(family.columns, tile)
    column_tiles = ceil_div(block_length, tile)
    square_products = row_tiles * inner_tiles * column_tiles
    accumulation = (
        row_tiles * column_tiles * max(0, inner_tiles - 1) * tile * tile
    )
    count = ArithmeticCount(
        multiplications=square_products * (tile**omega),
        additions=accumulation,
    ).scaled(family.count)
    return FamilyBlockPlan(
        family=family.name,
        block_length=block_length,
        tile_size=tile,
        row_tiles=row_tiles,
        inner_tiles=inner_tiles,
        column_tiles=column_tiles,
        square_products=square_products,
        leaf_size=None,
        count=count,
        accumulation_additions=accumulation * family.count,
        favorable_workspace_bytes=_workspace_bytes(
            family,
            block_length=block_length,
            tile_size=tile,
            q4_weight_bytes_per_scalar=q4_weight_bytes_per_scalar,
            activation_bytes_per_scalar=activation_bytes_per_scalar,
            scratch_bytes_per_scalar=scratch_bytes_per_scalar,
            scratch_square_tile_count=scratch_square_tile_count,
        ),
    )


def aggregate_count(plans: Iterable[FamilyBlockPlan]) -> ArithmeticCount:
    multiplications = 0.0
    additions = 0.0
    for plan in plans:
        multiplications += plan.count.multiplications
        additions += plan.count.additions
    return ArithmeticCount(multiplications=multiplications, additions=additions)


def evaluate_block(
    families: Sequence[TensorFamily],
    *,
    block_length: int,
    target_sweeps: float,
    proposal_billions: float,
    target_billions: float,
    allowed_fraction: float,
    peak_vram_bytes: int,
    omega: float,
    q4_weight_bytes_per_scalar: float = 0.5,
    activation_bytes_per_scalar: int = 2,
    scratch_bytes_per_scalar: int = 4,
    scratch_square_tile_count: int = 3,
) -> tuple[BlockEvaluation, tuple[FamilyBlockPlan, ...], tuple[FamilyBlockPlan, ...]]:
    if not families or block_length <= 0 or target_sweeps <= 0:
        raise HyperblockError("invalid block evaluation")
    if proposal_billions < 0 or target_billions <= 0 or allowed_fraction <= 0:
        raise HyperblockError("invalid cost fraction")
    constructive_plans = tuple(
        rectangular_strassen_plan(
            family,
            block_length,
            q4_weight_bytes_per_scalar=q4_weight_bytes_per_scalar,
            activation_bytes_per_scalar=activation_bytes_per_scalar,
            scratch_bytes_per_scalar=scratch_bytes_per_scalar,
            scratch_square_tile_count=scratch_square_tile_count,
        )
        for family in families
    )
    omega_plans = tuple(
        rectangular_omega_oracle_plan(
            family,
            block_length,
            omega=omega,
            q4_weight_bytes_per_scalar=q4_weight_bytes_per_scalar,
            activation_bytes_per_scalar=activation_bytes_per_scalar,
            scratch_bytes_per_scalar=scratch_bytes_per_scalar,
            scratch_square_tile_count=scratch_square_tile_count,
        )
        for family in families
    )
    classical = ArithmeticCount(0.0, 0.0)
    for family in families:
        row = classical_rectangular_count(family, block_length)
        classical = ArithmeticCount(
            classical.multiplications + row.multiplications,
            classical.additions + row.additions,
        )
    constructive = aggregate_count(constructive_plans)
    omega_count = aggregate_count(omega_plans)
    constructive_ratio = constructive.total / classical.total
    omega_ratio = omega_count.total / classical.total
    proposal_fraction = proposal_billions / target_billions
    traffic_fraction = proposal_fraction + target_sweeps / block_length
    constructive_total = proposal_fraction + target_sweeps * constructive_ratio
    omega_total = proposal_fraction + target_sweeps * omega_ratio
    remaining = allowed_fraction - proposal_fraction
    required_packing = (
        math.inf
        if remaining <= 0
        else target_sweeps * constructive_ratio / remaining
    )
    workspace = max(plan.favorable_workspace_bytes for plan in constructive_plans)
    traffic_pass = traffic_fraction <= allowed_fraction
    arithmetic_pass = constructive_total <= allowed_fraction
    workspace_pass = workspace <= peak_vram_bytes
    evaluation = BlockEvaluation(
        block_length=block_length,
        target_sweeps=target_sweeps,
        proposal_billions=proposal_billions,
        classical_operations=classical.total,
        constructive_operations=constructive.total,
        omega_oracle_operations=omega_count.total,
        constructive_ratio=constructive_ratio,
        omega_oracle_ratio=omega_ratio,
        traffic_fraction=traffic_fraction,
        constructive_total_fraction=constructive_total,
        omega_oracle_total_fraction=omega_total,
        required_constructive_packing_factor=required_packing,
        favorable_workspace_bytes=workspace,
        constructive_traffic_pass=traffic_pass,
        constructive_arithmetic_pass=arithmetic_pass,
        workspace_pass=workspace_pass,
        constructive_joint_pass=traffic_pass and arithmetic_pass and workspace_pass,
        omega_oracle_joint_pass=(
            traffic_pass and omega_total <= allowed_fraction and workspace_pass
        ),
    )
    return evaluation, constructive_plans, omega_plans


Matrix = list[list[int]]


def _validate_square_pair(left: Matrix, right: Matrix) -> int:
    size = len(left)
    if size == 0 or len(right) != size:
        raise HyperblockError("matrix sizes differ")
    if any(len(row) != size for row in left + right):
        raise HyperblockError("matrices must be square")
    if size & (size - 1):
        raise HyperblockError("matrix size must be a power of two")
    return size


def naive_matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    size = _validate_square_pair(left, right)
    return [
        [sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size)]
        for i in range(size)
    ]


def _matrix_add(left: Matrix, right: Matrix, sign: int = 1) -> Matrix:
    return [
        [a + sign * b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _split(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    half = len(matrix) // 2
    return (
        [row[:half] for row in matrix[:half]],
        [row[half:] for row in matrix[:half]],
        [row[:half] for row in matrix[half:]],
        [row[half:] for row in matrix[half:]],
    )


def _join(c11: Matrix, c12: Matrix, c21: Matrix, c22: Matrix) -> Matrix:
    return [
        c11[index] + c12[index] for index in range(len(c11))
    ] + [c21[index] + c22[index] for index in range(len(c21))]


def strassen_matrix_multiply(
    left: Matrix, right: Matrix, *, leaf_size: int = 1
) -> Matrix:
    size = _validate_square_pair(left, right)
    if leaf_size <= 0 or leaf_size & (leaf_size - 1) or leaf_size > size:
        raise HyperblockError("invalid exact-control leaf size")
    if size <= leaf_size:
        return naive_matrix_multiply(left, right)
    a11, a12, a21, a22 = _split(left)
    b11, b12, b21, b22 = _split(right)
    multiply = lambda a, b: strassen_matrix_multiply(  # noqa: E731
        a, b, leaf_size=leaf_size
    )
    m1 = multiply(_matrix_add(a11, a22), _matrix_add(b11, b22))
    m2 = multiply(_matrix_add(a21, a22), b11)
    m3 = multiply(a11, _matrix_add(b12, b22, -1))
    m4 = multiply(a22, _matrix_add(b21, b11, -1))
    m5 = multiply(_matrix_add(a11, a12), b22)
    m6 = multiply(_matrix_add(a21, a11, -1), _matrix_add(b11, b12))
    m7 = multiply(_matrix_add(a12, a22, -1), _matrix_add(b21, b22))
    c11 = _matrix_add(_matrix_add(_matrix_add(m1, m4), m5, -1), m7)
    c12 = _matrix_add(m3, m5)
    c21 = _matrix_add(m2, m4)
    c22 = _matrix_add(_matrix_add(_matrix_add(m1, m2, -1), m3), m6)
    return _join(c11, c12, c21, c22)
