"""Exact multiplication-count oracle for EXP-101A.

The module compares ordinary integral matrix-multiplication tensors with the
published structured 6x6x6 direct-sum relation

    <6,6,6> <= 137 <1> + 8 <1,1,2>

under recursive composition.  It deliberately grants every transform,
addition, movement, representation, and repair for free.  Therefore a failure
to cross the 10x multiplication Gate is decisive for this frozen tensor source,
while a pass only authorizes a later fully charged circuit Gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence


class StructuredDirectSumError(RuntimeError):
    pass


def ceil_div(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        raise StructuredDirectSumError("dimensions and divisors must be positive")
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, order=True)
class UniformScheme:
    a: int
    b: int
    c: int
    rank: int
    source: str

    @property
    def classical_rank(self) -> int:
        return self.a * self.b * self.c

    @property
    def rank_ratio(self) -> float:
        return self.rank / self.classical_rank

    def manifest(self) -> dict[str, Any]:
        return {**asdict(self), "classical_rank": self.classical_rank, "rank_ratio": self.rank_ratio}


@dataclass(frozen=True)
class CostWitness:
    multiplications: int
    action: str
    children: tuple[tuple[int, int, int, int], ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "multiplications": self.multiplications,
            "action": self.action,
            "children": [list(row) for row in self.children],
        }


def catalog_schemes(rows: Sequence[Mapping[str, Any]]) -> tuple[UniformScheme, ...]:
    """Build all cyclic/reflection orientations and safely remove dominated rows."""

    unique: dict[tuple[int, int, int], UniformScheme] = {}
    for row in rows:
        if row.get("status") != "eligible":
            continue
        shape = tuple(int(value) for value in row["shape"])
        rank = int(row["rank"])
        source = str(row["key"])
        for oriented in sorted(set(permutations(shape, 3))):
            key = tuple(oriented)
            candidate = UniformScheme(*key, rank=rank, source=source)
            incumbent = unique.get(key)
            if incumbent is None or (candidate.rank, candidate.source) < (
                incumbent.rank,
                incumbent.source,
            ):
                unique[key] = candidate

    fast = [row for row in unique.values() if row.rank < row.classical_rank]
    # Safe dominance: larger split factors and no larger rank always produce no
    # larger recursive children under ceil-div, for every positive parent shape.
    survivors: list[UniformScheme] = []
    for candidate in fast:
        dominated = False
        for other in fast:
            if other == candidate:
                continue
            if (
                other.a >= candidate.a
                and other.b >= candidate.b
                and other.c >= candidate.c
                and other.rank <= candidate.rank
                and (
                    other.a > candidate.a
                    or other.b > candidate.b
                    or other.c > candidate.c
                    or other.rank < candidate.rank
                )
            ):
                dominated = True
                break
        if not dominated:
            survivors.append(candidate)
    return tuple(sorted(survivors, key=lambda row: (row.rank_ratio, row.rank, row.a, row.b, row.c, row.source)))


def triple_cyclic_rank(unit_multiplicity: int, double_multiplicity: int, two_by_two_rank: int) -> int:
    u = int(unit_multiplicity)
    d = int(double_multiplicity)
    r2 = int(two_by_two_rank)
    # Pick zero, one, two, or three doubled cyclic children.  Three doubled
    # axes leave an ordinary 2x2x2 multiplication, hence rank r2.
    return u**3 + 3 * u**2 * d * 2 + 3 * u * d**2 * 4 + d**3 * r2


class MultiplicationOracle:
    def __init__(
        self,
        schemes: Sequence[UniformScheme],
        *,
        maximum_depth: int,
        structured_enabled: bool,
        unit_multiplicity: int = 137,
        double_multiplicity: int = 8,
        state_limit: int = 2_000_000,
    ) -> None:
        if maximum_depth < 0:
            raise StructuredDirectSumError("maximum_depth must be nonnegative")
        self.schemes = tuple(schemes)
        self.maximum_depth = int(maximum_depth)
        self.structured_enabled = bool(structured_enabled)
        self.unit_multiplicity = int(unit_multiplicity)
        self.double_multiplicity = int(double_multiplicity)
        self.state_limit = int(state_limit)
        self.state_count = 0
        self._witness: dict[tuple[int, int, int, int], CostWitness] = {}

        @lru_cache(maxsize=None)
        def solve(m: int, k: int, n: int, depth: int) -> int:
            self.state_count += 1
            if self.state_count > self.state_limit:
                raise StructuredDirectSumError(
                    f"state limit exceeded: {self.state_count}>{self.state_limit}"
                )
            classical = m * k * n
            best = classical
            witness = CostWitness(classical, "classical", ())
            if depth <= 0 or classical == 1:
                self._witness[(m, k, n, depth)] = witness
                return best

            for scheme in self.schemes:
                child = (
                    ceil_div(m, scheme.a),
                    ceil_div(k, scheme.b),
                    ceil_div(n, scheme.c),
                )
                if child == (m, k, n):
                    continue
                value = scheme.rank * solve(*child, depth - 1)
                if value < best:
                    best = value
                    witness = CostWitness(
                        value,
                        f"uniform:{scheme.source}:{scheme.a},{scheme.b},{scheme.c}:r{scheme.rank}",
                        ((child[0], child[1], child[2], scheme.rank),),
                    )

            if self.structured_enabled:
                base = (ceil_div(m, 6), ceil_div(k, 6), ceil_div(n, 6))
                if base != (m, k, n):
                    for axis, name in enumerate(("m", "k", "n")):
                        doubled = list(base)
                        doubled[axis] *= 2
                        doubled_t = tuple(doubled)
                        value = (
                            self.unit_multiplicity * solve(*base, depth - 1)
                            + self.double_multiplicity
                            * solve(*doubled_t, depth - 1)
                        )
                        if value < best:
                            best = value
                            witness = CostWitness(
                                value,
                                f"structured666:double_{name}",
                                (
                                    (base[0], base[1], base[2], self.unit_multiplicity),
                                    (
                                        doubled_t[0],
                                        doubled_t[1],
                                        doubled_t[2],
                                        self.double_multiplicity,
                                    ),
                                ),
                            )
            self._witness[(m, k, n, depth)] = witness
            return best

        self._solve = solve

    def cost(self, m: int, k: int, n: int) -> int:
        return self._solve(int(m), int(k), int(n), self.maximum_depth)

    def root_witness(self, m: int, k: int, n: int) -> CostWitness:
        self.cost(m, k, n)
        return self._witness[(int(m), int(k), int(n), self.maximum_depth)]

    def cache_info(self) -> dict[str, int]:
        info = self._solve.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "current_size": info.currsize,
            "state_count": self.state_count,
        }


def weighted_ratio(
    families: Iterable[tuple[int, int, int, int]],
    oracle: MultiplicationOracle,
) -> tuple[float, int, int]:
    """Families are (block length, input columns, output rows, repetition count)."""

    baseline = 0
    candidate = 0
    for m, k, n, count in families:
        baseline += int(count) * int(m) * int(k) * int(n)
        candidate += int(count) * oracle.cost(int(m), int(k), int(n))
    if baseline <= 0:
        raise StructuredDirectSumError("empty weighted population")
    return candidate / baseline, candidate, baseline
