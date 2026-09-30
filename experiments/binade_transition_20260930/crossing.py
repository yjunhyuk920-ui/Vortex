"""Compact fixed-itinerary dyadic-RNE maps, independent of native chart search.

This is NOT a complete native FMA/GEMM constructor. It implements the non-
enumerative constant-level chart algebra used in CROSSING_DERIVATION.md.
"""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction as Q

from reference import ceil, floor


def round_grid(z: Q, h: Q) -> Q:
    q = floor(z / h)
    residue = z / h - q
    return h * (q + int(residue > Q(1, 2) or (residue == Q(1, 2) and q % 2)))


def is_power_of_two(q: Q) -> bool:
    return q > 0 and (q.numerator & (q.numerator - 1)) == 0 and (q.denominator & (q.denominator - 1)) == 0


@dataclass(frozen=True)
class PeriodicMap:
    input_grid: Q
    peak_grid: Q
    period: int
    height: Q
    cuts: tuple[int, ...]  # Always begins with 0; remaining cuts are strict.
    levels: tuple[Q, ...]

    @staticmethod
    def identity(h0: Q):
        h0 = Q(h0)
        if not is_power_of_two(h0):
            raise ValueError("A power-of-two input grid is required")
        return PeriodicMap(h0, h0, 2, 2*h0, (0, 1), (Q(0), h0))

    def __call__(self, k: int) -> Q:
        q, r = divmod(k, self.period)
        ix = bisect_right(self.cuts, r) - 1
        return q * self.height + self.levels[ix]

    def inverse(self, threshold: Q, strict: bool = False) -> int:
        """Least integer k with f(k)>=threshold (or >threshold when strict).

        Each residue interval has one constant value plus a period height.
        At most three exact quotient computations replace a potentially huge
        period search. Values and period lengths are NOT unit-cost words.
        """
        candidates = []
        for cut, level in zip(self.cuts, self.levels):
            quotient = (threshold - level) / self.height
            q = floor(quotient) + 1 if strict else ceil(quotient)
            candidates.append(q * self.period + cut)
        return min(candidates)

    def then(self, h: Q, addend: Q):
        h, addend = Q(h), Q(addend)
        if not is_power_of_two(h):
            raise ValueError("Only dyadic power-of-two grids are supported")
        if h <= self.peak_grid:
            cuts, levels = [], []
            for cut, level in zip(self.cuts, self.levels):
                value = round_grid(level + addend, h)
                if not levels or value != levels[-1]:
                    cuts.append(cut)
                    levels.append(value)
            return PeriodicMap(self.input_grid, self.peak_grid, self.period, self.height,
                               tuple(cuts), tuple(levels))
        new_height = 2*h
        period_q = new_height / self.input_grid
        assert period_q.denominator == 1
        new_period = period_q.numerator
        base = round_grid(self(0) + addend, h)
        cuts, levels = [0], [base]
        for step in (1, 2):
            target = base + step*h
            target_integer = target / h
            assert target_integer.denominator == 1
            # At the midpoint the target is selected iff its grid index is even.
            strict = bool(target_integer.numerator % 2)
            threshold = target - h/2 - addend
            k = self.inverse(threshold, strict)
            assert k > 0  # At k=0 the new value is exactly base, below target.
            if k >= new_period:
                continue
            if k == cuts[-1]:
                levels[-1] = target  # Jump over an unoccupied middle level.
            else:
                cuts.append(k)
                levels.append(target)
        return PeriodicMap(self.input_grid, h, new_period, new_height, tuple(cuts), tuple(levels))

    def check_invariants(self):
        assert self.cuts[0] == 0
        assert len(self.cuts) == len(self.levels) <= 3
        assert all(a < b for a, b in zip(self.cuts, self.cuts[1:]))
        assert self.cuts[-1] < self.period
        assert all(a < b for a, b in zip(self.levels, self.levels[1:]))
        assert self.levels[-1] <= self.levels[0] + self.height
        assert self.height == 2*self.peak_grid == self.period*self.input_grid

