"""Exact guarded normal-binade transition algebra; no learned/model inputs.

All arithmetic here is a specification cost, not a fast native implementation.
The scalar reference is independently defined by a nearest-representable search.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Iterable


def pow2(e: int) -> Q:
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)


def floor(x: Q) -> int:
    return x.numerator // x.denominator


def ceil(x: Q) -> int:
    return -floor(-x)


@dataclass(frozen=True)
class Format:
    precision: int
    exponent_bits: int

    @property
    def frac_bits(self) -> int:
        return self.precision - 1

    @property
    def emin(self) -> int:
        return 2 - (1 << (self.exponent_bits - 1))

    @property
    def emax(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.frac_bits + self.exponent_bits)

    @property
    def inf(self) -> int:
        return ((1 << self.exponent_bits) - 1) << self.frac_bits

    @property
    def maxfinite(self) -> int:
        return self.inf - 1

    def sign(self, word: int) -> int:
        return int(bool(word & self.sign_bit))

    def magnitude(self, word: int) -> int:
        return word & (self.sign_bit - 1)

    def is_finite(self, word: int) -> bool:
        return self.magnitude(word) < self.inf

    def is_normal(self, word: int) -> bool:
        mag = self.magnitude(word)
        return (1 << self.frac_bits) <= mag < self.inf

    def exponent(self, word: int) -> int:
        if not self.is_normal(word):
            raise ValueError("A normal word is required")
        return (self.magnitude(word) >> self.frac_bits) + self.emin - 1

    def decode(self, word: int) -> Q:
        if not self.is_finite(word):
            raise ValueError("Infinity/NaN has no finite Fraction value")
        mag = self.magnitude(word)
        exponent_field = mag >> self.frac_bits
        significand = mag & ((1 << self.frac_bits) - 1)
        if exponent_field:
            significand += 1 << self.frac_bits
            e = exponent_field + self.emin - 1 - self.frac_bits
        else:
            e = self.emin - self.frac_bits
        return (-1 if self.sign(word) else 1) * significand * pow2(e)

    def round(self, value: Q, zero_sign: int = 0) -> int:
        """Independent nearest-representable reference, not the parity formula.

        Binary-search the ordered positive finite encodings. This never
        enumerates an FP32 table. A tie selects the even encoding. Overflow
        is treated using the exact midpoint to the next power of two.
        """
        value = Q(value)
        if not value:
            return self.sign_bit if zero_sign else 0
        sign = self.sign_bit if value < 0 else 0
        a = abs(value)
        maxvalue = self.decode(self.maxfinite)
        overflow_midpoint = maxvalue + pow2(self.emax - self.frac_bits - 1)
        if a >= overflow_midpoint:
            return sign | self.inf
        if a >= maxvalue:
            return sign | self.maxfinite
        low, high = 0, self.maxfinite
        while low + 1 < high:
            mid = (low + high) // 2
            if self.decode(mid) <= a:
                low = mid
            else:
                high = mid
        dl, dh = a - self.decode(low), self.decode(high) - a
        chosen = low if dl < dh or (dl == dh and low % 2 == 0) else high
        return sign | chosen

    def add_exact(self, accumulator: int, product: Q, product_zero_sign: int = 0) -> int:
        """One RNE operation on accumulator + exact dyadic product.

        Finite products and finite/infinite accumulators are supported. NaN
        behavior is deliberately not invented; an actual executor must call
        its original instruction on special cases.
        """
        mag = self.magnitude(accumulator)
        if mag > self.inf:
            raise ValueError("NaN payload semantics are outside this reference")
        if mag == self.inf:
            return accumulator
        a = self.decode(accumulator)
        z = a + product
        zero_sign = int(not a and not product and self.sign(accumulator) and product_zero_sign)
        return self.round(z, zero_sign=zero_sign)

    def fma(self, accumulator: int, w: int, x: int, operand_format: Format) -> int:
        product = operand_format.decode(w) * operand_format.decode(x)
        return self.add_exact(accumulator, product, operand_format.sign(w) ^ operand_format.sign(x))


FP32 = Format(24, 8)
BF16 = Format(8, 8)


def parity_interval(low: int, high: int, parity: int) -> tuple[int, int]:
    low += (parity - low) % 2
    high -= (high - parity) % 2
    return (low, high) if low <= high else (1, -1)


@dataclass(frozen=True)
class Grid:
    fmt: Format
    exponent: int
    negative: bool = False

    def __post_init__(self):
        if not self.fmt.emin <= self.exponent <= self.fmt.emax:
            raise ValueError("Not a normal binade")

    @property
    def h(self) -> Q:
        return pow2(self.exponent - self.fmt.frac_bits)

    @property
    def lower_integer(self) -> int:
        return 1 << self.fmt.frac_bits

    @property
    def upper_integer(self) -> int:
        return 1 << self.fmt.precision

    @property
    def state_bounds(self) -> tuple[int, int]:
        lo, hi = self.lower_integer, self.upper_integer
        if self.exponent == self.fmt.emax:
            hi -= 1  # The next power of two is infinity, not a finite state.
        return (-hi, -lo) if self.negative else (lo, hi)

    @property
    def exact_guard(self) -> tuple[Q, bool, Q, bool]:
        """Safe exact-sum interval in units of h; closed flags explicit.

        Includes the full left rounding cell of the lower endpoint and the
        uniform-grid-compatible half-cell above the upper endpoint.
        """
        lo = Q(self.lower_integer) - (Q(1, 2) if self.exponent == self.fmt.emin else Q(1, 4))
        if self.exponent == self.fmt.emax:
            hi, hc = Q(self.upper_integer) - Q(1, 2), False
        else:
            hi, hc = Q(self.upper_integer) + Q(1, 2), True
        return (-hi, hc, -lo, True) if self.negative else (lo, True, hi, hc)


@dataclass(frozen=True)
class Summary:
    grid: Grid
    delta: tuple[int, int]
    domains: tuple[tuple[int, int], tuple[int, int]]
    length: int = 0

    def accepts(self, k: int) -> bool:
        lo, hi = self.domains[k % 2]
        return lo <= k <= hi

    def apply(self, k: int) -> int:
        if not self.accepts(k):
            raise ValueError("Outside the constructed guard")
        return k + self.delta[k % 2]


def identity(grid: Grid) -> Summary:
    lo, hi = grid.state_bounds
    return Summary(grid, (0, 0), tuple(parity_interval(lo, hi, s) for s in range(2)))


def leaf(grid: Grid, exact_product: Q) -> Summary:
    t = Q(exact_product) / grid.h
    q = floor(t)
    r = t - q
    if r < Q(1, 2):
        delta = (q, q)
    elif r > Q(1, 2):
        delta = (q + 1, q + 1)
    else:
        delta = tuple(q + ((s + q) % 2) for s in range(2))
    gl, lc, gh, hc = grid.exact_guard
    low_value, high_value = gl - t, gh - t
    low = ceil(low_value)
    high = floor(high_value)
    if not lc and Q(low) == low_value:
        low += 1
    if not hc and Q(high) == high_value:
        high -= 1
    state_lo, state_hi = grid.state_bounds
    low, high = max(low, state_lo), min(high, state_hi)
    domains = tuple(parity_interval(low, high, s) for s in range(2))
    # A physical representation never retains an enormous irrelevant offset.
    # Nonempty-domain offsets are bounded by the width of grid.state_bounds.
    delta = tuple(delta[s] if domains[s][0] <= domains[s][1] else 0 for s in range(2))
    return Summary(grid, delta, domains, 1)


def compose(first: Summary, second: Summary) -> Summary:
    """Ordered composition second(first(k)), including every intermediate guard."""
    if first.grid != second.grid:
        raise ValueError("Only summaries on the same grid compose directly")
    deltas, domains = [], []
    for s in range(2):
        d = first.delta[s]
        r = (s + d) % 2
        lo1, hi1 = first.domains[s]
        lo2, hi2 = second.domains[r]
        if lo1 > hi1 or lo2 > hi2:
            domain = (1, -1)
        else:
            domain = parity_interval(max(lo1, lo2 - d), min(hi1, hi2 - d), s)
        deltas.append(d + second.delta[r] if domain[0] <= domain[1] else 0)
        domains.append(domain)
    return Summary(first.grid, tuple(deltas), tuple(domains), first.length + second.length)


def compile_summary(grid: Grid, products: Iterable[Q]) -> Summary:
    result = identity(grid)
    for product in products:
        result = compose(result, leaf(grid, product))
    return result


def execute_with_escapes(fmt: Format, initial: int, products: Iterable[Q | tuple[Q, int]]) -> tuple[int, dict]:
    """One-pass exact executor with a paid scalar instruction on first escape.

    Products are supplied as exact finite dyadics. In an FMA implementation,
    forming them from both operands is charged separately, once per term.
    No product, intermediate guard, or coefficient is skipped.
    """
    current = initial
    grid = None
    running = None
    start = None
    counts = dict(terms=0, leaves=0, compositions=0, guards=0, escapes=0,
                  special_steps=0, decodes=0, accepted_terms=0, segments=0)

    def flush():
        nonlocal current, grid, running, start
        if running is not None:
            value = running.apply(start) * grid.h
            current = fmt.round(value)
            counts['decodes'] += 1
        grid = running = start = None

    for item in products:
        product, zero_sign = item if isinstance(item, tuple) else (item, 0)
        product = Q(product)
        counts['terms'] += 1
        if running is None:
            if not fmt.is_normal(current):
                current = fmt.add_exact(current, product, zero_sign)
                counts['special_steps'] += 1
                continue
            grid = Grid(fmt, fmt.exponent(current), bool(fmt.sign(current)))
            quotient = fmt.decode(current) / grid.h
            assert quotient.denominator == 1
            start = quotient.numerator
            running = identity(grid)
            counts['segments'] += 1
        candidate = compose(running, leaf(grid, product))
        counts['leaves'] += 1
        counts['compositions'] += 1
        counts['guards'] += 1
        if candidate.accepts(start):
            running = candidate
            counts['accepted_terms'] += 1
        else:
            flush()
            current = fmt.add_exact(current, product, zero_sign)
            counts['escapes'] += 1
    flush()
    return current, counts
