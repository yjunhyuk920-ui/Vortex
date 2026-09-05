"""Scoped, executable native-observation source, not a fast inference engine.

Mathematical reference inspired by the raw-exponent A100 model in
https://arxiv.org/html/2512.07004v3 and upstream A100TC.m blob
 a8bd8af19589671099f31f0c0d8c58642c3398c1.
No upstream implementation is copied or executed. This does not validate a GPU.
Scope: normal finite BF16 weights/inputs; non-overflowing normal FP32 block
results (or mathematical zero); one explicitly present final BF16 store.
A BF16 store MUST NOT be inserted between native blocks that retain FP32.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F


def p2(e: int) -> F:
    return F(1 << e) if e >= 0 else F(1, 1 << -e)


def floor_log2(q: F) -> int:
    if q <= 0:
        raise ValueError('positive rational required')
    e = q.numerator.bit_length() - q.denominator.bit_length()
    return e if q >= p2(e) else e - 1


def trunc(q: F) -> int:
    return (1 if q >= 0 else -1) * (abs(q.numerator) // q.denominator)


def rounded(q: F, precision: int, nearest: bool) -> F:
    """Exact rational rounding in the declared normal, unbounded-exponent model."""
    if not q:
        return F(0)
    sign = 1 if q > 0 else -1
    q = abs(q)
    unit = p2(floor_log2(q) - precision + 1)
    z = q / unit
    a, r = divmod(z.numerator, z.denominator)
    if nearest and (2*r > z.denominator or (2*r == z.denominator and a & 1)):
        a += 1
    return sign * a * unit


def stored(q: F) -> F:
    """Normal BF16 nearest-even store. Exceptional ranges fail closed."""
    if q and not p2(-126) <= abs(q) < p2(127):
        raise ValueError('outside the scoped normal-result domain')
    return rounded(q, 8, True)


@dataclass(frozen=True)
class BFloat:
    sign: int
    exponent: int
    mantissa: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1) or not -126 <= self.exponent <= 127:
            raise ValueError('normal BF16 sign/exponent required')
        if not 0 <= self.mantissa < 128:
            raise ValueError('seven-bit mantissa required')

    def value(self) -> F:
        return self.sign * (128 + self.mantissa) * p2(self.exponent - 7)


class PrefixSource:
    """Only headers and subsequently requested bits are exposed to the generator.

    Preparing this object reads all original 16-bit weights once. Logical query
    counts exclude no headers. They are NOT PCIe/page transactions or timings.
    """
    def __init__(self, weights: list[BFloat]):
        if len(weights) != 8:
            raise ValueError('exactly eight weights required')
        self.__words = tuple(weights)
        self.__positions = [0] * 8
        self.header_bits = self.mantissa_bits = 0
        self.preparation_bits = 16 * len(weights)

    def headers(self) -> list[tuple[int, int]]:
        self.header_bits += 9 * len(self.__words)
        return [(w.sign, w.exponent) for w in self.__words]

    def next_bit(self, i: int) -> int:
        k = self.__positions[i]
        if k >= 7:
            raise ValueError('no undisclosed mantissa bit remains')
        self.__positions[i] += 1
        self.mantissa_bits += 1
        return (self.__words[i].mantissa >> (6-k)) & 1


def prefix_interval(header: tuple[int, int], prefix: int, k: int) -> tuple[F, F]:
    sign, exponent = header
    mlo = prefix << (7-k)
    mhi = mlo + (1 << (7-k)) - 1
    v = [sign * (128+m) * p2(exponent-7) for m in (mlo, mhi)]
    return min(v), max(v)


def interval_cell(lo: F, hi: F) -> F | None:
    """Apply monotonicity of the final CAST, not of the tensor-core operator."""
    if lo > hi:
        raise ValueError('reversed interval')
    if lo <= 0 <= hi:
        return F(0) if lo == hi == 0 else None
    a, b = stored(lo), stored(hi)
    return a if a == b else None


def generate(source: PrefixSource, x: list[BFloat], c: F = F(0)) -> dict:
    """Generate a certified store using paid prefix reads; no reference output."""
    if len(x) != 8:
        raise ValueError('eight inputs required')
    h = source.headers()
    if any(abs(e) > 30 for _, e in h) or any(abs(v.exponent) > 30 for v in x):
        raise ValueError('reference scope requires operand exponents in [-30,30]')
    if c and (not p2(-60) <= abs(c) <= p2(60) or rounded(c,24,False) != c):
        raise ValueError('reference scope requires an exact normal FP32 accumulator')
    E = max([h[i][1] + x[i].exponent for i in range(8)]
            + ([floor_log2(abs(c))] if c else []) + [-132])
    unit = p2(E-24)
    qc = trunc(c / unit)
    prefixes, known = [0]*8, [0]*8
    qlo, qhi = [0]*8, [0]*8
    endpoint_products = 0

    def update(i: int) -> None:
        nonlocal endpoint_products
        lo, hi = prefix_interval(h[i], prefixes[i], known[i])
        a, b = lo*x[i].value(), hi*x[i].value()
        qlo[i], qhi[i] = trunc(min(a, b)/unit), trunc(max(a, b)/unit)
        endpoint_products += 2

    for i in range(8):
        update(i)
    checks = 0
    while True:
        checks += 1
        lo = rounded((qc + sum(qlo))*unit, 24, False)
        hi = rounded((qc + sum(qhi))*unit, 24, False)
        out = interval_cell(lo, hi)
        if out is not None:
            return {'value': out, 'lo': lo, 'hi': hi, 'known': known,
                    'header_bits': source.header_bits,
                    'mantissa_bits': source.mantissa_bits,
                    'endpoint_products': endpoint_products, 'checks': checks,
                    'preparation_bits': source.preparation_bits}
        choices = [i for i in range(8) if known[i] < 7]
        if not choices:
            raise AssertionError('full information must produce a singleton')
        i = max(choices, key=lambda j: (qhi[j]-qlo[j], -j))
        prefixes[i] = 2*prefixes[i] + source.next_bit(i)
        known[i] += 1
        update(i)


def oracle(weights: list[BFloat], x: list[BFloat], c: F = F(0)) -> F:
    """Independent integer-product formulation, outside the generator."""
    E = max([a.exponent+b.exponent for a,b in zip(weights,x)]
            + ([floor_log2(abs(c))] if c else []) + [-132])
    total = trunc(c / p2(E-24))
    for a, b in zip(weights, x):
        m = (128+a.mantissa)*(128+b.mantissa)
        shift = a.exponent+b.exponent-14-(E-24)
        aligned = m << shift if shift >= 0 else m >> -shift
        total += a.sign*b.sign*aligned
    # Truncate integer significand to 24 significant bits before BF16 store.
    if not total:
        return F(0)
    cut = max(0, abs(total).bit_length()-24)
    kept = (abs(total) >> cut) << cut
    z = (1 if total > 0 else -1) * kept * p2(E-24)
    return stored(z)


def operand_model(a: list[F], b: list[F], c: F) -> F:
    """Rational reference, zero operands allowed, for the hand-derived witness."""
    E = max([floor_log2(abs(u))+floor_log2(abs(v))
             for u,v in zip(a,b) if u and v]
            + ([floor_log2(abs(c))] if c else []) + [-132])
    unit = p2(E-24)
    total = trunc(c/unit) + sum(trunc(u*v/unit) for u,v in zip(a,b))
    return rounded(total*unit, 24, False)
