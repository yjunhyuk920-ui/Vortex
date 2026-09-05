"""Exact binary32 transfer references. No target-model performance claim.

Native ABI: fmaf, RN-even, gradual underflow, fixed serial order.  The phase
summary is guarded; the suffix evaluator pays for every probe and retry.
Python big integers/Fractions are executable specifications, not GPU kernels.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
_FMA = _LIBM.fmaf
_FMA.argtypes = [ctypes.c_float] * 3
_FMA.restype = ctypes.c_float
if _LIBM.fegetround() != 0:
    raise RuntimeError("This reference requires FE_TONEAREST")


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def fma(w: float, x: float, a: float) -> float:
    return float(_FMA(w, x, a))


def dyadic(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def round_even(x: Fraction) -> int:
    q, r = divmod(x.numerator, x.denominator)
    t = 2 * r - x.denominator
    return q + int(t > 0 or (t == 0 and q % 2 != 0))


def floor_log2(x: Fraction) -> int:
    if x <= 0:
        raise ValueError("positive argument required")
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e - int(x < dyadic(e))


def ceil_log2(x: Fraction) -> int:
    e = floor_log2(x)
    return e + int(x != dyadic(e))


def round32_fraction(x: Fraction, zero_sign: int = 0) -> int:
    """Independent integer RN-even oracle, including subnormals/overflow."""
    if not x:
        return zero_sign << 31
    sign = int(x < 0)
    x = abs(x)
    e = floor_log2(x)
    q = max(e - 23, -149)
    m = round_even(x / dyadic(q))
    if m == 0:
        return sign << 31
    if e < -126 and m < (1 << 23):
        return (sign << 31) | m
    if m >= (1 << 24):
        m >>= 1
        e += 1
    e = max(e, -126)
    if e > 127:
        return (sign << 31) | 0x7F800000
    return (sign << 31) | ((e + 127) << 23) | (m - (1 << 23))


def exact_fma_word(w: float, x: float, a: float) -> int:
    if not all(math.isfinite(t) for t in (w, x, a)):
        raise ValueError("integer oracle accepts finite binary32 operands")
    w, x, a = (value(bits(t)) for t in (w, x, a))
    product = Fraction(w) * Fraction(x)
    total = product + Fraction(a)
    zs = 0
    if product == 0 and a == 0:
        zs = ((bits(w) ^ bits(x)) >> 31) & (bits(a) >> 31)
    return round32_fraction(total, zs)


def reference_dot(w: Sequence[float], x: Sequence[float], initial: float = 0.0) -> float:
    if len(w) != len(x):
        raise ValueError("shape mismatch")
    a = initial
    for wi, xi in zip(w, x):
        a = fma(wi, xi, a)
    return a


@dataclass(frozen=True)
class PhaseSummary:
    delta: tuple[int, int] = (0, 0)
    low: tuple[int, int] = (0, 0)
    high: tuple[int, int] = (0, 0)
    length: int = 0

    @staticmethod
    def leaf(scaled_product: Fraction) -> "PhaseSummary":
        d = tuple(round_even(Fraction(p) + scaled_product) - p for p in (0, 1))
        return PhaseSummary(d, tuple(min(0, v) for v in d),
                            tuple(max(0, v) for v in d), 1)

    def then(self, other: "PhaseSummary") -> "PhaseSummary":
        d, lo, hi = [], [], []
        for p in (0, 1):
            q = (p + self.delta[p]) & 1
            d.append(self.delta[p] + other.delta[q])
            lo.append(min(self.low[p], self.delta[p] + other.low[q]))
            hi.append(max(self.high[p], self.delta[p] + other.high[q]))
        return PhaseSummary(tuple(d), tuple(lo), tuple(hi), self.length + other.length)

    def apply(self, initial: float, exponent: int) -> float | None:
        """None is a rejected guard, never an approximate result."""
        b = bits(initial)
        raw_exp = (b >> 23) & 255
        if raw_exp in (0, 255) or raw_exp - 127 != exponent:
            return None
        sign = -1 if b >> 31 else 1
        k = sign * ((1 << 23) | (b & 0x7FFFFF))
        lower, upper = ((1 << 23) + 1, (1 << 24) - 1)
        if sign < 0:
            lower, upper = -upper, -lower
        p = k & 1
        if k + self.low[p] < lower or k + self.high[p] > upper:
            return None
        out = k + self.delta[p]
        return value(bits(math.ldexp(float(out), exponent - 23)))


def summarize(w: Sequence[float], x: Sequence[float], exponent: int,
              balanced: bool = True) -> PhaseSummary:
    """Paid O(n) products/leaf construction. No offline product oracle."""
    if len(w) != len(x):
        raise ValueError("shape mismatch")
    if not all(math.isfinite(t) for t in (*w, *x)):
        raise ValueError("nonfinite operand: use the original native path")
    w, x = tuple(value(bits(t)) for t in w), tuple(value(bits(t)) for t in x)
    if not -126 <= exponent <= 127:
        raise ValueError("normal binary32 binade exponent required")
    unit = dyadic(exponent - 23)
    nodes = [PhaseSummary.leaf(Fraction(wi) * Fraction(xi) / unit)
             for wi, xi in zip(w, x)]
    if not nodes:
        return PhaseSummary()
    if not balanced:
        ans = PhaseSummary()
        for node in nodes:
            ans = ans.then(node)
        return ans
    while len(nodes) > 1:
        nodes = [nodes[i].then(nodes[i + 1]) if i + 1 < len(nodes) else nodes[i]
                 for i in range(0, len(nodes), 2)]
    return nodes[0]


@dataclass(frozen=True)
class PrefixIndex:
    weights: tuple[float, ...]
    block: int
    exponents: tuple[int | None, ...]
    constructor_weight_reads: int

    @staticmethod
    def build(weights: Sequence[float], block: int = 32) -> "PrefixIndex":
        if block < 1:
            raise ValueError("positive block required")
        w = tuple(value(bits(t)) for t in weights)
        if not all(math.isfinite(t) for t in w):
            raise ValueError("index requires finite weights")
        total = Fraction(0)
        out: list[int | None] = [None]
        for i, wi in enumerate(w, 1):
            total += abs(Fraction(wi))
            if i % block == 0 or i == len(w):
                out.append(ceil_log2(total) if total else None)
        return PrefixIndex(w, block, tuple(out), len(w))

    @property
    def metadata_bytes(self) -> int:
        # Signed 16-bit exponent with a zero sentinel; lengths/header additional.
        return 2 * len(self.exponents) + 16

    def bound(self, prefix_length: int, xmax: float) -> float | None:
        if prefix_length == 0:
            return 0.0
        n = len(self.weights)
        if not 0 <= prefix_length <= n or not math.isfinite(xmax) or xmax < 0:
            raise ValueError("invalid bound arguments")
        u, eta = dyadic(-24), dyadic(-150)
        if prefix_length * u >= 1:
            return None
        index = (prefix_length + self.block - 1) // self.block
        exponent = self.exponents[index]
        s = Fraction(0) if exponent is None else dyadic(exponent)
        r = (s * Fraction(xmax) + prefix_length * eta) / (1 - prefix_length * u)
        if not r:
            return 0.0
        e = ceil_log2(r)
        if e > 126:  # Conservative finite-intermediate guarantee; paid full path.
            return None
        e = max(e, -149)
        return value(bits(math.ldexp(1.0, e)))


@dataclass
class Counts:
    xmax_input_reads: int = 0
    endpoint_fmas: int = 0
    full_path_fmas: int = 0
    distinct_weight_reads: int = 0
    repeated_weight_operand_reads: int = 0
    repeated_x_operand_reads: int = 0
    metadata_reads: int = 0
    attempts: int = 0
    largest_suffix: int = 0
    certified: bool = False

    @property
    def total_fmas(self) -> int:
        return self.endpoint_fmas + self.full_path_fmas


def suffix_dot(index: PrefixIndex, x: Sequence[float]) -> tuple[float, Counts]:
    """Causal, terminating evaluator with explicit expensive unresolved cases.

    All weights are available in this CPU reference. Counters describe logical
    access, NOT measured SSD/PCIe pages or GPU traffic. Weight caching needs up
    to n*4 bytes plus input; repeats are separately counted.
    """
    n = len(index.weights)
    if len(x) != n:
        raise ValueError("shape mismatch")
    xx = tuple(value(bits(t)) for t in x)
    if not all(math.isfinite(t) for t in xx):
        raise ValueError("finite input required")
    c = Counts(xmax_input_reads=n)
    if n == 0:
        return 0.0, c
    xmax = max(abs(t) for t in xx)
    cached: set[int] = set()
    k = min(index.block, n)
    while k < n:
        r = n - k
        c.attempts += 1
        c.metadata_reads += 1
        radius = index.bound(r, xmax)
        if radius is None:
            break
        low, high = -radius, radius
        for j in range(r, n):
            cached.add(j)
            low = fma(index.weights[j], xx[j], low)
            high = fma(index.weights[j], xx[j], high)
        c.endpoint_fmas += 2 * k
        c.repeated_weight_operand_reads += k  # one load can serve both endpoints
        c.repeated_x_operand_reads += k
        c.largest_suffix = k
        if math.isfinite(low) and bits(low) == bits(high):
            c.certified = True
            c.distinct_weight_reads = len(cached)
            return low, c
        k = min(2 * k, n)
    out = 0.0
    for j in range(n):
        cached.add(j)
        out = fma(index.weights[j], xx[j], out)
    c.full_path_fmas = n
    c.repeated_weight_operand_reads += n
    c.repeated_x_operand_reads += n
    c.distinct_weight_reads = len(cached)
    c.largest_suffix = n
    return out, c


MIN_FINITE_RANK = 0x00800000
MAX_FINITE_RANK = 0xFF7FFFFF


def rank_word(word: int) -> int:
    return (~word & 0xFFFFFFFF) if word >> 31 else (word ^ 0x80000000)


def unrank_word(rank: int) -> int:
    return (rank ^ 0x80000000) if rank >> 31 else (~rank & 0xFFFFFFFF)


def preimage(w: float, x: float, out_low: int, out_high: int) -> tuple[int, int, int] | None:
    """Finite exact native-FMA inverse by two bounded monotone searches.

    Arguments/results are total-order ranks, including both signed zero words.
    A nonempty inverse uses <=64 fmaf calls; no universe-sized transition table.
    Nonfinite operands/target NaNs are outside this primitive's admitted scope.
    """
    if not math.isfinite(w) or not math.isfinite(x):
        raise ValueError("finite operands required")
    if not MIN_FINITE_RANK <= out_low <= out_high <= MAX_FINITE_RANK:
        raise ValueError("finite ordered output interval required")
    calls = 0

    def bound(target: int, strict: bool) -> int:
        nonlocal calls
        lo, hi = MIN_FINITE_RANK, MAX_FINITE_RANK + 1
        while lo < hi:
            mid = (lo + hi) // 2
            y = fma(w, x, value(unrank_word(mid)))
            calls += 1
            rank = rank_word(bits(y))
            if rank > target if strict else rank >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    left = bound(out_low, False)
    right = bound(out_high, True) - 1
    if left > right or left > MAX_FINITE_RANK or right < MIN_FINITE_RANK:
        return None
    return left, right, calls


def _first_above(t: Fraction, inclusive: bool) -> int:
    """First finite IEEE total-order rank with numerical a >= t (or > t)."""
    biggest = Fraction(value(0x7F7FFFFF))
    if t > biggest or (t == biggest and not inclusive):
        return MAX_FINITE_RANK + 1
    if t < -biggest:
        return MIN_FINITE_RANK
    if t == 0:
        return rank_word(0x80000000) if inclusive else rank_word(1)
    word = round32_fraction(t)
    if word & 0x7FFFFFFF == 0x7F800000:
        return MIN_FINITE_RANK if word >> 31 else MAX_FINITE_RANK + 1
    a = Fraction(value(word))
    return rank_word(word) + int(a < t or (a == t and not inclusive))


def _last_below(t: Fraction, inclusive: bool) -> int:
    """Last finite rank with numerical a <= t (or < t), including signed zero."""
    biggest = Fraction(value(0x7F7FFFFF))
    if t < -biggest or (t == -biggest and not inclusive):
        return MIN_FINITE_RANK - 1
    if t > biggest:
        return MAX_FINITE_RANK
    if t == 0:
        return rank_word(0) if inclusive else rank_word(0x80000001)
    word = round32_fraction(t)
    if word & 0x7FFFFFFF == 0x7F800000:
        return MAX_FINITE_RANK if not word >> 31 else MIN_FINITE_RANK - 1
    a = Fraction(value(word))
    return rank_word(word) - int(a > t or (a == t and not inclusive))


def preimage_direct(w: float, x: float, out_low: int, out_high: int) -> tuple[int, int, str, int] | None:
    """Exact rounding-cell inverse; no fmaf search for nonzero endpoints.

    Return (first rank,last rank,method,native calls). Integer/Fraction costs
    are NOT zero. A zero endpoint is dispatched to the bounded native method,
    preserving signed-zero semantics rather than silently assuming real zeros.
    """
    if not math.isfinite(w) or not math.isfinite(x):
        raise ValueError("finite operands required")
    if not MIN_FINITE_RANK <= out_low <= out_high <= MAX_FINITE_RANK:
        raise ValueError("finite ordered output interval required")
    lw, hw = unrank_word(out_low), unrank_word(out_high)
    if (lw & 0x7FFFFFFF) == 0 or (hw & 0x7FFFFFFF) == 0:
        result = preimage(w,x,out_low,out_high)
        if result is None:
            return None
        return result[0],result[1],'native_zero_boundary',result[2]
    low,high = Fraction(value(lw)),Fraction(value(hw))
    if out_low == MIN_FINITE_RANK:
        left = low - dyadic(103)
    else:
        left = (Fraction(value(unrank_word(out_low-1))) + low) / 2
    if out_high == MAX_FINITE_RANK:
        right = high + dyadic(103)
    else:
        right = (high + Fraction(value(unrank_word(out_high+1)))) / 2
    product = Fraction(value(bits(w))) * Fraction(value(bits(x)))
    start = _first_above(left-product, inclusive=(lw & 1)==0)
    stop = _last_below(right-product, inclusive=(hw & 1)==0)
    start=max(start,MIN_FINITE_RANK); stop=min(stop,MAX_FINITE_RANK)
    if start>stop:
        return None
    return start,stop,'exact_rounding_cell',0


def suffix_preimage(w: Sequence[float], x: Sequence[float], output_word: int) -> tuple[int,int,int,int] | None:
    """Explicit backward program for a known suffix and a proposed output word.

    No claim that the proposed output is the target's unknown true output.
    A caller must prove its incoming state lies in the returned interval.
    Finite intermediate-state semantics only; otherwise the native path applies.
    """
    if len(w)!=len(x):
        raise ValueError('shape mismatch')
    if not 0 <= output_word <= 0xFFFFFFFF or output_word & 0x7F800000 == 0x7F800000:
        raise ValueError("finite output word required")
    lo=hi=rank_word(output_word)
    native_calls=direct_stages=0
    for wi,xi in zip(reversed(w),reversed(x)):
        ans=preimage_direct(wi,xi,lo,hi)
        if ans is None:
            return None
        lo,hi,method,calls=ans
        native_calls+=calls
        direct_stages+=method=='exact_rounding_cell'
    return lo,hi,native_calls,direct_stages
