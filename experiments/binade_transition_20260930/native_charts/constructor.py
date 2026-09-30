"""Complete non-enumerative scalar native chart constructor for finite addends.

All coefficient products and all chart work are paid; this is not acceleration.
NaN initial words and nonfinite product operands are deliberately rejected.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crossing import PeriodicMap
from reference import BF16, FP32, Format, pow2


@dataclass(frozen=True)
class Addend:
    value: Q
    zero_sign: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'value', Q(self.value))
        denominator = self.value.denominator
        if denominator & (denominator-1):
            raise ValueError('The declared product domain is exact finite dyadics')
        if self.zero_sign not in (0, 1):
            raise ValueError('Zero sign must be 0 or 1')


@dataclass(frozen=True)
class Region:
    low: Q | None
    low_closed: bool
    high: Q | None
    high_closed: bool
    kind: str  # grid, poszero, negzero, exactzero, posinf, neginf
    grid: Q | None = None

    def before(self, value):
        return self.high is not None and (self.high < value or (self.high == value and not self.high_closed))

    def after(self, value):
        return self.low is not None and (self.low > value or (self.low == value and not self.low_closed))


def regions(fmt: Format) -> list[Region]:
    hmin = pow2(fmt.emin - fmt.frac_bits)
    minimum_normal = pow2(fmt.emin)
    overflow = fmt.decode(fmt.maxfinite) + pow2(fmt.emax - fmt.frac_bits - 1)
    positive = [Region(Q(0), False, hmin/2, True, 'poszero'),
                Region(hmin/2, False, minimum_normal, False, 'grid', hmin)]
    for e in range(fmt.emin, fmt.emax+1):
        hi = pow2(e+1) if e < fmt.emax else overflow
        positive.append(Region(pow2(e), True, hi, False, 'grid', pow2(e-fmt.frac_bits)))
    positive.append(Region(overflow, True, None, False, 'posinf'))
    negative = []
    for region in reversed(positive):
        kind = {'poszero': 'negzero', 'posinf': 'neginf'}.get(region.kind, region.kind)
        negative.append(Region(-region.high if region.high is not None else None,
                               region.high_closed,
                               -region.low if region.low is not None else None,
                               region.low_closed, kind, region.grid))
    return negative + [Region(Q(0), True, Q(0), True, 'exactzero')] + positive


def word_key(fmt, word):
    if not isinstance(word, int) or not 0 <= word < 2*fmt.sign_bit:
        raise ValueError('A valid bounded format word is required')
    mag = fmt.magnitude(word)
    if mag > fmt.inf:
        raise ValueError('NaN payloads are outside the declared scalar domain')
    return fmt.inf-mag if fmt.sign(word) else fmt.inf+1+mag


def key_word(fmt, key):
    if not 0 <= key <= 2*fmt.inf+1:
        raise ValueError('Invalid ordered word key')
    return fmt.sign_bit | (fmt.inf-key) if key <= fmt.inf else key-fmt.inf-1


@dataclass(frozen=True)
class Chart:
    low: int
    high: int
    offset: int = 0  # Initial lattice index k = ordered input key + offset.
    mapping: PeriodicMap | None = None
    constant: int | None = None


class NativeCharts:
    def __init__(self, fmt: Format):
        self.fmt = fmt
        self.regions = regions(fmt)
        self.charts = self.initial_charts()
        self.stats = dict(steps=0, supplied_addends=0, exact_products=0, coefficient_reads=0,
                          input_reads=0, chart_visits=0, region_advances=0, region_intersections=0,
                          inverse_calls=0, inverse_level_quotients=0, map_updates=0,
                          intersection_inverse_calls=0, intersection_inverse_level_quotients=0,
                          growth_inverse_calls=0, growth_inverse_level_quotients=0,
                          growing_grid_updates=0, map_grid_round_calls=0,
                          map_internal_evaluations=0,
                          scalar_round_calls=0, constant_steps=0, endpoint_evaluations=0,
                          input_charts=len(self.charts), peak_charts=len(self.charts),
                          final_charts=len(self.charts), max_cut_bits=0, max_level_numerator_bits=0,
                          max_level_denominator_bits=0, constant_intervals_created=0,
                          max_addend_numerator_bits=0, max_addend_denominator_bits=0,
                          adjacent_merges=0, invariant_chart_checks=0, invariant_field_checks=0,
                          query_calls=0, query_chart_comparisons=0, query_scalar_round_calls=0,
                          query_word_checks=0)
        self.steps = []

    def initial_charts(self):
        fmt, charts = self.fmt, []
        for word in (fmt.sign_bit | fmt.inf, fmt.sign_bit, 0, fmt.inf):
            key = word_key(fmt, word)
            charts.append(Chart(key, key, constant=word))
        L = 1 << fmt.frac_bits
        # Subnormal nonzero intervals, then normal exponent bins. O(exponent count),
        # never O(number of words).
        for sign in (0, 1):
            sign_mask = fmt.sign_bit if sign else 0
            for maglo, maghi, h, signed_k_low in [(1, L-1, pow2(fmt.emin-fmt.frac_bits),
                                                  -(L-1) if sign else 1)]:
                if maglo <= maghi:
                    klo = word_key(fmt, sign_mask | (maghi if sign else maglo))
                    khi = word_key(fmt, sign_mask | (maglo if sign else maghi))
                    charts.append(Chart(klo, khi, signed_k_low-klo, PeriodicMap.identity(h)))
            for e in range(fmt.emin, fmt.emax+1):
                field = e-fmt.emin+1
                maglo, maghi = field*L, (field+1)*L-1
                klo = word_key(fmt, sign_mask | (maghi if sign else maglo))
                khi = word_key(fmt, sign_mask | (maglo if sign else maghi))
                signed_k_low = -(2*L-1) if sign else L
                charts.append(Chart(klo, khi, signed_k_low-klo,
                                    PeriodicMap.identity(pow2(e-fmt.frac_bits))))
        charts.sort(key=lambda chart: chart.low)
        return charts

    def inverse(self, mapping, threshold, strict):
        self.stats['inverse_calls'] += 1
        self.stats['inverse_level_quotients'] += len(mapping.levels)
        self.stats['intersection_inverse_calls'] += 1
        self.stats['intersection_inverse_level_quotients'] += len(mapping.levels)
        return mapping.inverse(threshold, strict)

    def update_mapping(self, mapping, h, addend):
        """Charge sealed PeriodicMap.then's internal work without changing it.

        On grid growth, that exact implementation computes self(0), rounds one
        base level, and calls inverse for EACH of the two target levels. On a
        non-growing grid it rounds each prior level. Validator03 independently
        spies on every inverse call, including these internal ones.
        """
        if h > mapping.peak_grid:
            calls, terms = 2, 2*len(mapping.levels)
            self.stats['growing_grid_updates'] += 1
            self.stats['growth_inverse_calls'] += calls
            self.stats['growth_inverse_level_quotients'] += terms
            self.stats['inverse_calls'] += calls
            self.stats['inverse_level_quotients'] += terms
            self.stats['map_internal_evaluations'] += 1
            self.stats['map_grid_round_calls'] += 1
        else:
            self.stats['map_grid_round_calls'] += len(mapping.levels)
        return mapping.then(h, addend)

    def intersect(self, chart, region, addend):
        lo, hi = chart.low+chart.offset, chart.high+chart.offset
        if region.low is not None:
            lo = max(lo, self.inverse(chart.mapping, region.low-addend, not region.low_closed))
        if region.high is not None:
            hi = min(hi, self.inverse(chart.mapping, region.high-addend, region.high_closed)-1)
        return (lo-chart.offset, hi-chart.offset) if lo <= hi else None

    def make_constant(self, low, high, word):
        self.stats['constant_intervals_created'] += 1
        return Chart(low, high, constant=word)

    def canonicalize(self, chart):
        if chart.constant is not None:
            return chart
        lo_value = chart.mapping(chart.low+chart.offset)
        hi_value = chart.mapping(chart.high+chart.offset)
        self.stats['endpoint_evaluations'] += 2
        # Nonconstant-grid charts exclude both signed-zero basins before update.
        if not lo_value or not hi_value:
            raise AssertionError('An untagged zero escaped its explicit constant region')
        if lo_value == hi_value:
            self.stats['scalar_round_calls'] += 1
            word = self.fmt.round(lo_value)
            assert self.fmt.is_finite(word)
            return self.make_constant(chart.low, chart.high, word)
        return chart

    def merge(self, charts):
        result = []
        for chart in charts:
            if result:
                previous = result[-1]
                assert previous.high+1 == chart.low
                same_constant = previous.constant is not None and previous.constant == chart.constant
                same_map = (previous.constant is None and chart.constant is None and
                            previous.offset == chart.offset and previous.mapping == chart.mapping)
                if same_constant or same_map:
                    result[-1] = replace(previous, high=chart.high)
                    self.stats['adjacent_merges'] += 1
                    continue
            result.append(chart)
        return result

    def append(self, addend: Addend):
        fmt, c = self.fmt, addend.value
        new, cursor = [], 0
        before = dict(self.stats)
        old_count = len(self.charts)
        self.stats['steps'] += 1
        self.stats['supplied_addends'] += 1
        self.stats['max_addend_numerator_bits'] = max(self.stats['max_addend_numerator_bits'],
                                                     abs(c.numerator).bit_length())
        self.stats['max_addend_denominator_bits'] = max(self.stats['max_addend_denominator_bits'],
                                                       c.denominator.bit_length())
        for chart in self.charts:
            self.stats['chart_visits'] += 1
            if chart.constant is not None:
                self.stats['constant_steps'] += 1
                if fmt.is_finite(chart.constant):
                    self.stats['scalar_round_calls'] += 1
                word = fmt.add_exact(chart.constant, c, addend.zero_sign)
                new.append(self.make_constant(chart.low, chart.high, word))
                continue
            zlo = chart.mapping(chart.low+chart.offset)+c
            zhi = chart.mapping(chart.high+chart.offset)+c
            self.stats['endpoint_evaluations'] += 2
            assert zlo <= zhi
            while self.regions[cursor].before(zlo):
                cursor += 1
                self.stats['region_advances'] += 1
            index, last = cursor, cursor
            while index < len(self.regions) and not self.regions[index].after(zhi):
                region = self.regions[index]
                self.stats['region_intersections'] += 1
                interval = self.intersect(chart, region, c)
                if interval is not None:
                    lo, hi = interval
                    if region.kind == 'grid':
                        self.stats['map_updates'] += 1
                        mapping = self.update_mapping(chart.mapping, region.grid, c)
                        part = self.canonicalize(Chart(lo, hi, chart.offset, mapping))
                    else:
                        word = {'poszero': 0, 'negzero': fmt.sign_bit, 'exactzero': 0,
                                'posinf': fmt.inf, 'neginf': fmt.sign_bit | fmt.inf}[region.kind]
                        # If c=0, mapped intervals were nonzero by invariant.
                        # Exact cancellation of nonzero operands returns +0.
                        part = self.make_constant(lo, hi, word)
                    new.append(part)
                last = index
                index += 1
            cursor = last
        self.charts = self.merge(new)
        self.stats['peak_charts'] = max(self.stats['peak_charts'], len(self.charts))
        self.stats['final_charts'] = len(self.charts)
        self.check_invariants()
        delta = {key: self.stats[key]-before[key] for key in self.stats if key not in
                 ('peak_charts','final_charts','input_charts','max_cut_bits',
                  'max_level_numerator_bits','max_level_denominator_bits',
                  'max_addend_numerator_bits','max_addend_denominator_bits')}
        self.steps.append(dict(before_charts=old_count, after_charts=len(self.charts), **delta))
        assert delta['region_intersections'] <= old_count+len(self.regions)

    def check_invariants(self):
        assert self.charts[0].low == 0 and self.charts[-1].high == 2*self.fmt.inf+1
        for i, chart in enumerate(self.charts):
            self.stats['invariant_chart_checks'] += 1
            assert chart.low <= chart.high
            if i:
                assert self.charts[i-1].high+1 == chart.low
            if chart.constant is not None:
                assert self.fmt.magnitude(chart.constant) <= self.fmt.inf
            else:
                chart.mapping.check_invariants()
                for cut in chart.mapping.cuts:
                    self.stats['invariant_field_checks'] += 1
                    self.stats['max_cut_bits'] = max(self.stats['max_cut_bits'], cut.bit_length())
                for value in chart.mapping.levels:
                    self.stats['invariant_field_checks'] += 1
                    self.stats['max_level_numerator_bits'] = max(self.stats['max_level_numerator_bits'],
                                                                abs(value.numerator).bit_length())
                    self.stats['max_level_denominator_bits'] = max(self.stats['max_level_denominator_bits'],
                                                                  value.denominator.bit_length())
        # A disjoint ordered R-region partition introduces <=R-1 cuts globally.
        # The exact-zero singleton has two adjacent cuts already in that count.
        assert len(self.charts) <= self.stats['input_charts'] + (len(self.regions)-1)*self.stats['steps']

    def query(self, word):
        self.stats['query_calls'] += 1
        key = word_key(self.fmt, word)
        # Search is logarithmic without materializing a fresh list per query.
        lo, hi = 0, len(self.charts)
        while lo+1 < hi:
            self.stats['query_chart_comparisons'] += 1
            mid = (lo+hi)//2
            if self.charts[mid].low <= key:
                lo = mid
            else:
                hi = mid
        chart = self.charts[lo]
        assert chart.low <= key <= chart.high
        if chart.constant is not None:
            return chart.constant
        value = chart.mapping(key+chart.offset)
        assert value
        self.stats['query_scalar_round_calls'] += 1
        result = self.fmt.round(value)
        self.stats['query_word_checks'] += 1
        assert self.fmt.decode(result) == value  # An exact native state, not another rounding.
        return result


def compile_addends(fmt: Format, addends):
    result = NativeCharts(fmt)
    result.check_invariants()
    for addend in addends:
        result.append(addend if isinstance(addend, Addend) else Addend(addend))
    return result


def compile_fma_operands(pairs, accumulator_format=FP32, operand_format=BF16):
    result = NativeCharts(accumulator_format)
    result.check_invariants()
    for w, x in pairs:
        word_key(operand_format, w)
        word_key(operand_format, x)
        if not operand_format.is_finite(w) or not operand_format.is_finite(x):
            raise ValueError('Nonfinite product operands require a separately pinned ABI')
        result.stats['coefficient_reads'] += 1
        result.stats['input_reads'] += 1
        result.stats['exact_products'] += 1
        product = operand_format.decode(w) * operand_format.decode(x)
        result.append(Addend(product, operand_format.sign(w) ^ operand_format.sign(x)))
    return result
