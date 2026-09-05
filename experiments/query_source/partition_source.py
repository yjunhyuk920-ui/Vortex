"""Paid query-dependent source for a SERIAL binary32 FMA matrix-vector ABI.

This is NOT a universal subdense algorithm or a Hugging Face executor.
Its complete index and selection costs can be much worse than dense execution.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, asdict
from typing import Sequence

from experiments.native_transfer.native_transfer import bits, value, fma


def native(w: int, x: int, a: int) -> int:
    return bits(fma(value(w), value(x), value(a)))


def _word(x: int) -> int:
    if not isinstance(x, int) or not 0 <= x <= 0xFFFFFFFF:
        raise ValueError("expected a uint32 word")
    return x


def _merge(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Deterministic sort/merge, not a perfect or free hash-table selector."""
    pairs.sort(key=lambda p: p[0])
    out: list[tuple[int, int]] = []
    for word, mask in pairs:
        if out and out[-1][0] == word:
            out[-1] = (word, out[-1][1] | mask)
        else:
            out.append((word, mask))
    return out


@dataclass(frozen=True)
class Column:
    lo: int
    hi: int
    buckets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Index:
    rows: int
    cols: int
    columns: tuple[Column, ...]

    @property
    def mask_words(self) -> int:
        return (self.rows + 63) // 64

    def encode(self) -> bytes:
        """An actual lossless serialized index; Python resident memory is extra."""
        out = bytearray(struct.pack("<8sIIQQ", b"QSPFMA01", self.rows,
                                    self.cols, sum(len(c.buckets) for c in self.columns), 0))
        width = 8 * self.mask_words
        for col in self.columns:
            out.extend(struct.pack("<III", col.lo, col.hi, len(col.buckets)))
            for word, mask in col.buckets:
                out.extend(struct.pack("<I", word))
                out.extend(mask.to_bytes(width, "little"))
        struct.pack_into("<Q", out, 24, len(out))
        return bytes(out)


def build(weights: Sequence[Sequence[int]]) -> Index:
    """Read all mn original words ONCE, group every column by exact weight bits."""
    m = len(weights)
    if m == 0:
        raise ValueError("empty matrix")
    n = len(weights[0])
    if n == 0 or any(len(row) != n for row in weights):
        raise ValueError("nonrectangular or empty matrix")
    cols: list[Column] = []
    for j in range(n):
        pairs: list[tuple[int, int]] = []
        for i, row in enumerate(weights):
            word = _word(row[j])
            if not math.isfinite(value(word)):
                raise ValueError("this bounded reference requires finite weights")
            pairs.append((word, 1 << i))
        buckets = tuple(_merge(pairs))
        lo = min((w for w, _ in buckets), key=value)
        hi = max((w for w, _ in buckets), key=value)
        cols.append(Column(lo, hi, buckets))
    return Index(m, n, tuple(cols))


@dataclass
class Counts:
    header_bytes: int = 32
    input_words: int = 0
    initial_state_words: int = 0
    bound_record_bytes: int = 0
    bound_fmas: int = 0
    bucket_fmas: int = 0
    range_cuts: int = 0
    range_cut_row_updates: int = 0
    mask_intersections: int = 0
    mask_words_examined: int = 0
    merge_records: int = 0
    merge_mask_words_upper: int = 0
    cold_bucket_payload_bytes: int = 0
    output_words: int = 0
    maximum_state_classes: int = 0

    def json(self) -> dict:
        out = asdict(self)
        out["total_native_fmas"] = self.bound_fmas + self.bucket_fmas
        out["logical_cold_index_bytes"] = self.header_bytes + self.bound_record_bytes + self.cold_bucket_payload_bytes
        out["bitmap_AND_operand_bytes_upper"] = 16 * self.mask_words_examined
        out["bitmap_merge_bytes_upper"] = 24 * self.merge_mask_words_upper
        return out


def query(index: Index, x: Sequence[int], initial: Sequence[int] | None = None
          ) -> tuple[list[int], Counts]:
    """Return ALL row results; no target output, trace or future token is an input.

    Uses an index, not the original matrix. Range collapse is only accepted for
    equal, finite, NONZERO endpoint result words; other cases use exact buckets.
    The latter is a paid continuation of this pass, not an uncharged restart.
    """
    if len(x) != index.cols:
        raise ValueError("input width mismatch")
    x = tuple(_word(v) for v in x)
    if not all(math.isfinite(value(v)) for v in x):
        raise ValueError("finite input required by this reference")
    initial = [0] * index.rows if initial is None else list(initial)
    if len(initial) != index.rows:
        raise ValueError("initial state length mismatch")
    states = _merge([(_word(v), 1 << i) for i, v in enumerate(initial)])
    cnt = Counts(input_words=index.cols, initial_state_words=index.rows)
    L = index.mask_words
    for col, xj in zip(index.columns, x):
        cnt.bound_record_bytes += 12
        cnt.maximum_state_classes = max(cnt.maximum_state_classes, len(states))
        nxt: list[tuple[int, int]] = []
        decoded = False
        for a, rows in states:
            cnt.bound_fmas += 2
            lo, hi = native(col.lo, xj, a), native(col.hi, xj, a)
            if lo == hi and math.isfinite(value(lo)) and value(lo) != 0.0 and math.isfinite(value(a)):
                cnt.range_cuts += 1
                cnt.range_cut_row_updates += rows.bit_count()
                nxt.append((lo, rows))
                continue
            if not decoded:
                cnt.cold_bucket_payload_bytes += len(col.buckets) * (4 + 8 * L)
                decoded = True
            for w, members in col.buckets:
                cnt.mask_intersections += 1
                cnt.mask_words_examined += L
                both = rows & members
                if both:
                    cnt.bucket_fmas += 1
                    nxt.append((native(w, xj, a), both))
        cnt.merge_records += len(nxt)
        cnt.merge_mask_words_upper += len(nxt) * L
        states = _merge(nxt)
    out = [0] * index.rows
    for word, rows in states:
        while rows:
            bit = rows & -rows
            out[bit.bit_length() - 1] = word
            rows ^= bit
    cnt.output_words = len(out)
    cnt.maximum_state_classes = max(cnt.maximum_state_classes, len(states))
    return out, cnt


def reference(weights: Sequence[Sequence[int]], x: Sequence[int],
              initial: Sequence[int] | None = None) -> list[int]:
    out = [0] * len(weights) if initial is None else list(initial)
    for i, row in enumerate(weights):
        for w, v in zip(row, x):
            out[i] = native(w, v, out[i])
    return out
