"""Exact conditional output-cohort executor for a frozen FP32/BF16 ABI.

Not a Transformers implementation and not a universal acceleration claim.
The runtime reader consumes only requested raw BF16 row groups; metadata is hot.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


def decode(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def encode(values: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(values, dtype=np.float32)
    u = a.view(np.uint32)
    with np.errstate(over='ignore', invalid='ignore'):
        z = ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = ((u & 0x7f800000) == 0x7f800000) & ((u & 0x7fffff) != 0)
    return np.where(nan, np.uint16(0x7fc0), z).astype(np.uint16)


def is_finite(words: np.ndarray) -> bool:
    return bool(np.all((np.asarray(words, dtype=np.uint16) & 0x7f80) != 0x7f80))


def reduction(products: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(products, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError('nonempty matrix of products required')
    n = p.shape[1]
    width = 1 << (n - 1).bit_length()
    finite = np.all(np.isfinite(p), axis=1)
    if n != width:
        p = np.pad(p, ((0, 0), (0, width - n)), constant_values=0)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        while p.shape[1] > 1:
            p = np.add(p[:, 0::2], p[:, 1::2], dtype=np.float32)
            finite &= np.all(np.isfinite(p), axis=1)
    return p[:, 0], finite


def direct(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.uint16)
    x = np.asarray(x, dtype=np.uint16)
    if w.ndim != 2 or x.shape != (w.shape[1],):
        raise ValueError('shape mismatch')
    if not is_finite(w) or not is_finite(x):
        raise ValueError('input NaN/Inf outside declared ABI')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        p = np.multiply(decode(w), decode(x)[None, :], dtype=np.float32)
    return encode(reduction(p)[0])


@dataclass
class Plan:
    shape: tuple[int, int]
    packet: int
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def build(cls, w: np.ndarray, packet: int = 256) -> 'Plan':
        w = np.asarray(w, dtype=np.uint16)
        if w.ndim != 2 or min(w.shape) == 0 or packet <= 0:
            raise ValueError('invalid shape or packet')
        if not is_finite(w):
            raise ValueError('nonfinite original coefficient')
        v = decode(w)
        lo, hi = [], []
        for a in range(0, len(w), packet):
            lo.append(encode(np.min(v[a:a+packet], axis=0)))
            hi.append(encode(np.max(v[a:a+packet], axis=0)))
        return cls(tuple(w.shape), packet, np.array(lo), np.array(hi))

    def save(self, path) -> None:
        np.savez(path, shape=np.array(self.shape, dtype='<u8'),
                 packet=np.array(self.packet, dtype='<u8'), low=self.low, high=self.high)

    @classmethod
    def load(cls, path) -> 'Plan':
        with np.load(path, allow_pickle=False) as z:
            shape = tuple(int(i) for i in z['shape'])
            packet = int(z['packet'])
            lo, hi = z['low'].copy(), z['high'].copy()
        if len(shape) != 2 or min(shape) < 1 or packet < 1:
            raise ValueError('invalid plan dimensions')
        expected = ((shape[0] + packet - 1)//packet, shape[1])
        if lo.shape != expected or hi.shape != expected or not is_finite(lo) or not is_finite(hi):
            raise ValueError('invalid envelope')
        if np.any(decode(lo) > decode(hi)):
            raise ValueError('inverted envelope')
        return cls(shape, packet, lo, hi)


def execute(plan: Plan, x: np.ndarray,
            reader: Callable[[int, int], np.ndarray]) -> tuple[np.ndarray, dict]:
    m, n = plan.shape
    x = np.asarray(x, dtype=np.uint16)
    if x.shape != (n,) or not is_finite(x):
        raise ValueError('invalid input')
    xx = decode(x)[None, :]
    lo, hi = decode(plan.low), decode(plan.high)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        pl = np.multiply(np.where(xx < 0, hi, lo), xx, dtype=np.float32)
        ph = np.multiply(np.where(xx < 0, lo, hi), xx, dtype=np.float32)
    l, lsafe = reduction(pl)
    h, hsafe = reduction(ph)
    lb, hb = encode(l), encode(h)
    accepted = lsafe & hsafe & (lb == hb) & ((lb & 0x7fff) != 0) & ((lb & 0x7f80) != 0x7f80)
    out = np.empty(m, dtype=np.uint16)
    rows = 0
    traces = []
    for group, start in enumerate(range(0, m, plan.packet)):
        end = min(start + plan.packet, m)
        if accepted[group]:
            out[start:end] = lb[group]
            rows += end - start
        else:
            w = np.asarray(reader(start, end), dtype=np.uint16)
            if w.shape != (end-start, n):
                raise ValueError('reader returned wrong shape')
            out[start:end] = direct(w, x)
        traces.append({'row_start':start,'row_end':end,'accepted':bool(accepted[group]),
                       'lower_f32_bits':int(l[group].view(np.uint32)),
                       'upper_f32_bits':int(h[group].view(np.uint32)),
                       'lower_bf16_bits':int(lb[group]), 'upper_bf16_bits':int(hb[group])})
    groups = len(accepted)
    width = 1 << (n - 1).bit_length()
    remaining = m - rows
    baseline_ops = m * (n + width - 1)
    actual_ops = (2 * groups + remaining) * (n + width - 1)
    metadata_payload = 2 * groups * n * 2
    cold_payload = remaining * n * 2
    return out, {'rows':m, 'columns':n, 'groups':groups,
                 'accepted_groups':int(np.count_nonzero(accepted)), 'accepted_rows':rows,
                 'raw_weight_bytes':m*n*2,'summary_payload_bytes':metadata_payload,
                 'cold_weight_bytes_read':cold_payload,'hot_summary_bytes_read':metadata_payload,
                 'coefficient_payload_ratio':(metadata_payload+cold_payload)/(m*n*2),
                 'baseline_fp_ops':baseline_ops,'interval_and_direct_fp_ops':actual_ops,
                 'fp_work_ratio':actual_ops/baseline_ops,
                 'input_bytes':x.nbytes, 'output_bytes':out.nbytes,
                 'logical_counts_not_physical_traffic':True, 'groups_trace':traces}
