from __future__ import annotations

import bz2, hashlib, math, zlib
from typing import Any, Iterable, Sequence

import numpy as np
import torch


class ContextCodeError(RuntimeError):
    pass


def words(t: torch.Tensor) -> np.ndarray:
    t = t.detach().contiguous().cpu()
    if t.dtype != torch.bfloat16 or t.ndim != 2 or not torch.isfinite(t.float()).all():
        raise ContextCodeError("expected finite BF16 matrix")
    return (t.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)).numpy().astype(np.uint64)


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def entropy(a: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    if not a.size:
        raise ContextCodeError("empty entropy population")
    _, c = np.unique(a, return_counts=True)
    p = c.astype(np.float64) / a.size
    return float(-(p * np.log2(p)).sum())


def percentile(values: Iterable[float], q: float) -> float:
    x = sorted(map(float, values)); p = (len(x) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return x[lo] if lo == hi else x[lo] * (hi - p) + x[hi] * (p - lo)


def pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, np.uint64) | (np.asarray(b, np.uint64) << np.uint64(16))


def triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return pair(a, b) | (np.asarray(c, np.uint64) << np.uint64(32))


def majority(context: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    c = np.asarray(context, np.uint64).reshape(-1); t = np.asarray(target, np.uint64).reshape(-1)
    if c.shape != t.shape or not c.size: raise ContextCodeError("bad context population")
    order = np.lexsort((t, c)); cs, ts = c[order], t[order]
    starts = np.r_[0, np.nonzero((cs[1:] != cs[:-1]) | (ts[1:] != ts[:-1]))[0] + 1]
    stops = np.r_[starts[1:], len(cs)]; counts = stops - starts
    pc, pt = cs[starts], ts[starts]
    gs = np.r_[0, np.nonzero(pc[1:] != pc[:-1])[0] + 1]
    ge = np.r_[gs[1:], len(pc)]; gids = np.repeat(np.arange(len(gs)), ge - gs)
    maxima = np.maximum.reduceat(counts, gs); idx = np.arange(len(counts))
    chosen = np.minimum.reduceat(np.where(counts == maxima[gids], idx, len(idx)), gs)
    keys, table = pc[chosen], pt[chosen]
    pos = np.searchsorted(keys, c)
    if not np.array_equal(keys[pos], c): raise ContextCodeError("lookup failure")
    pred = table[pos]; residual = pred ^ t
    return {"keys": keys, "table": table, "pred": pred, "residual": residual,
            "hit": float(np.mean(pred == t)), "exact": bool(np.array_equal(pred ^ residual, t))}


def lane_residual_bits(r: np.ndarray) -> float:
    r = np.asarray(r, np.uint64); n = r.size
    return sum(entropy(((r >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.uint16)) * n for k in range(3))


def candidate(name: str, baseline: float, table: float, sequence: float, residual: float,
              contexts: int, hit: float, exact: bool, meta: dict[str, Any]) -> dict[str, Any]:
    total = float(table + sequence + residual)
    return {"name": name, "total_bits": total, "baseline_bits": baseline,
            "information_fraction": total / baseline, "table_bits": float(table),
            "sequence_bits": float(sequence), "residual_bits": float(residual),
            "context_count": int(contexts), "hit_fraction": float(hit),
            "reconstruction_exact": bool(exact), "metadata": meta}


def audit_layer(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                layer: int, periods: Sequence[int]) -> dict[str, Any]:
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ContextCodeError("unaligned SwiGLU shapes")
    G, U, D = words(gate), words(up), words(down.T.contiguous())
    arrays = {"G": G, "U": U, "D": D}; n = G.size
    h = {k: entropy(v.astype(np.uint16)) for k, v in arrays.items()}
    baseline = sum(h.values()) * n; names = ("G", "U", "D"); out = []
    for base in names:
        targets = [x for x in names if x != base]; target = pair(arrays[targets[0]], arrays[targets[1]])
        p = majority(arrays[base], target); r = p["residual"]
        residual = sum(entropy(((r >> np.uint64(16*k)) & 0xFFFF).astype(np.uint16)) * n for k in range(2))
        out.append(candidate(f"one_source_{base}", baseline, 32*len(p["table"]), h[base]*n,
                             residual, len(p["keys"]), p["hit"], p["exact"], {"targets": targets}))
    for target_name in names:
        bases = [x for x in names if x != target_name]; context = pair(arrays[bases[0]], arrays[bases[1]])
        p = majority(context, arrays[target_name]); r = p["residual"].astype(np.uint16)
        out.append(candidate(f"two_source_{''.join(bases)}_to_{target_name}", baseline,
                             48*len(p["table"]), entropy(context)*n, entropy(r)*n,
                             len(p["keys"]), p["hit"], p["exact"], {}))
    T = triplet(G, U, D); p = majority(T[:, :-1], T[:, 1:])
    out.append(candidate("previous_triplet", baseline, 96*len(p["table"]), 48*T.shape[0],
                         lane_residual_bits(p["residual"]), len(p["keys"]), p["hit"], p["exact"], {}))
    for period in periods:
        pred = np.empty_like(T); table = []
        for i in range(period):
            for j in range(period):
                values = T[i::period, j::period].reshape(-1)
                if not values.size: continue
                unique, counts = np.unique(values, return_counts=True); value = unique[int(np.argmax(counts))]
                table.append(value); pred[i::period, j::period] = value
        residual = pred ^ T
        out.append(candidate(f"coordinate_period_{period}", baseline, 48*len(table), 0,
                             lane_residual_bits(residual), len(table), float(np.mean(pred == T)),
                             bool(np.array_equal(pred ^ residual, T)), {"period": period}))
    if not all(x["reconstruction_exact"] for x in out): raise ContextCodeError("reconstruction mismatch")
    best = min(out, key=lambda x: (x["information_fraction"], x["name"]))
    raw = [x.detach().contiguous().view(torch.uint8).numpy().tobytes() for x in (gate, up, down.T.contiguous())]
    interleaved = np.stack((G, U, D), -1).astype("<u2").tobytes()
    codecs = {}
    for name, payload in {"concat": b"".join(raw), "interleaved": interleaved}.items():
        codecs[name] = {"zlib9_fraction": len(zlib.compress(payload, 9))/len(payload),
                        "bz2_9_fraction": len(bz2.compress(payload, 9))/len(payload)}
    return {"layer_index": layer, "shape": list(gate.shape),
            "baseline_entropy_bits": baseline, "baseline_bits_per_triplet": baseline/n,
            "role_entropy_bits": h, "candidates": out, "best_candidate": best,
            "codec_diagnostics": codecs,
            "weight_sha256": {"gate": sha(gate), "up": sha(up), "down": sha(down)}}


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fractions = [x["best_candidate"]["information_fraction"] for x in rows]
    return {"layer_count": len(rows), "best_information_fraction_p05": percentile(fractions, .05),
            "best_information_fraction_p50": percentile(fractions, .50),
            "best_information_fraction_p95": percentile(fractions, .95),
            "minimum_information_fraction": min(fractions), "maximum_information_fraction": max(fractions),
            "best_candidate_names": [x["best_candidate"]["name"] for x in rows],
            "all_reconstructions_exact": all(c["reconstruction_exact"] for x in rows for c in x["candidates"])}
