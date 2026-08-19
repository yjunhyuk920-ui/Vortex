from __future__ import annotations

import hashlib, json, math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

P50 = 1.2 * 4.0 / 405.0
TOTAL = 405_849_243_648
ONE_MLP_PROJ = 109_924_319_232
MLP_SHARE = 3.0 * ONE_MLP_PROJ / TOTAL
H405, I405, L405 = 16_384, 53_248, 126


class MicroprogramError(RuntimeError):
    pass


def pct(values: Iterable[float], q: float) -> float:
    a = sorted(float(x) for x in values)
    if not a:
        raise MicroprogramError("empty percentile")
    p = (len(a) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return a[lo] if lo == hi else a[lo] * (hi - p) + a[hi] * (p - lo)


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-30)


def exact_report(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    if ref.shape != cand.shape or ref.dtype != torch.bfloat16 or cand.dtype != torch.bfloat16:
        raise MicroprogramError("BF16 shape/dtype mismatch")
    eq = ref == cand
    vec = eq.all(1)
    rows = eq.float().mean(1)
    bad = (~eq).sum(1)
    return {
        "vector_exact_fraction": float(vec.float().mean()),
        "row_exact_fraction_p50": pct(rows.tolist(), .5),
        "row_exact_fraction_p95": pct(rows.tolist(), .95),
        "mismatched_rows_p50": pct(bad.tolist(), .5),
        "mismatched_rows_max": int(bad.max()),
    }


def cluster(x: torch.Tensor, k: int, iters: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or k < 1 or x.shape[0] < k:
        raise MicroprogramError("invalid cluster input")
    z = norm_rows(x)
    chosen = [0]
    nearest = torch.full((z.shape[0],), float("inf"), dtype=torch.float64)
    while len(chosen) < k:
        nearest = torch.minimum(nearest, ((z - z[chosen[-1]]) ** 2).sum(1))
        nearest[torch.tensor(chosen)] = -1
        chosen.append(int(torch.argmax(nearest)))
    c = z[chosen].clone()
    a = torch.zeros(z.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = ((z[:, None] - c[None]) ** 2).sum(2)
        a = torch.argmin(d, 1)
        counts = torch.bincount(a, minlength=k)
        if torch.any(counts == 0):
            loss = d.gather(1, a[:, None]).squeeze(1)
            for empty in torch.nonzero(counts == 0).flatten().tolist():
                donor = int(torch.argmax(loss)); a[donor] = empty; loss[donor] = -1
        new = []
        for j in range(k):
            v = z[a == j].mean(0); new.append(v / torch.linalg.vector_norm(v).clamp_min(1e-30))
        new_c = torch.stack(new)
        if torch.equal(new_c, c):
            break
        c = new_c
    return a, c


def _sha(t: torch.Tensor) -> str:
    b = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class Program:
    xc64: torch.Tensor; yc64: torch.Tensor; v64: torch.Tensor; c64: torch.Tensor
    xc32: torch.Tensor; yc32: torch.Tensor; v32: torch.Tensor; c32: torch.Tensor
    centroid: torch.Tensor; build_count: int; rank: int; fingerprint: str

    def query(self, x: torch.Tensor, precision: str) -> torch.Tensor:
        if x.ndim == 1: x = x[None]
        if x.ndim != 2 or x.shape[1] != self.xc64.numel() or not torch.isfinite(x.float()).all():
            raise MicroprogramError("invalid query")
        if precision == "fp64":
            y = self.yc64 + ((x.double() - self.xc64) @ self.v64) @ self.c64
        elif precision == "fp32":
            y = self.yc32 + ((x.float() - self.xc32) @ self.v32) @ self.c32
        else:
            raise MicroprogramError("invalid precision")
        return y.to(torch.bfloat16)


def fit_program(x: torch.Tensor, y: torch.Tensor, rank_cap: int, centroid: torch.Tensor) -> Program:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise MicroprogramError("invalid build pairs")
    x, y = x.double(), y.double()
    xc, yc = x.mean(0), y.mean(0)
    xx, yy = x - xc, y - yc
    u, s, vh = torch.linalg.svd(xx, full_matrices=False)
    tol = max(xx.shape) * torch.finfo(torch.float64).eps * float(s[0])
    rank = min(rank_cap, int((s > tol).sum()))
    if rank < 1: raise MicroprogramError("zero rank")
    v = vh[:rank].T.contiguous()
    coef = ((u[:, :rank].T @ yy) / s[:rank, None]).contiguous()
    fp = hashlib.sha256(json.dumps({"x":_sha(xc),"y":_sha(yc),"v":_sha(v),"c":_sha(coef)},sort_keys=True).encode()).hexdigest()
    return Program(xc,yc,v,coef,xc.float(),yc.float(),v.float(),coef.float(),centroid.double(),x.shape[0],rank,fp)


@dataclass(frozen=True)
class Library:
    programs: tuple[Program, ...]; rank_cap: int; fingerprint: str

    def _pred(self, x: torch.Tensor, precision: str) -> torch.Tensor:
        return torch.stack([p.query(x, precision) for p in self.programs], 1)

    def query(self, x: torch.Tensor, ref: torch.Tensor, precision: str, oracle: bool) -> dict[str, Any]:
        pred = self._pred(x, precision)
        if oracle:
            selected = torch.argmax((pred == ref[:, None]).sum(2), 1)
            mode = "oracle"
        else:
            z = norm_rows(x); c = torch.stack([p.centroid for p in self.programs])
            selected = torch.argmin(((z[:, None] - c[None]) ** 2).sum(2), 1)
            mode = "router"
        cand = pred[torch.arange(pred.shape[0]), selected]
        return {
            "mode": mode, "precision": precision,
            "selected": [int(v) for v in selected.tolist()],
            "use_counts": [int(v) for v in torch.bincount(selected,minlength=len(self.programs)).tolist()],
            "report": exact_report(ref, cand), "candidate": cand,
        }


def fit_library(x: torch.Tensor, y: torch.Tensor, k: int, rank_cap: int) -> tuple[Library, torch.Tensor]:
    a, centroids = cluster(x, k)
    programs = []
    for j in range(k):
        mask = a == j
        if int(mask.sum()) < 2: raise MicroprogramError("small cluster")
        programs.append(fit_program(x[mask], y[mask], rank_cap, centroids[j]))
    fp = hashlib.sha256("|".join(p.fingerprint for p in programs).encode()).hexdigest()
    return Library(tuple(programs), rank_cap, fp), a


def target_resources(k: int, rank: int, scalar_bytes: int = 4) -> dict[str, Any]:
    scalars = 3 * H405 + 2 * H405 * rank
    sidecar = k * L405 * scalars * scalar_bytes
    router_ops = H405 + 3 * k * H405
    program_ops = 2 * H405 * rank + H405
    frac = (router_ops + program_ops) / (3 * H405 * I405)
    return {
        "sidecar_bytes": sidecar, "sidecar_gib": sidecar / 1024**3,
        "compiled_mlp_operation_fraction": frac,
        "whole_model_fraction_if_only_mlp_replaced": 1 - MLP_SHARE + MLP_SHARE * frac,
        "uncompiled_non_mlp_fraction": 1 - MLP_SHARE,
    }


def residual_diagnostics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    xor = torch.bitwise_xor(ref.contiguous().view(torch.int16), cand.contiguous().view(torch.int16))
    hashes = [_sha(row) for row in xor]
    unique = [int(torch.unique(xor[:, j]).numel()) for j in range(xor.shape[1])]
    return {
        "unique_residual_vector_fraction": len(set(hashes)) / ref.shape[0],
        "zero_word_fraction": float((xor == 0).float().mean()),
        "unique_words_per_coordinate_p50": pct(unique,.5),
        "unique_words_per_coordinate_p95": pct(unique,.95),
    }
