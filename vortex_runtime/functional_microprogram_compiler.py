from __future__ import annotations

import hashlib
import json
import math
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
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MicroprogramError("empty percentile")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    values = x.double()
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-30)


def exact_report(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if (
        reference.shape != candidate.shape
        or reference.dtype != torch.bfloat16
        or candidate.dtype != torch.bfloat16
    ):
        raise MicroprogramError("BF16 shape/dtype mismatch")
    equal = reference == candidate
    vector_exact = equal.all(1)
    row_exact = equal.float().mean(1)
    mismatched = (~equal).sum(1)
    return {
        "vector_exact_fraction": float(vector_exact.float().mean()),
        "row_exact_fraction_p50": pct(row_exact.tolist(), 0.50),
        "row_exact_fraction_p95": pct(row_exact.tolist(), 0.95),
        "mismatched_rows_p50": pct(mismatched.tolist(), 0.50),
        "mismatched_rows_max": int(mismatched.max()),
    }


def cluster(x: torch.Tensor, k: int, iterations: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or k < 1 or x.shape[0] < k:
        raise MicroprogramError("invalid cluster input")
    normalized = norm_rows(x)
    selected = [0]
    nearest = torch.full((normalized.shape[0],), float("inf"), dtype=torch.float64)
    while len(selected) < k:
        nearest = torch.minimum(
            nearest, ((normalized - normalized[selected[-1]]) ** 2).sum(1)
        )
        nearest[torch.tensor(selected)] = -1
        selected.append(int(torch.argmax(nearest)))
    centroids = normalized[selected].clone()
    assignments = torch.zeros(normalized.shape[0], dtype=torch.long)
    for _ in range(iterations):
        distances = ((normalized[:, None] - centroids[None]) ** 2).sum(2)
        assignments = torch.argmin(distances, 1)
        counts = torch.bincount(assignments, minlength=k)
        if torch.any(counts == 0):
            loss = distances.gather(1, assignments[:, None]).squeeze(1)
            for empty in torch.nonzero(counts == 0).flatten().tolist():
                donor = int(torch.argmax(loss))
                assignments[donor] = empty
                loss[donor] = -1
        updated = []
        for index in range(k):
            value = normalized[assignments == index].mean(0)
            updated.append(value / torch.linalg.vector_norm(value).clamp_min(1e-30))
        new_centroids = torch.stack(updated)
        if torch.equal(new_centroids, centroids):
            break
        centroids = new_centroids
    # Recompute membership against the returned centroids so build routing and
    # compilation use exactly the same deterministic partition.
    distances = ((normalized[:, None] - centroids[None]) ** 2).sum(2)
    assignments = torch.argmin(distances, 1)
    if torch.any(torch.bincount(assignments, minlength=k) < 2):
        raise MicroprogramError("returned centroid partition has a small cluster")
    return assignments, centroids


def _sha(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Program:
    xc64: torch.Tensor
    yc64: torch.Tensor
    v64: torch.Tensor
    c64: torch.Tensor
    xc32: torch.Tensor
    yc32: torch.Tensor
    v32: torch.Tensor
    c32: torch.Tensor
    centroid: torch.Tensor
    build_count: int
    rank: int
    fingerprint: str

    def query(self, x: torch.Tensor, precision: str) -> torch.Tensor:
        if x.ndim == 1:
            x = x[None]
        if (
            x.ndim != 2
            or x.shape[1] != self.xc64.numel()
            or not torch.isfinite(x.float()).all()
        ):
            raise MicroprogramError("invalid query")
        if precision == "fp64":
            output = self.yc64 + ((x.double() - self.xc64) @ self.v64) @ self.c64
        elif precision == "fp32":
            output = self.yc32 + ((x.float() - self.xc32) @ self.v32) @ self.c32
        else:
            raise MicroprogramError("invalid precision")
        return output.to(torch.bfloat16)


def fit_program(
    x: torch.Tensor,
    y: torch.Tensor,
    rank_cap: int,
    centroid: torch.Tensor,
) -> Program:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise MicroprogramError("invalid build pairs")
    x64, y64 = x.double(), y.double()
    x_center, y_center = x64.mean(0), y64.mean(0)
    centered_x, centered_y = x64 - x_center, y64 - y_center
    u, singular, vh = torch.linalg.svd(centered_x, full_matrices=False)
    tolerance = (
        max(centered_x.shape)
        * torch.finfo(torch.float64).eps
        * float(singular[0])
    )
    rank = min(rank_cap, int((singular > tolerance).sum()))
    if rank < 1:
        raise MicroprogramError("zero rank")
    basis = vh[:rank].T.contiguous()
    coefficients = (
        (u[:, :rank].T @ centered_y) / singular[:rank, None]
    ).contiguous()
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "x": _sha(x_center),
                "y": _sha(y_center),
                "v": _sha(basis),
                "c": _sha(coefficients),
                "rank_cap": int(rank_cap),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return Program(
        x_center,
        y_center,
        basis,
        coefficients,
        x_center.float(),
        y_center.float(),
        basis.float(),
        coefficients.float(),
        centroid.double(),
        int(x64.shape[0]),
        rank,
        fingerprint,
    )


@dataclass(frozen=True)
class Library:
    programs: tuple[Program, ...]
    rank_cap: int
    fingerprint: str

    def _predictions(self, x: torch.Tensor, precision: str) -> torch.Tensor:
        return torch.stack([program.query(x, precision) for program in self.programs], 1)

    def query(
        self,
        x: torch.Tensor,
        reference: torch.Tensor,
        precision: str,
        oracle: bool,
    ) -> dict[str, Any]:
        predictions = self._predictions(x, precision)
        if oracle:
            selected = torch.argmax((predictions == reference[:, None]).sum(2), 1)
            mode = "oracle"
        else:
            normalized = norm_rows(x)
            centroids = torch.stack([program.centroid for program in self.programs])
            selected = torch.argmin(
                ((normalized[:, None] - centroids[None]) ** 2).sum(2), 1
            )
            mode = "router"
        candidate = predictions[torch.arange(predictions.shape[0]), selected]
        return {
            "mode": mode,
            "precision": precision,
            "selected": [int(value) for value in selected.tolist()],
            "use_counts": [
                int(value)
                for value in torch.bincount(
                    selected, minlength=len(self.programs)
                ).tolist()
            ],
            "report": exact_report(reference, candidate),
            "candidate": candidate,
        }


def fit_library_from_partition(
    x: torch.Tensor,
    y: torch.Tensor,
    assignments: torch.Tensor,
    centroids: torch.Tensor,
    rank_cap: int,
) -> Library:
    if assignments.ndim != 1 or assignments.shape[0] != x.shape[0]:
        raise MicroprogramError("invalid fixed partition")
    programs = []
    for index in range(int(centroids.shape[0])):
        mask = assignments == index
        if int(mask.sum()) < 2:
            raise MicroprogramError("small cluster")
        programs.append(
            fit_program(x[mask], y[mask], rank_cap, centroids[index])
        )
    fingerprint = hashlib.sha256(
        "|".join(program.fingerprint for program in programs).encode()
    ).hexdigest()
    return Library(tuple(programs), rank_cap, fingerprint)


def fit_library(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rank_cap: int,
) -> tuple[Library, torch.Tensor, torch.Tensor]:
    assignments, centroids = cluster(x, k)
    return (
        fit_library_from_partition(x, y, assignments, centroids, rank_cap),
        assignments,
        centroids,
    )


def target_resources(k: int, rank: int, scalar_bytes: int = 4) -> dict[str, Any]:
    scalars = 3 * H405 + 2 * H405 * rank
    sidecar = k * L405 * scalars * scalar_bytes
    router_ops = H405 + 3 * k * H405
    program_ops = 2 * H405 * rank + H405
    fraction = (router_ops + program_ops) / (3 * H405 * I405)
    whole = 1 - MLP_SHARE + MLP_SHARE * fraction
    return {
        "sidecar_bytes": sidecar,
        "sidecar_gib": sidecar / 1024**3,
        "compiled_mlp_operation_fraction": fraction,
        "whole_model_operation_fraction_if_only_mlp_replaced": whole,
        # Retained alias for old evidence readers.
        "whole_model_fraction_if_only_mlp_replaced": whole,
        "uncompiled_non_mlp_fraction": 1 - MLP_SHARE,
    }


def residual_diagnostics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    xor = torch.bitwise_xor(
        reference.contiguous().view(torch.int16),
        candidate.contiguous().view(torch.int16),
    )
    hashes = [_sha(row) for row in xor]
    unique = [
        int(torch.unique(xor[:, column]).numel())
        for column in range(xor.shape[1])
    ]
    return {
        "unique_residual_vector_fraction": len(set(hashes)) / reference.shape[0],
        "zero_word_fraction": float((xor == 0).float().mean()),
        "unique_words_per_coordinate_p50": pct(unique, 0.50),
        "unique_words_per_coordinate_p95": pct(unique, 0.95),
    }
