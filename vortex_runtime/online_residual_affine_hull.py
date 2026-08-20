from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


class ResidualAffineHullInvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class AffineTransportResult:
    scores: torch.Tensor
    neighbor_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    coefficient_sum: float
    coefficient_abs_max: float
    reconstruction_l2: float


@dataclass(frozen=True)
class SpanProjectionReport:
    projected_residual: torch.Tensor
    coefficients: torch.Tensor
    effective_rank: int
    maximum_eigenvalue: float
    minimum_retained_eigenvalue: float
    relative_cutoff: float
    gram_symmetry_max_abs: float
    target_residual_l2: float
    projection_residual_l2: float
    projection_relative_l2: float

    def summary(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("projected_residual")
        value.pop("coefficients")
        return value


@dataclass(frozen=True)
class RankSummary:
    count: int
    p50: float
    p95: float
    maximum: int
    top1_fraction: float
    top4_fraction: float
    top16_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _finite_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ResidualAffineHullInvariantError(
            f"{name} must be rank two, got {tuple(value.shape)}"
        )
    result = value.detach().to(torch.float64).contiguous().cpu()
    if not bool(torch.isfinite(result).all().item()):
        raise ResidualAffineHullInvariantError(f"{name} contains nonfinite values")
    return result


def normalize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _finite_matrix("row matrix", value)
    norms = torch.linalg.vector_norm(matrix, ord=2, dim=1)
    if bool((norms <= 0).any().item()):
        raise ResidualAffineHullInvariantError("row normalization encountered zero norm")
    return matrix / norms.unsqueeze(1), norms


def stable_nearest_indices(
    hidden_bank: torch.Tensor, query_hidden: torch.Tensor, count: int
) -> tuple[int, ...]:
    bank, _ = normalize_rows(hidden_bank)
    query = query_hidden.detach().to(torch.float64).reshape(1, -1).contiguous().cpu()
    if query.shape[1] != bank.shape[1]:
        raise ResidualAffineHullInvariantError("hidden/query widths differ")
    query_norm = torch.linalg.vector_norm(query, ord=2, dim=1)
    if float(query_norm.item()) <= 0.0:
        raise ResidualAffineHullInvariantError("query hidden has zero norm")
    query = query / query_norm.unsqueeze(1)
    k = int(count)
    if k <= 0 or k > int(bank.shape[0]):
        raise ResidualAffineHullInvariantError(
            f"invalid neighbor count {k} for {bank.shape[0]} rows"
        )
    distances = torch.sum((bank - query) ** 2, dim=1)
    # Python tuple sorting makes the tie rule explicit and independent of backend.
    ordered = sorted(
        range(int(distances.numel())),
        key=lambda index: (float(distances[index].item()), int(index)),
    )
    return tuple(int(index) for index in ordered[:k])


def affine_ridge_weights(
    selected_hidden: torch.Tensor,
    query_hidden: torch.Tensor,
    *,
    ridge_lambda: float,
    coefficient_clip: float,
) -> tuple[torch.Tensor, float]:
    selected, _ = normalize_rows(selected_hidden)
    query = query_hidden.detach().to(torch.float64).reshape(-1).contiguous().cpu()
    if query.numel() != selected.shape[1]:
        raise ResidualAffineHullInvariantError("selected/query widths differ")
    query_norm = torch.linalg.vector_norm(query, ord=2)
    if float(query_norm.item()) <= 0.0:
        raise ResidualAffineHullInvariantError("query hidden has zero norm")
    query = query / query_norm
    ridge = float(ridge_lambda)
    clip = float(coefficient_clip)
    if ridge <= 0.0 or clip <= 0.0:
        raise ResidualAffineHullInvariantError("ridge and clip must be positive")

    k = int(selected.shape[0])
    gram = selected @ selected.T
    rhs = selected @ query
    kkt = torch.zeros((k + 1, k + 1), dtype=torch.float64)
    kkt[:k, :k] = gram + ridge * torch.eye(k, dtype=torch.float64)
    kkt[:k, k] = 1.0
    kkt[k, :k] = 1.0
    target = torch.cat((rhs, torch.ones(1, dtype=torch.float64)))
    try:
        solution = torch.linalg.solve(kkt, target)
    except RuntimeError as exc:
        raise ResidualAffineHullInvariantError(
            f"affine ridge KKT solve failed: {exc}"
        ) from exc
    weights = torch.clamp(solution[:k], min=-clip, max=clip)
    total = torch.sum(weights)
    if not bool(torch.isfinite(weights).all().item()) or abs(float(total.item())) < 1e-15:
        raise ResidualAffineHullInvariantError("affine ridge produced invalid weights")
    weights = weights / total
    reconstruction = weights @ selected
    reconstruction_l2 = float(torch.linalg.vector_norm(reconstruction - query).item())
    return weights, reconstruction_l2


def causal_affine_transport(
    *,
    coarse_logits: torch.Tensor,
    residual_bank: torch.Tensor,
    hidden_bank: torch.Tensor,
    query_hidden: torch.Tensor,
    neighbor_count: int,
    ridge_lambda: float,
    coefficient_clip: float,
) -> AffineTransportResult:
    bank = _finite_matrix("residual bank", residual_bank)
    hidden = _finite_matrix("hidden bank", hidden_bank)
    if bank.shape[0] != hidden.shape[0]:
        raise ResidualAffineHullInvariantError("residual and hidden row counts differ")
    coarse = coarse_logits.detach().to(torch.float64).reshape(-1).contiguous().cpu()
    if coarse.numel() != bank.shape[1]:
        raise ResidualAffineHullInvariantError("coarse/residual vocabulary widths differ")
    neighbors = stable_nearest_indices(hidden, query_hidden, int(neighbor_count))
    selected_hidden = hidden[list(neighbors)]
    weights, reconstruction_l2 = affine_ridge_weights(
        selected_hidden,
        query_hidden,
        ridge_lambda=float(ridge_lambda),
        coefficient_clip=float(coefficient_clip),
    )
    selected_residual = bank[list(neighbors)]
    scores = (coarse + weights @ selected_residual).contiguous()
    if not bool(torch.isfinite(scores).all().item()):
        raise ResidualAffineHullInvariantError("causal affine scores are nonfinite")
    return AffineTransportResult(
        scores=scores,
        neighbor_indices=neighbors,
        coefficients=tuple(float(value) for value in weights.tolist()),
        coefficient_sum=float(torch.sum(weights).item()),
        coefficient_abs_max=float(torch.max(torch.abs(weights)).item()),
        reconstruction_l2=reconstruction_l2,
    )


def stable_argmax64(scores: torch.Tensor) -> int:
    vector = scores.detach().to(torch.float64).reshape(-1).contiguous().cpu()
    if not bool(torch.isfinite(vector).all().item()):
        raise ResidualAffineHullInvariantError("argmax scores are nonfinite")
    # torch.argmax returns the first index on a tie.
    return int(torch.argmax(vector).item())


def stable_true_token_rank(scores: torch.Tensor, token_id: int) -> int:
    vector = scores.detach().to(torch.float64).reshape(-1).contiguous().cpu()
    token = int(token_id)
    if token < 0 or token >= int(vector.numel()):
        raise ResidualAffineHullInvariantError("token ID outside score vector")
    if not bool(torch.isfinite(vector).all().item()):
        raise ResidualAffineHullInvariantError("rank scores are nonfinite")
    target = vector[token]
    greater = int(torch.sum(vector > target).item())
    earlier_equal = int(torch.sum(vector[:token] == target).item())
    return 1 + greater + earlier_equal


def best_single_residual_rank(
    coarse_logits: torch.Tensor,
    residual_bank: torch.Tensor,
    token_id: int,
) -> tuple[int, int]:
    bank = _finite_matrix("residual bank", residual_bank)
    coarse = coarse_logits.detach().to(torch.float64).reshape(-1).contiguous().cpu()
    if coarse.numel() != bank.shape[1]:
        raise ResidualAffineHullInvariantError("coarse/residual widths differ")
    best_rank: int | None = None
    best_index = -1
    for index in range(int(bank.shape[0])):
        rank = stable_true_token_rank(coarse + bank[index], int(token_id))
        if best_rank is None or (rank, index) < (best_rank, best_index):
            best_rank = rank
            best_index = index
    if best_rank is None:
        raise ResidualAffineHullInvariantError("empty residual bank")
    return int(best_rank), int(best_index)


def residual_span_projection(
    residual_bank: torch.Tensor,
    target_residual: torch.Tensor,
    *,
    relative_eigenvalue_cutoff: float,
) -> SpanProjectionReport:
    bank = _finite_matrix("residual bank", residual_bank)
    target = _finite_matrix("target residual", target_residual)
    if bank.shape[1] != target.shape[1]:
        raise ResidualAffineHullInvariantError("bank/target vocabulary widths differ")
    cutoff = float(relative_eigenvalue_cutoff)
    if not 0.0 < cutoff < 1.0:
        raise ResidualAffineHullInvariantError("relative cutoff must be in (0,1)")

    gram_raw = bank @ bank.T
    symmetry = float(torch.max(torch.abs(gram_raw - gram_raw.T)).item())
    gram = (gram_raw + gram_raw.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = float(torch.max(eigenvalues).item())
    if not maximum > 0.0:
        raise ResidualAffineHullInvariantError("residual bank has zero Gram spectrum")
    retained = eigenvalues > maximum * cutoff
    rank = int(torch.sum(retained).item())
    if rank <= 0:
        raise ResidualAffineHullInvariantError("eigenvalue cutoff removed the whole span")
    u = eigenvectors[:, retained]
    values = eigenvalues[retained]
    gram_pinv = (u / values.unsqueeze(0)) @ u.T
    coefficients = (target @ bank.T) @ gram_pinv
    projected = (coefficients @ bank).contiguous()
    if not bool(torch.isfinite(projected).all().item()):
        raise ResidualAffineHullInvariantError("span projection is nonfinite")
    target_l2 = float(torch.linalg.vector_norm(target).item())
    residual_l2 = float(torch.linalg.vector_norm(target - projected).item())
    relative = residual_l2 / target_l2 if target_l2 > 0.0 else 0.0
    return SpanProjectionReport(
        projected_residual=projected,
        coefficients=coefficients,
        effective_rank=rank,
        maximum_eigenvalue=maximum,
        minimum_retained_eigenvalue=float(torch.min(values).item()),
        relative_cutoff=cutoff,
        gram_symmetry_max_abs=symmetry,
        target_residual_l2=target_l2,
        projection_residual_l2=residual_l2,
        projection_relative_l2=relative,
    )


def summarize_ranks(ranks: Sequence[int]) -> RankSummary:
    values = [int(value) for value in ranks]
    if not values:
        raise ResidualAffineHullInvariantError("cannot summarize empty ranks")
    tensor = torch.tensor(values, dtype=torch.float64)
    return RankSummary(
        count=len(values),
        p50=float(torch.quantile(tensor, 0.50, interpolation="linear").item()),
        p95=float(torch.quantile(tensor, 0.95, interpolation="linear").item()),
        maximum=max(values),
        top1_fraction=sum(value <= 1 for value in values) / len(values),
        top4_fraction=sum(value <= 4 for value in values) / len(values),
        top16_fraction=sum(value <= 16 for value in values) / len(values),
    )


def accepted_prefix(candidate: Sequence[int], target: Sequence[int]) -> int:
    count = 0
    for left, right in zip(candidate, target):
        if int(left) != int(right):
            break
        count += 1
    return count


def project_target_hot_bytes(
    *,
    inherited_hot_bytes: int,
    block_length: int,
    vocabulary_size: int,
    hidden_size: int,
) -> dict[str, Any]:
    k = int(block_length)
    residual = k * int(vocabulary_size) * 8
    hidden = k * int(hidden_size) * 8
    gram = k * k * 8
    total = int(inherited_hot_bytes) + hidden + gram
    # inherited_hot_bytes already includes the float64 residual block from EXP-094A.
    return {
        "inherited_hot_bytes_including_residual_bank": int(inherited_hot_bytes),
        "float64_residual_bank_bytes": residual,
        "float64_hidden_bank_bytes": hidden,
        "float64_gram_bytes": gram,
        "total_hot_bytes": total,
        "total_hot_gib": total / float(1 << 30),
    }


def model_free_controls() -> dict[str, Any]:
    torch.manual_seed(95)
    hidden = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float64,
    )
    residual = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [2.0, 3.0, 0.0], [-2.0, 3.0, -2.0]],
        dtype=torch.float64,
    )
    query = torch.tensor([1.0, 1.0], dtype=torch.float64)
    causal = causal_affine_transport(
        coarse_logits=torch.zeros(3, dtype=torch.float64),
        residual_bank=residual,
        hidden_bank=hidden,
        query_hidden=query,
        neighbor_count=4,
        ridge_lambda=2.0**-10,
        coefficient_clip=4.0,
    )
    affine_finite = bool(torch.isfinite(causal.scores).all().item())
    affine_sum = abs(causal.coefficient_sum - 1.0) <= 1e-12

    bank = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target_in = torch.tensor(
        [[2.0, -3.0, 2.0, -3.0]], dtype=torch.float64
    )
    positive = residual_span_projection(
        bank, target_in, relative_eigenvalue_cutoff=2.0**-40
    )
    positive_exact = bool(torch.equal(positive.projected_residual, target_in))

    target_out = torch.tensor(
        [[0.0, 0.0, 1.0, -1.0]], dtype=torch.float64
    )
    negative = residual_span_projection(
        bank, target_out, relative_eigenvalue_cutoff=2.0**-40
    )
    negative_detected = float(negative.projection_residual_l2) > 0.0

    tie_scores = torch.tensor([4.0, 4.0, 3.0], dtype=torch.float64)
    stable_tie = stable_argmax64(tie_scores) == 0 and stable_true_token_rank(
        tie_scores, 1
    ) == 2
    best_rank, best_index = best_single_residual_rank(
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
        1,
    )
    single_oracle = best_rank == 1 and best_index == 0

    passed = all(
        [
            affine_finite,
            affine_sum,
            positive_exact,
            negative_detected,
            stable_tie,
            single_oracle,
        ]
    )
    return {
        "passed": passed,
        "affine_finite": affine_finite,
        "affine_coefficient_sum": affine_sum,
        "span_positive_exact": positive_exact,
        "span_orthogonal_negative_detected": negative_detected,
        "stable_tie_rule": stable_tie,
        "single_residual_oracle": single_oracle,
        "positive_projection_sha256": tensor_sha256(positive.projected_residual),
        "negative_projection_sha256": tensor_sha256(negative.projected_residual),
    }
