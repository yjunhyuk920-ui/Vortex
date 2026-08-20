from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch


class TrueTokenRankInvariantError(RuntimeError):
    """Fail-closed violation of the frozen one-sweep rank contract."""


@dataclass(frozen=True)
class RankMetrics:
    count: int
    minimum: int
    p50: float
    p95: float
    maximum: int
    mean: float
    top1_fraction: float
    top4_fraction: float
    top16_fraction: float
    static_path_information_bits: float

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


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def stable_true_token_rank(logits: torch.Tensor, true_token: int) -> int:
    """Return one-based rank under descending logit and ascending token-ID ties.

    The ordering exactly matches the frozen `torch.argmax` first-index tie rule
    used by the official reference. The function does not sort the vocabulary;
    it counts strictly better entries and equal entries with a smaller ID.
    """

    values = logits.detach().to(torch.float32).contiguous().cpu()
    if values.ndim != 1 or int(values.numel()) <= 0:
        raise TrueTokenRankInvariantError("logits must be one nonempty row")
    if not torch.isfinite(values).all():
        raise TrueTokenRankInvariantError("logits contain nonfinite values")
    token = int(true_token)
    if token < 0 or token >= int(values.numel()):
        raise TrueTokenRankInvariantError("true token is outside the vocabulary")
    target = values[token]
    strictly_better = int((values > target).sum().item())
    if token:
        earlier_ties = int((values[:token] == target).sum().item())
    else:
        earlier_ties = 0
    rank = 1 + strictly_better + earlier_ties
    if rank < 1 or rank > int(values.numel()):
        raise TrueTokenRankInvariantError("computed rank is outside the vocabulary")
    return rank


def stable_argmax_token(logits: torch.Tensor) -> int:
    values = logits.detach().to(torch.float32).contiguous().cpu()
    if values.ndim != 1 or int(values.numel()) <= 0:
        raise TrueTokenRankInvariantError("logits must be one nonempty row")
    if not torch.isfinite(values).all():
        raise TrueTokenRankInvariantError("logits contain nonfinite values")
    return int(torch.argmax(values).item())


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        raise TrueTokenRankInvariantError("cannot aggregate an empty rank population")
    if not 0.0 <= float(q) <= 100.0:
        raise TrueTokenRankInvariantError("percentile must be in [0, 100]")
    return float(
        np.percentile(
            np.asarray([int(value) for value in values], dtype=np.float64),
            float(q),
            method="linear",
        )
    )


def rank_metrics(ranks: Sequence[int]) -> RankMetrics:
    values = tuple(int(value) for value in ranks)
    if not values or any(value <= 0 for value in values):
        raise TrueTokenRankInvariantError("ranks must be positive and nonempty")
    count = len(values)
    return RankMetrics(
        count=count,
        minimum=min(values),
        p50=percentile(values, 50.0),
        p95=percentile(values, 95.0),
        maximum=max(values),
        mean=float(sum(values) / count),
        top1_fraction=sum(value <= 1 for value in values) / count,
        top4_fraction=sum(value <= 4 for value in values) / count,
        top16_fraction=sum(value <= 16 for value in values) / count,
        static_path_information_bits=float(
            sum(math.log2(float(value)) for value in values)
        ),
    )


def rank_histogram(ranks: Sequence[int]) -> list[dict[str, int]]:
    counts: dict[int, int] = {}
    for raw in ranks:
        value = int(raw)
        if value <= 0:
            raise TrueTokenRankInvariantError("ranks must be positive")
        counts[value] = counts.get(value, 0) + 1
    return [{"rank": rank, "count": counts[rank]} for rank in sorted(counts)]


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    result = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        result += 1
    return result


def static_topk_gate(
    ranks: Sequence[int], *, p95_limit: int, maximum_limit: int
) -> dict[str, Any]:
    if int(p95_limit) <= 0 or int(maximum_limit) < int(p95_limit):
        raise TrueTokenRankInvariantError("invalid static top-k thresholds")
    metrics = rank_metrics(ranks)
    return {
        "p95_limit": int(p95_limit),
        "maximum_limit": int(maximum_limit),
        "p95_passed": metrics.p95 <= int(p95_limit),
        "maximum_passed": metrics.maximum <= int(maximum_limit),
        "passed": (
            metrics.p95 <= int(p95_limit)
            and metrics.maximum <= int(maximum_limit)
        ),
        "metrics": metrics.to_dict(),
    }


def candidate_id_bytes(
    *, block_length: int, maximum_candidates: int, token_bytes: int = 2
) -> int:
    length = int(block_length)
    candidates = int(maximum_candidates)
    width = int(token_bytes)
    if length <= 0 or candidates <= 0 or width <= 0:
        raise TrueTokenRankInvariantError("candidate dimensions must be positive")
    return length * candidates * width


def model_free_controls() -> dict[str, Any]:
    tied = torch.tensor([7.0, 7.0, 6.0, 7.0], dtype=torch.float32)
    ranks = [stable_true_token_rank(tied, token) for token in range(4)]
    order_control = ranks == [1, 2, 4, 3]

    logits = torch.tensor([1.0, 4.0, 3.0, 2.0], dtype=torch.float32)
    strict_control = [stable_true_token_rank(logits, token) for token in range(4)]
    strict_passed = strict_control == [4, 1, 2, 3]

    positive = static_topk_gate(
        [1] * 96 + [4] * 26 + [16] * 6,
        p95_limit=4,
        maximum_limit=16,
    )
    negative = static_topk_gate(
        [1] * 122 + [17] * 6,
        p95_limit=4,
        maximum_limit=16,
    )
    candidate_bytes = candidate_id_bytes(
        block_length=128, maximum_candidates=16, token_bytes=2
    )
    passed = all(
        [
            order_control,
            strict_passed,
            positive["passed"],
            not negative["passed"],
            negative["p95_passed"],
            not negative["maximum_passed"],
            candidate_bytes == 4096,
        ]
    )
    return {
        "tied_rank_sequence": ranks,
        "strict_rank_sequence": strict_control,
        "positive_gate": positive,
        "negative_gate": negative,
        "candidate_id_bytes": candidate_bytes,
        "passed": passed,
    }


def flatten_rank_rows(rows: Iterable[Sequence[int]]) -> tuple[int, ...]:
    result: list[int] = []
    for row in rows:
        result.extend(int(value) for value in row)
    if not result:
        raise TrueTokenRankInvariantError("rank-row population is empty")
    return tuple(result)
