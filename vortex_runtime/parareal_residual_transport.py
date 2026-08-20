from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from vortex_runtime.one_sweep_true_token_rank import rank_metrics


class PararealTransportInvariantError(RuntimeError):
    """Fail-closed violation of the frozen residual-transport contract."""


@dataclass(frozen=True)
class TransportMetrics:
    accepted_prefix: int
    exact_positions: int
    exact_fraction: float
    rank_p50: float
    rank_p95: float
    rank_maximum: int
    top1_fraction: float
    top4_fraction: float
    top16_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row64(value: torch.Tensor, name: str) -> torch.Tensor:
    row = value.detach().to(torch.float64).contiguous().cpu()
    if row.ndim != 1 or int(row.numel()) <= 0:
        raise PararealTransportInvariantError(f"{name} must be one nonempty row")
    if not torch.isfinite(row).all():
        raise PararealTransportInvariantError(f"{name} contains nonfinite values")
    return row


def exact_residual(fine_old: torch.Tensor, coarse_old: torch.Tensor) -> torch.Tensor:
    fine = _row64(fine_old, "fine_old")
    coarse = _row64(coarse_old, "coarse_old")
    if fine.shape != coarse.shape:
        raise PararealTransportInvariantError("fine/coarse vocabulary shapes differ")
    return (fine - coarse).contiguous()


def transport_scores(
    fine_old: torch.Tensor,
    coarse_old: torch.Tensor,
    coarse_new: torch.Tensor,
) -> torch.Tensor:
    """Apply one Parareal logit correction in exact float64 working arithmetic.

    P(new; old) = G(new) + F(old) - G(old)

    All three inputs are finite-precision values produced by the frozen source
    graph. Promoting them to float64 makes the subtraction/addition exact for
    the represented BF16/FP32 values and gives the mechanism its most favorable
    finite-word interpretation.
    """

    fine = _row64(fine_old, "fine_old")
    old = _row64(coarse_old, "coarse_old")
    new = _row64(coarse_new, "coarse_new")
    if fine.shape != old.shape or fine.shape != new.shape:
        raise PararealTransportInvariantError("transport vocabulary shapes differ")
    return (new + (fine - old)).contiguous()


def stable_argmax64(scores: torch.Tensor) -> int:
    values = _row64(scores, "scores")
    # torch.argmax returns the first maximum and therefore preserves the frozen
    # ascending-token-ID tie rule without narrowing the float64 transport.
    return int(torch.argmax(values).item())


def rank_true_token(scores: torch.Tensor, true_token: int) -> int:
    values = _row64(scores, "scores")
    token = int(true_token)
    if token < 0 or token >= int(values.numel()):
        raise PararealTransportInvariantError("true token outside score vector")
    target = values[token]
    strictly_better = int((values > target).sum().item())
    earlier_ties = int((values[:token] == target).sum().item()) if token else 0
    return 1 + strictly_better + earlier_ties


def accepted_prefix(candidate: Sequence[int], target: Sequence[int]) -> int:
    if len(candidate) != len(target):
        raise PararealTransportInvariantError("candidate/target lengths differ")
    result = 0
    for lhs, rhs in zip(candidate, target):
        if int(lhs) != int(rhs):
            break
        result += 1
    return result


def summarize_transport(
    candidate: Sequence[int], target: Sequence[int], true_path_ranks: Sequence[int]
) -> TransportMetrics:
    if len(candidate) != len(target) or len(target) != len(true_path_ranks):
        raise PararealTransportInvariantError("transport populations differ")
    if not target:
        raise PararealTransportInvariantError("transport population is empty")
    exact = sum(int(lhs) == int(rhs) for lhs, rhs in zip(candidate, target))
    ranks = rank_metrics([int(value) for value in true_path_ranks])
    return TransportMetrics(
        accepted_prefix=accepted_prefix(candidate, target),
        exact_positions=exact,
        exact_fraction=exact / len(target),
        rank_p50=ranks.p50,
        rank_p95=ranks.p95,
        rank_maximum=ranks.maximum,
        top1_fraction=ranks.top1_fraction,
        top4_fraction=ranks.top4_fraction,
        top16_fraction=ranks.top16_fraction,
    )


def residual_buffer_bytes(
    *, block_length: int, vocabulary_size: int, value_bytes: int = 8
) -> int:
    length = int(block_length)
    vocab = int(vocabulary_size)
    width = int(value_bytes)
    if length <= 0 or vocab <= 0 or width <= 0:
        raise PararealTransportInvariantError("residual dimensions must be positive")
    return length * vocab * width


def logical_source_fraction(
    accepted_tokens: int,
    *,
    fine_sweeps: int = 1,
    compression_ratio: float = 1.0,
) -> float:
    accepted = int(accepted_tokens)
    sweeps = int(fine_sweeps)
    ratio = float(compression_ratio)
    if accepted <= 0 or sweeps <= 0 or not math.isfinite(ratio) or ratio < 1.0:
        return math.inf
    return sweeps / (accepted * ratio)


def transport_argmax(
    fine_old: torch.Tensor, coarse_old: torch.Tensor, coarse_new: torch.Tensor
) -> int:
    return stable_argmax64(transport_scores(fine_old, coarse_old, coarse_new))


def model_free_controls() -> dict[str, Any]:
    fine = torch.tensor([9.0, 4.0, -3.0, 1.0], dtype=torch.float32)
    coarse_old = torch.tensor([1.5, 2.0, 0.0, -1.0], dtype=torch.float32)
    recovered = transport_scores(fine, coarse_old, coarse_old)
    recovery_passed = torch.equal(recovered, fine.to(torch.float64))

    coarse_new = torch.tensor([1.5, 12.0, 0.0, -1.0], dtype=torch.float32)
    shifted = transport_scores(fine, coarse_old, coarse_new)
    shifted_expected = torch.tensor([9.0, 14.0, -3.0, 1.0], dtype=torch.float64)
    shift_passed = torch.equal(shifted, shifted_expected)
    argmax_passed = stable_argmax64(shifted) == 1
    rank_passed = rank_true_token(shifted, 0) == 2

    candidate = [4, 5, 6, 9]
    target = [4, 5, 7, 9]
    metrics = summarize_transport(candidate, target, [1, 1, 2, 1])
    summary_passed = (
        metrics.accepted_prefix == 2
        and metrics.exact_positions == 3
        and metrics.rank_maximum == 2
    )
    bytes_control = residual_buffer_bytes(
        block_length=128, vocabulary_size=49_152, value_bytes=8
    )
    traffic_control = logical_source_fraction(
        128, fine_sweeps=1, compression_ratio=1.261972
    )
    passed = all(
        [
            recovery_passed,
            shift_passed,
            argmax_passed,
            rank_passed,
            summary_passed,
            bytes_control == 50_331_648,
            abs(traffic_control - 0.006190707876244481) <= 1e-15,
        ]
    )
    return {
        "recovery_passed": recovery_passed,
        "shift_passed": shift_passed,
        "argmax_passed": argmax_passed,
        "rank_passed": rank_passed,
        "summary": metrics.to_dict(),
        "dev_residual_buffer_bytes": bytes_control,
        "compressed_source_fraction_at_128": traffic_control,
        "passed": passed,
    }
