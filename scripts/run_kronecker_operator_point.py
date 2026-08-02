from __future__ import annotations

"""Compatibility-free teacher-forced token metrics.

Several structural-compiler experiments originally imported this helper from a
now-rejected Kronecker experiment branch. Keeping the metric in this tiny module
makes those experiments independently reproducible without restoring the
rejected runtime or any of its implementation dependencies.
"""

import torch

from vortex_runtime.candidate_coverage import token_rank


def teacher_summary(
    *,
    logits: torch.Tensor,
    exact_tokens: torch.Tensor,
) -> dict[str, object]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, positions, vocabulary]")
    if exact_tokens.ndim != 2:
        raise ValueError("exact_tokens must have shape [batch, positions]")
    if logits.shape[:2] != exact_tokens.shape:
        raise ValueError("logit and token leading dimensions must match")
    if exact_tokens.shape[0] != 1:
        raise ValueError("teacher_summary currently requires batch size one")

    ranks: list[int] = []
    matches = 0
    for position in range(exact_tokens.shape[1]):
        exact_token = int(exact_tokens[0, position].item())
        position_logits = logits[0, position]
        predicted = int(torch.argmax(position_logits).item())
        matches += int(predicted == exact_token)
        ranks.append(token_rank(position_logits, exact_token))
    if not ranks:
        raise ValueError("at least one token is required")
    return {
        "tokens": len(ranks),
        "top1_rate": matches / len(ranks),
        "top4_rate": sum(rank <= 4 for rank in ranks) / len(ranks),
        "top32_rate": sum(rank <= 32 for rank in ranks) / len(ranks),
        "mean_exact_token_rank": sum(ranks) / len(ranks),
        "maximum_exact_token_rank": max(ranks),
        "ranks": ranks,
    }
