"""Reference contracts for intermediate-layer token finalization.

EXP-051 uses exact target prefixes and asks when the current next-token decision
becomes equal to, and remains equal to, the full-depth target token. Oracle
finalization depths use later layer outputs and are explicitly non-deployable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class LayerFinalizationError(ValueError):
    """Raised when a layer probe or byte contract is invalid."""


def first_match_depth(tokens: Sequence[int]) -> int:
    if not tokens:
        raise LayerFinalizationError("tokens must not be empty")
    final = int(tokens[-1])
    for depth, token in enumerate(tokens):
        if int(token) == final:
            return depth
    raise LayerFinalizationError("final token did not match itself")


def suffix_stable_depth(tokens: Sequence[int]) -> int:
    if not tokens:
        raise LayerFinalizationError("tokens must not be empty")
    final = int(tokens[-1])
    depth = len(tokens) - 1
    while depth > 0 and int(tokens[depth - 1]) == final:
        depth -= 1
    return depth


def token_changes(tokens: Sequence[int]) -> int:
    if not tokens:
        raise LayerFinalizationError("tokens must not be empty")
    return sum(int(left) != int(right) for left, right in zip(tokens, tokens[1:]))


def post_first_match_wrong_depths(tokens: Sequence[int]) -> int:
    first = first_match_depth(tokens)
    final = int(tokens[-1])
    return sum(int(token) != final for token in tokens[first + 1 :])


def fixed_depths(block_count: int) -> tuple[int, ...]:
    if block_count <= 0:
        raise LayerFinalizationError("block_count must be positive")
    fractions = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    depths = {
        min(block_count, max(0, math.ceil(fraction * block_count)))
        for fraction in fractions
    }
    depths.add(block_count)
    return tuple(sorted(depths))


@dataclass(frozen=True)
class LayerTraffic:
    block_parameter_bytes: tuple[int, ...]
    embedding_row_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int

    def validate(self) -> None:
        if not self.block_parameter_bytes:
            raise LayerFinalizationError("at least one block is required")
        if any(value <= 0 for value in self.block_parameter_bytes):
            raise LayerFinalizationError("block bytes must be positive")
        if self.embedding_row_bytes < 0:
            raise LayerFinalizationError("embedding row bytes must be non-negative")
        if self.final_norm_bytes <= 0 or self.lm_head_bytes <= 0:
            raise LayerFinalizationError("norm/head bytes must be positive")

    @property
    def block_count(self) -> int:
        return len(self.block_parameter_bytes)

    @property
    def full_logical_bytes(self) -> int:
        self.validate()
        return (
            self.embedding_row_bytes
            + sum(self.block_parameter_bytes)
            + self.final_norm_bytes
            + self.lm_head_bytes
        )

    @property
    def lm_head_fraction(self) -> float:
        return self.lm_head_bytes / self.full_logical_bytes

    def bytes_at_depth(self, depth: int) -> int:
        self.validate()
        if not 0 <= depth <= self.block_count:
            raise LayerFinalizationError("depth outside [0, block_count]")
        return (
            self.embedding_row_bytes
            + sum(self.block_parameter_bytes[:depth])
            + self.final_norm_bytes
            + self.lm_head_bytes
        )

    def fraction_at_depth(self, depth: int) -> float:
        return self.bytes_at_depth(depth) / self.full_logical_bytes


@dataclass(frozen=True)
class LayerProbeAnalysis:
    intermediate_tokens: tuple[int, ...]
    margins: tuple[float, ...]
    first_match_depth: int
    suffix_stable_depth: int
    first_match_block_fraction: float
    suffix_stable_block_fraction: float
    suffix_stable_logical_byte_fraction: float
    post_first_match_wrong_depths: int
    token_changes: int
    final_token: int


def analyze_layer_probe(
    *,
    intermediate_tokens: Sequence[int],
    margins: Sequence[float],
    traffic: LayerTraffic,
) -> LayerProbeAnalysis:
    traffic.validate()
    tokens = tuple(int(token) for token in intermediate_tokens)
    margin_values = tuple(float(value) for value in margins)
    expected = traffic.block_count + 1
    if len(tokens) != expected or len(margin_values) != expected:
        raise LayerFinalizationError(
            f"expected {expected} depth probes, got {len(tokens)} tokens and {len(margin_values)} margins"
        )
    if any(token < 0 for token in tokens):
        raise LayerFinalizationError("tokens must be non-negative")
    if any(not math.isfinite(value) for value in margin_values):
        raise LayerFinalizationError("margins contain NaN or Inf")
    first = first_match_depth(tokens)
    stable = suffix_stable_depth(tokens)
    if first > stable:
        raise LayerFinalizationError("first match cannot follow stable depth")
    return LayerProbeAnalysis(
        intermediate_tokens=tokens,
        margins=margin_values,
        first_match_depth=first,
        suffix_stable_depth=stable,
        first_match_block_fraction=first / traffic.block_count,
        suffix_stable_block_fraction=stable / traffic.block_count,
        suffix_stable_logical_byte_fraction=traffic.fraction_at_depth(stable),
        post_first_match_wrong_depths=post_first_match_wrong_depths(tokens),
        token_changes=token_changes(tokens),
        final_token=tokens[-1],
    )


@dataclass(frozen=True)
class LateDecisionResidualChain:
    """Two-token residual chain whose exact decision flips only at final layer."""

    block_count: int
    early_token: int = 0
    final_token: int = 1

    def probe(self) -> tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[float, float], ...]]:
        if self.block_count <= 0:
            raise LayerFinalizationError("block_count must be positive")
        if self.early_token == self.final_token:
            raise LayerFinalizationError("early and final token must differ")
        if min(self.early_token, self.final_token) < 0:
            raise LayerFinalizationError("tokens must be non-negative")

        # Identity two-row output head. Initial hidden [2, 0] favors token 0.
        # Every early residual is zero. The final residual [-4, +3] produces
        # hidden [-2, 3], flipping the exact decision only at the final block.
        states: list[tuple[float, float]] = [(2.0, 0.0)]
        for _ in range(self.block_count - 1):
            states.append(states[-1])
        states.append((-2.0, 3.0))
        tokens: list[int] = []
        margins: list[float] = []
        for left, right in states:
            if left >= right:
                tokens.append(self.early_token)
                margins.append(left - right)
            else:
                tokens.append(self.final_token)
                margins.append(right - left)
        return tuple(tokens), tuple(margins), tuple(states)
