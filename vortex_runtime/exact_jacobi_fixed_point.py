from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch


class JacobiInvariantError(RuntimeError):
    """Fail-closed violation of the frozen exact-Jacobi contract."""


@dataclass(frozen=True)
class SeedManifest:
    mode: str
    block_length: int
    prompt_length: int
    boundary_token: int
    token_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeedSelection:
    mode: str
    minimum_first_sweep_accepted: int
    total_first_sweep_accepted: int
    minimum_best_accepted: int
    total_best_accepted: int
    worst_fixed_point_sweep: int | None
    mode_order: int

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


def validate_tokens(tokens: Sequence[int], *, allow_empty: bool = False) -> tuple[int, ...]:
    values = tuple(int(value) for value in tokens)
    if not values and not allow_empty:
        raise JacobiInvariantError("token sequence is empty")
    if any(value < 0 for value in values):
        raise JacobiInvariantError("token IDs must be nonnegative")
    return values


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        length += 1
    return length


def stable_argmax_rows(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise JacobiInvariantError("logits must have at least two dimensions")
    values = logits.detach().to(torch.float32)
    if not torch.isfinite(values).all():
        raise JacobiInvariantError("logits contain nonfinite values")
    # torch.argmax returns the first maximum, which is the frozen token-ID tie rule.
    return torch.argmax(values, dim=-1).to(torch.int64).contiguous().cpu()


def _boundary_repeat(boundary_token: int, block_length: int) -> list[int]:
    return [int(boundary_token)] * int(block_length)


def _prompt_suffix_cycle(prompt_ids: Sequence[int], block_length: int, width: int) -> list[int]:
    prompt = validate_tokens(prompt_ids)
    count = min(int(width), len(prompt))
    source = list(prompt[-count:])
    return [source[index % len(source)] for index in range(int(block_length))]


def _prompt_longest_ngram(
    prompt_ids: Sequence[int],
    boundary_token: int,
    block_length: int,
    maximum_order: int,
) -> list[int]:
    prompt = list(validate_tokens(prompt_ids))
    boundary = int(boundary_token)
    if boundary < 0:
        raise JacobiInvariantError("boundary token must be nonnegative")
    database = prompt + [boundary]
    context = list(database)
    guesses: list[int] = []

    for _ in range(int(block_length)):
        successor: int | None = None
        max_order = min(int(maximum_order), len(context), max(0, len(database) - 1))
        for order in range(max_order, 0, -1):
            suffix = context[-order:]
            # Search from the latest committed occurrence backward. The final
            # database position cannot begin a match because it has no successor.
            for start in range(len(database) - order - 1, -1, -1):
                if database[start : start + order] == suffix:
                    successor = int(database[start + order])
                    break
            if successor is not None:
                break
        if successor is None:
            successor = boundary
        guesses.append(successor)
        context.append(successor)
    return guesses


def build_seed(
    mode: str,
    *,
    prompt_ids: Sequence[int],
    boundary_token: int,
    block_length: int,
) -> tuple[tuple[int, ...], SeedManifest]:
    prompt = validate_tokens(prompt_ids)
    boundary = int(boundary_token)
    length = int(block_length)
    if boundary < 0 or length <= 0:
        raise JacobiInvariantError("invalid boundary token or block length")

    if mode == "boundary_repeat":
        values = _boundary_repeat(boundary, length)
    elif mode == "prompt_suffix_cycle_16":
        values = _prompt_suffix_cycle(prompt, length, 16)
    elif mode == "prompt_longest_ngram_8":
        values = _prompt_longest_ngram(prompt, boundary, length, 8)
    else:
        raise JacobiInvariantError(f"unsupported seed mode: {mode}")

    seed = tuple(int(value) for value in values)
    if len(seed) != length or any(value < 0 for value in seed):
        raise JacobiInvariantError("seed construction violated its length/token contract")
    manifest = SeedManifest(
        mode=str(mode),
        block_length=length,
        prompt_length=len(prompt),
        boundary_token=boundary,
        token_sha256=canonical_sha256(seed),
    )
    return seed, manifest


def jacobi_input(boundary_token: int, guess: Sequence[int]) -> tuple[int, ...]:
    values = validate_tokens(guess)
    boundary = int(boundary_token)
    if boundary < 0:
        raise JacobiInvariantError("boundary token must be nonnegative")
    return (boundary, *values[:-1])


def self_consistent_prefix(guess: Sequence[int], proposal: Sequence[int]) -> int:
    left = validate_tokens(guess)
    right = validate_tokens(proposal)
    if len(left) != len(right):
        raise JacobiInvariantError("guess/proposal lengths differ")
    return longest_common_prefix(left, right)


def changed_positions(guess: Sequence[int], proposal: Sequence[int]) -> int:
    left = validate_tokens(guess)
    right = validate_tokens(proposal)
    if len(left) != len(right):
        raise JacobiInvariantError("guess/proposal lengths differ")
    return sum(int(lhs != rhs) for lhs, rhs in zip(left, right))


def required_tokens_per_sweep(
    whole_model_fraction_limit: float,
    *,
    compression_ratio: float = 1.0,
) -> int:
    fraction = float(whole_model_fraction_limit)
    ratio = float(compression_ratio)
    if (
        not math.isfinite(fraction)
        or fraction <= 0.0
        or fraction > 1.0
        or not math.isfinite(ratio)
        or ratio < 1.0
    ):
        raise JacobiInvariantError("invalid traffic fraction or compression ratio")
    return int(math.ceil(1.0 / (fraction * ratio)))


def logical_weight_fraction(
    sweeps: int,
    accepted_tokens: int,
    *,
    compression_ratio: float = 1.0,
) -> float:
    sweep_count = int(sweeps)
    accepted = int(accepted_tokens)
    ratio = float(compression_ratio)
    if sweep_count <= 0 or accepted <= 0 or not math.isfinite(ratio) or ratio < 1.0:
        return math.inf
    return sweep_count / (accepted * ratio)


def select_seed_mode(
    rows: Sequence[Mapping[str, Any]], mode_order: Sequence[str]
) -> SeedSelection:
    if not rows or not mode_order:
        raise JacobiInvariantError("selection population/order is empty")
    order = {str(mode): index for index, mode in enumerate(mode_order)}
    seen = {str(row["mode"]) for row in rows}
    missing = [mode for mode in mode_order if mode not in seen]
    if missing:
        raise JacobiInvariantError(f"missing build rows for seed modes: {missing}")

    selections: list[SeedSelection] = []
    for mode in mode_order:
        selected = [row for row in rows if str(row["mode"]) == str(mode)]
        first = [int(row["first_sweep_accepted"]) for row in selected]
        best = [int(row["best_accepted"]) for row in selected]
        fixed = [row.get("fixed_point_sweep") for row in selected]
        worst_fixed: int | None
        if all(value is not None for value in fixed):
            worst_fixed = max(int(value) for value in fixed)
        else:
            worst_fixed = None
        selections.append(
            SeedSelection(
                mode=str(mode),
                minimum_first_sweep_accepted=min(first),
                total_first_sweep_accepted=sum(first),
                minimum_best_accepted=min(best),
                total_best_accepted=sum(best),
                worst_fixed_point_sweep=worst_fixed,
                mode_order=int(order[str(mode)]),
            )
        )

    def key(item: SeedSelection) -> tuple[Any, ...]:
        fixed_cost = item.worst_fixed_point_sweep
        return (
            -item.minimum_first_sweep_accepted,
            -item.total_first_sweep_accepted,
            -item.minimum_best_accepted,
            -item.total_best_accepted,
            math.inf if fixed_cost is None else fixed_cost,
            item.mode_order,
        )

    return sorted(selections, key=key)[0]


def triangular_toy_proposal(guess: Sequence[int], target: Sequence[int]) -> tuple[int, ...]:
    candidate = validate_tokens(guess)
    truth = validate_tokens(target)
    if len(candidate) != len(truth):
        raise JacobiInvariantError("toy guess/target lengths differ")
    proposal: list[int] = []
    prefix_exact = True
    for index, target_token in enumerate(truth):
        if index == 0 or prefix_exact:
            proposal.append(int(target_token))
        else:
            wrong = int(target_token) + 1
            proposal.append(wrong)
        prefix_exact = prefix_exact and candidate[index] == target_token
    return tuple(proposal)


def run_triangular_toy(
    seed: Sequence[int], target: Sequence[int], maximum_iterations: int
) -> list[dict[str, Any]]:
    guess = validate_tokens(seed)
    truth = validate_tokens(target)
    if len(guess) != len(truth) or int(maximum_iterations) <= 0:
        raise JacobiInvariantError("invalid toy trajectory")
    rows: list[dict[str, Any]] = []
    for sweep in range(1, int(maximum_iterations) + 1):
        proposal = triangular_toy_proposal(guess, truth)
        accepted = self_consistent_prefix(guess, proposal)
        target_prefix = longest_common_prefix(guess, truth)
        rows.append(
            {
                "sweep": sweep,
                "guess": list(guess),
                "proposal": list(proposal),
                "self_consistent_prefix": accepted,
                "guess_target_prefix": target_prefix,
                "changed_positions": changed_positions(guess, proposal),
                "fixed_point": guess == proposal,
            }
        )
        guess = proposal
    return rows


def exhaustive_toy_prefix_control(length: int) -> dict[str, Any]:
    width = int(length)
    if width <= 0 or width > 12:
        raise JacobiInvariantError("toy width must be in [1, 12]")
    target = tuple(range(1, width + 1))
    checked = 0
    mismatches = 0
    for mask in range(1 << width):
        guess = tuple(
            target[index] if mask & (1 << index) else target[index] + 100
            for index in range(width)
        )
        proposal = triangular_toy_proposal(guess, target)
        accepted = self_consistent_prefix(guess, proposal)
        if tuple(guess[:accepted]) != tuple(target[:accepted]):
            mismatches += 1
        checked += 1
    return {
        "width": width,
        "checked": checked,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }


def future_mutation_control() -> dict[str, Any]:
    target = (1, 2, 3, 4, 5, 6)
    base = (1, 2, 3, 404, 505, 606)
    mutated = (1, 2, 3, 404, 9999, 8888)
    left = triangular_toy_proposal(base, target)
    right = triangular_toy_proposal(mutated, target)
    # Mutations at indices 4 and 5 cannot affect proposals through index 4.
    passed = left[:5] == right[:5]
    return {
        "left": list(left),
        "right": list(right),
        "prefix_equal_through": 5,
        "passed": passed,
    }
