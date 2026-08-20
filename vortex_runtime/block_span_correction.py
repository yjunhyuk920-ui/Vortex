from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch


DEFAULT_PRIMES: tuple[int, ...] = (1_000_003, 1_000_033, 1_000_037)
LLAMA405_HIDDEN_SIZE = 16_384
LLAMA405_INTERMEDIATE_SIZE = 53_248
LLAMA405_LAYER_COUNT = 126
LLAMA405_OPERATOR_INPUT_WIDTH_SUM = (
    3 * LLAMA405_HIDDEN_SIZE + LLAMA405_INTERMEDIATE_SIZE
)


class BlockSpanCorrectionError(RuntimeError):
    """Fail-closed invariant error for EXP-092A."""


@dataclass(frozen=True)
class ModularRankRow:
    prime: int
    guess_rank: int
    union_rank: int
    raw_extra_rank: int
    certifies_guess_full_row_rank: bool


@dataclass(frozen=True)
class BlockSpanReport:
    states: int
    width: int
    selected_columns: int
    selected_column_sha256: str
    guess_sha256: str
    true_sha256: str
    modular_rows: tuple[ModularRankRow, ...]
    guess_full_row_rank_certified: bool
    certifying_prime_count: int
    certified_extra_rank_lower_bound: int
    certified_extra_fraction_lower_bound: float
    zero_extra_directions_observed: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modular_rows"] = [asdict(row) for row in self.modular_rows]
        return value


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_columns(width: int, count: int, *, seed: str) -> tuple[int, ...]:
    width = int(width)
    count = int(count)
    if width <= 0:
        raise BlockSpanCorrectionError("width must be positive")
    if count <= 0 or count > width:
        raise BlockSpanCorrectionError("column count must be in [1,width]")
    chosen: set[int] = set()
    counter = 0
    while len(chosen) < count:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            value = int.from_bytes(digest[offset : offset + 4], "little") % width
            chosen.add(value)
            if len(chosen) == count:
                break
        counter += 1
    return tuple(sorted(chosen))


def _bf16_components(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if tensor.dtype != torch.bfloat16:
        raise BlockSpanCorrectionError(f"expected BF16, got {tensor.dtype}")
    if tensor.ndim != 2:
        raise BlockSpanCorrectionError(f"expected [states,width], got {tuple(tensor.shape)}")
    bits = (
        tensor.detach().contiguous().cpu().view(torch.int16).numpy().view(np.uint16)
    )
    sign = ((bits >> np.uint16(15)) & np.uint16(1)).astype(np.int64)
    exponent_bits = ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    fraction = (bits & np.uint16(0x7F)).astype(np.int64)
    if np.any(exponent_bits == 0xFF):
        raise BlockSpanCorrectionError("non-finite BF16 activation")
    normal = exponent_bits != 0
    mantissa = np.where(normal, 128 + fraction, fraction).astype(np.int64)
    mantissa = np.where(sign == 0, mantissa, -mantissa)
    exponent = np.where(normal, exponent_bits - 134, -133).astype(np.int64)
    exponent = np.where(mantissa == 0, 0, exponent)
    return mantissa, exponent


def joint_dyadic_integer_rows(
    guess: torch.Tensor,
    true: torch.Tensor,
    *,
    columns: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if guess.shape != true.shape:
        raise BlockSpanCorrectionError(
            f"guess/true shape mismatch: {tuple(guess.shape)} != {tuple(true.shape)}"
        )
    if guess.dtype != true.dtype or guess.dtype != torch.bfloat16:
        raise BlockSpanCorrectionError("guess and true must both be BF16")
    column_array = np.asarray(tuple(map(int, columns)), dtype=np.int64)
    if column_array.ndim != 1 or column_array.size == 0:
        raise BlockSpanCorrectionError("empty column selection")
    if int(column_array.min()) < 0 or int(column_array.max()) >= int(guess.shape[1]):
        raise BlockSpanCorrectionError("selected column outside activation width")
    if len(set(column_array.tolist())) != int(column_array.size):
        raise BlockSpanCorrectionError("duplicate selected columns")

    g_m, g_e = _bf16_components(guess[:, column_array.tolist()])
    t_m, t_e = _bf16_components(true[:, column_array.tolist()])
    mantissa = np.concatenate([g_m, t_m], axis=0)
    exponent = np.concatenate([g_e, t_e], axis=0)
    nonzero = mantissa != 0
    sentinel = np.iinfo(np.int64).max
    coordinate_exp = np.min(np.where(nonzero, exponent, sentinel), axis=0)
    coordinate_exp = np.where(coordinate_exp == sentinel, 0, coordinate_exp)
    shifts = np.where(nonzero, exponent - coordinate_exp[None, :], 0)
    maximum_shift = int(shifts.max(initial=0))
    if maximum_shift > 53:
        raise BlockSpanCorrectionError(
            f"exact int64 dyadic encoding requires shift {maximum_shift}>53"
        )
    scaled = np.zeros_like(mantissa, dtype=np.int64)
    for shift in np.unique(shifts):
        mask = shifts == shift
        if int(shift) == 0:
            scaled[mask] = mantissa[mask]
        else:
            scaled[mask] = mantissa[mask] << int(shift)
    if np.any(np.abs(scaled) > (1 << 60)):
        raise BlockSpanCorrectionError("dyadic integer headroom exceeded")
    states = int(guess.shape[0])
    return scaled[:states], scaled[states:]


def modular_rank(matrix: np.ndarray, prime: int) -> int:
    if matrix.ndim != 2:
        raise BlockSpanCorrectionError("modular_rank expects a matrix")
    prime = int(prime)
    if prime <= 2:
        raise BlockSpanCorrectionError("prime must exceed two")
    a = np.remainder(np.asarray(matrix, dtype=np.int64), prime).copy()
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        candidates = np.flatnonzero(a[rank:, col])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        inverse = pow(int(a[rank, col]), prime - 2, prime)
        a[rank, col:] = np.remainder(a[rank, col:] * inverse, prime)
        if rank + 1 < rows:
            factors = a[rank + 1 :, col].copy()
            nonzero_rows = np.flatnonzero(factors)
            if nonzero_rows.size:
                target = a[rank + 1 + nonzero_rows, col:]
                products = factors[nonzero_rows, None] * a[rank : rank + 1, col:]
                a[rank + 1 + nonzero_rows, col:] = np.remainder(
                    target - products, prime
                )
        rank += 1
        if rank == rows:
            break
    return int(rank)


def analyze_block_span(
    guess: torch.Tensor,
    true: torch.Tensor,
    *,
    columns: Sequence[int],
    primes: Sequence[int] = DEFAULT_PRIMES,
) -> BlockSpanReport:
    guess_i, true_i = joint_dyadic_integer_rows(guess, true, columns=columns)
    union_i = np.concatenate([guess_i, true_i], axis=0)
    modular_rows: list[ModularRankRow] = []
    for prime in map(int, primes):
        guess_rank = modular_rank(guess_i, prime)
        union_rank = modular_rank(union_i, prime)
        if union_rank < guess_rank:
            raise BlockSpanCorrectionError("union rank fell below guess rank")
        modular_rows.append(
            ModularRankRow(
                prime=prime,
                guess_rank=guess_rank,
                union_rank=union_rank,
                raw_extra_rank=union_rank - guess_rank,
                certifies_guess_full_row_rank=guess_rank == int(guess.shape[0]),
            )
        )
    states = int(guess.shape[0])
    certifying = [row for row in modular_rows if row.certifies_guess_full_row_rank]
    lower_bound = max((row.union_rank - states for row in certifying), default=0)
    return BlockSpanReport(
        states=states,
        width=int(guess.shape[1]),
        selected_columns=len(tuple(columns)),
        selected_column_sha256=canonical_sha256(tuple(map(int, columns))),
        guess_sha256=tensor_sha256(guess),
        true_sha256=tensor_sha256(true),
        modular_rows=tuple(modular_rows),
        guess_full_row_rank_certified=bool(certifying),
        certifying_prime_count=len(certifying),
        certified_extra_rank_lower_bound=lower_bound,
        certified_extra_fraction_lower_bound=lower_bound / states,
        zero_extra_directions_observed=all(
            row.union_rank == row.guess_rank for row in modular_rows
        ),
    )


def optimistic_sidecar_bytes(
    directions_per_operator: int,
    *,
    layer_count: int = LLAMA405_LAYER_COUNT,
    bytes_per_value: int = 2,
) -> int:
    directions_per_operator = int(directions_per_operator)
    if directions_per_operator < 0:
        raise BlockSpanCorrectionError("direction count cannot be negative")
    return int(
        directions_per_operator
        * LLAMA405_OPERATOR_INPUT_WIDTH_SUM
        * int(layer_count)
        * int(bytes_per_value)
    )


def correction_operation_multiplier(block_length: int, extra_directions: int) -> float:
    block_length = int(block_length)
    extra_directions = int(extra_directions)
    if block_length <= 0 or extra_directions < 0:
        raise BlockSpanCorrectionError("invalid block/direction count")
    return 1.0 + extra_directions / block_length
