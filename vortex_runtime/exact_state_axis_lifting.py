from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch


P50_WHOLE_MODEL_FRACTION = 1.2 * 4.0 / 405.0
P95_WHOLE_MODEL_FRACTION = 1.5 * 4.0 / 405.0
LLAMA405_TOTAL_PARAMETERS = 405_849_243_648
LLAMA405_ONE_MLP_PROJECTION_PARAMETERS = 109_924_319_232
LLAMA405_MLP_PARAMETER_SHARE = (
    3.0 * LLAMA405_ONE_MLP_PROJECTION_PARAMETERS / LLAMA405_TOTAL_PARAMETERS
)


class ExactStateAxisLiftingError(RuntimeError):
    """Fail-closed invariant failure for the exact state-axis lifting Gate."""


_POPCOUNT16 = np.array([int(i).bit_count() for i in range(1 << 16)], dtype=np.uint8)


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    return (
        _POPCOUNT16[(values & np.uint64(0xFFFF)).astype(np.uint16)].astype(np.uint16)
        + _POPCOUNT16[
            ((values >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)
        ].astype(np.uint16)
        + _POPCOUNT16[
            ((values >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16)
        ].astype(np.uint16)
        + _POPCOUNT16[
            ((values >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint16)
        ].astype(np.uint16)
    )


def _signed_abs_u64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values == np.iinfo(np.int64).min):
        raise ExactStateAxisLiftingError("int64 minimum cannot be safely absolutized")
    return np.abs(values).astype(np.uint64)


def _popcount_i64(values: np.ndarray) -> np.ndarray:
    return _popcount_u64(_signed_abs_u64(values))


def _bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise ExactStateAxisLiftingError(f"expected bfloat16 tensor, got {tensor.dtype}")
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.detach().contiguous()
    if tensor.ndim != 2:
        raise ExactStateAxisLiftingError(f"expected [states,width], got {tuple(tensor.shape)}")
    return tensor.view(torch.int16).numpy().view(np.uint16).copy()


def _decode_bf16(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits, dtype=np.uint16)
    sign = ((bits >> np.uint16(15)) & np.uint16(1)).astype(np.int64)
    exponent_bits = ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    fraction = (bits & np.uint16(0x7F)).astype(np.int64)
    if np.any(exponent_bits == 0xFF):
        raise ExactStateAxisLiftingError("nonfinite BF16 value")
    normal = exponent_bits != 0
    mantissa = np.where(normal, 128 + fraction, fraction).astype(np.int64)
    mantissa = np.where(sign == 0, mantissa, -mantissa)
    exponent = np.where(normal, exponent_bits - 134, -133).astype(np.int64)
    exponent = np.where(mantissa == 0, 0, exponent)
    return mantissa, exponent


@dataclass(frozen=True)
class ExactDyadicBlock:
    """Per-coordinate exact integer encoding of BF16 values.

    values[t,j] = scaled[t,j] * 2**coordinate_exponent[j]
    """

    scaled: np.ndarray
    coordinate_exponent: np.ndarray
    source_bits_sha256: str
    maximum_shift: int

    @property
    def states(self) -> int:
        return int(self.scaled.shape[0])

    @property
    def width(self) -> int:
        return int(self.scaled.shape[1])


def encode_bf16_block_exact(tensor: torch.Tensor) -> ExactDyadicBlock:
    bits = _bf16_bits(tensor)
    mantissa, exponent = _decode_bf16(bits)
    nonzero = mantissa != 0
    sentinel = np.iinfo(np.int64).max
    coordinate_exponent = np.min(np.where(nonzero, exponent, sentinel), axis=0)
    coordinate_exponent = np.where(
        coordinate_exponent == sentinel, 0, coordinate_exponent
    ).astype(np.int64)
    shifts = np.where(nonzero, exponent - coordinate_exponent[None, :], 0).astype(
        np.int64
    )
    maximum_shift = int(shifts.max(initial=0))
    if maximum_shift > 53:
        raise ExactStateAxisLiftingError(
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
        raise ExactStateAxisLiftingError("dyadic integer headroom exceeded")
    return ExactDyadicBlock(
        scaled=scaled,
        coordinate_exponent=coordinate_exponent,
        source_bits_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
        maximum_shift=maximum_shift,
    )


PredictorMode = Literal["vector", "coordinate"]


@dataclass
class LiftingAnalysis:
    mode: PredictorMode
    states: int
    width: int
    residual_nonzero_count: int
    residual_nonzero_fraction: float
    residual_atom_union_count: int
    atom_union_operation_fraction: float
    maximum_residual_popcount: int
    residual_popcount_sum: int
    reconstruction_mismatches: int
    predictor_histogram: dict[str, int]
    residual_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_vectors(values: np.ndarray, index: int) -> list[tuple[str, np.ndarray]]:
    current = values[index]
    candidates: list[tuple[str, np.ndarray]] = [("zero", np.zeros_like(current))]
    for parent in range(index):
        candidates.append((f"copy:{parent}", values[parent]))
    if index >= 2:
        candidates.append(("linear2", 2 * values[index - 1] - values[index - 2]))
    if index >= 3:
        candidates.append(
            (
                "parallelogram3",
                values[index - 1] + values[index - 2] - values[index - 3],
            )
        )
        candidates.append(
            (
                "quadratic3",
                3 * values[index - 1] - 3 * values[index - 2] + values[index - 3],
            )
        )
    return candidates


def _score_residual(residual: np.ndarray) -> tuple[int, int, int]:
    nonzero = int(np.count_nonzero(residual))
    absolute = _signed_abs_u64(residual)
    popcount = int(_popcount_u64(absolute).sum(dtype=np.uint64))
    if nonzero:
        magnitude_bits = int(
            np.where(
                absolute > 0,
                np.floor(np.log2(np.maximum(absolute, 1))) + 1,
                0,
            ).sum()
        )
    else:
        magnitude_bits = 0
    return nonzero, popcount, magnitude_bits


def _select_vector_residuals(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    states, width = values.shape
    residuals = np.zeros_like(values, dtype=np.int64)
    histogram: dict[str, int] = {"base": width}
    residuals[0] = values[0]
    for index in range(1, states):
        best_name = ""
        best_residual: np.ndarray | None = None
        best_score: tuple[int, int, int] | None = None
        for name, predictor in _candidate_vectors(values, index):
            residual = values[index] - predictor
            score = _score_residual(residual)
            if best_score is None or score < best_score or (
                score == best_score and name < best_name
            ):
                best_name = name
                best_residual = residual
                best_score = score
        if best_residual is None:
            raise ExactStateAxisLiftingError("no vector predictor selected")
        selected_predictor = values[index] - best_residual
        if not np.array_equal(selected_predictor + best_residual, values[index]):
            raise ExactStateAxisLiftingError("vector predictor reconstruction mismatch")
        residuals[index] = best_residual
        histogram[best_name] = histogram.get(best_name, 0) + width
    return residuals, histogram


def _select_coordinate_residuals(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    states, width = values.shape
    residuals = np.zeros_like(values, dtype=np.int64)
    residuals[0] = values[0]
    histogram: dict[str, int] = {"base": width}
    for index in range(1, states):
        current = values[index]
        best_residual = current.copy()
        best_popcount = _popcount_i64(best_residual).astype(np.int64)
        best_nonzero = best_residual != 0
        best_name = np.full(width, "zero", dtype=object)
        for name, predictor in _candidate_vectors(values, index)[1:]:
            residual = current - predictor
            popcount = _popcount_i64(residual).astype(np.int64)
            nonzero = residual != 0
            improve = (nonzero.astype(np.int8) < best_nonzero.astype(np.int8)) | (
                (nonzero == best_nonzero) & (popcount < best_popcount)
            )
            if np.any(improve):
                best_residual[improve] = residual[improve]
                best_popcount[improve] = popcount[improve]
                best_nonzero[improve] = nonzero[improve]
                best_name[improve] = name
        selected_predictor = values[index] - best_residual
        if not np.array_equal(selected_predictor + best_residual, values[index]):
            raise ExactStateAxisLiftingError(
                "coordinate predictor reconstruction mismatch"
            )
        residuals[index] = best_residual
        names, counts = np.unique(best_name, return_counts=True)
        for name, count in zip(names.tolist(), counts.tolist()):
            histogram[str(name)] = histogram.get(str(name), 0) + int(count)
    return residuals, histogram


def _signed_atom_union_count(residuals: np.ndarray) -> int:
    """Count signed dyadic atoms once per coordinate across all residual states.

    This is an intentionally favorable oracle lower bound: one positive and one
    negative bit mask per power-of-two atom may be shared across every future state,
    and mask generation, routing, reconstruction, and native-order repair are free.
    """

    if residuals.shape[0] <= 1:
        return 0
    tail = residuals[1:]
    positive = np.where(tail > 0, tail, 0).astype(np.uint64)
    negative = np.where(tail < 0, -tail, 0).astype(np.uint64)
    positive_union = np.bitwise_or.reduce(positive, axis=0, initial=np.uint64(0))
    negative_union = np.bitwise_or.reduce(negative, axis=0, initial=np.uint64(0))
    return int(
        _popcount_u64(positive_union).sum(dtype=np.uint64)
        + _popcount_u64(negative_union).sum(dtype=np.uint64)
    )


def analyze_lifting_block(
    tensor: torch.Tensor,
    *,
    mode: PredictorMode,
) -> LiftingAnalysis:
    encoded = encode_bf16_block_exact(tensor)
    values = encoded.scaled
    if encoded.states < 2:
        raise ExactStateAxisLiftingError("lifting block requires at least two states")
    if mode == "vector":
        residuals, histogram = _select_vector_residuals(values)
    elif mode == "coordinate":
        residuals, histogram = _select_coordinate_residuals(values)
    else:
        raise ExactStateAxisLiftingError(f"unsupported mode: {mode}")

    reconstruction_mismatches = 0
    if residuals.shape != values.shape or residuals.dtype != np.int64:
        reconstruction_mismatches += 1

    residual_nonzero_count = int(np.count_nonzero(residuals[1:]))
    denominator = max(1, (encoded.states - 1) * encoded.width)
    atom_union = _signed_atom_union_count(residuals)
    operation_fraction = (encoded.width + atom_union) / (
        encoded.states * encoded.width
    )
    popcounts = _popcount_i64(residuals[1:])
    return LiftingAnalysis(
        mode=mode,
        states=encoded.states,
        width=encoded.width,
        residual_nonzero_count=residual_nonzero_count,
        residual_nonzero_fraction=residual_nonzero_count / denominator,
        residual_atom_union_count=atom_union,
        atom_union_operation_fraction=float(operation_fraction),
        maximum_residual_popcount=int(popcounts.max(initial=0)),
        residual_popcount_sum=int(popcounts.sum(dtype=np.uint64)),
        reconstruction_mismatches=reconstruction_mismatches,
        predictor_histogram=histogram,
        residual_sha256=hashlib.sha256(residuals.tobytes()).hexdigest(),
    )


@dataclass
class MLPBlockAnalysis:
    block_size: int
    mode: PredictorMode
    input: LiftingAnalysis
    down_input: LiftingAnalysis
    projected_mlp_operation_fraction: float
    projected_whole_model_operation_fraction: float
    projected_whole_model_weight_fraction: float
    projected_joint_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_mlp_block(
    mlp_input: torch.Tensor,
    down_input: torch.Tensor,
    *,
    mode: PredictorMode,
) -> MLPBlockAnalysis:
    if mlp_input.shape[0] != down_input.shape[0]:
        raise ExactStateAxisLiftingError("input/down-input state count mismatch")
    input_analysis = analyze_lifting_block(mlp_input, mode=mode)
    down_analysis = analyze_lifting_block(down_input, mode=mode)
    mlp_operation_fraction = (
        2.0 * input_analysis.atom_union_operation_fraction
        + down_analysis.atom_union_operation_fraction
    ) / 3.0
    whole_operation = LLAMA405_MLP_PARAMETER_SHARE * mlp_operation_fraction
    whole_weight = LLAMA405_MLP_PARAMETER_SHARE / int(mlp_input.shape[0])
    return MLPBlockAnalysis(
        block_size=int(mlp_input.shape[0]),
        mode=mode,
        input=input_analysis,
        down_input=down_analysis,
        projected_mlp_operation_fraction=float(mlp_operation_fraction),
        projected_whole_model_operation_fraction=float(whole_operation),
        projected_whole_model_weight_fraction=float(whole_weight),
        projected_joint_fraction=float(max(whole_operation, whole_weight)),
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()
