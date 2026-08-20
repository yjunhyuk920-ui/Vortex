from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


P50_WHOLE_MODEL_FRACTION = 1.2 * 4.0 / 405.0
LLAMA405_TOTAL_PARAMETERS = 405_849_243_648
LLAMA405_ONE_MLP_PROJECTION_PARAMETERS = 109_924_319_232
LLAMA405_MLP_PARAMETER_SHARE = (
    3.0 * LLAMA405_ONE_MLP_PROJECTION_PARAMETERS / LLAMA405_TOTAL_PARAMETERS
)
LLAMA405_HIDDEN_SIZE = 16_384
LLAMA405_INTERMEDIATE_SIZE = 53_248
LLAMA405_LAYER_COUNT = 126
BF16_WORD_BITS = 16


class QuadraticResidualError(RuntimeError):
    """Fail-closed invariant violation for the finite-word residual compiler."""


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise QuadraticResidualError("empty percentile population")
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _require_bf16_2d(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise QuadraticResidualError(f"{name} must be a 2-D BF16 tensor")
    if not torch.isfinite(tensor.float()).all():
        raise QuadraticResidualError(f"{name} contains nonfinite values")
    return tensor.detach().contiguous().cpu()


def bf16_words(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _require_bf16_2d(tensor, name="BF16 word tensor")
    signed = tensor.view(torch.int16).to(torch.int32)
    return torch.bitwise_and(signed, 0xFFFF)


def words_to_bf16(words: torch.Tensor) -> torch.Tensor:
    if words.dtype not in (torch.int32, torch.int64) or words.ndim != 2:
        raise QuadraticResidualError("word tensor must be a 2-D integer tensor")
    bounded = torch.bitwise_and(words.to(torch.int32), 0xFFFF)
    signed = torch.where(bounded >= 0x8000, bounded - 0x10000, bounded).to(torch.int16)
    return signed.contiguous().view(torch.bfloat16)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    values = x.double()
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-30)


def exact_report(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = _require_bf16_2d(reference, name="reference")
    candidate = _require_bf16_2d(candidate, name="candidate")
    if reference.shape != candidate.shape:
        raise QuadraticResidualError("candidate/reference shape mismatch")
    equal = reference.view(torch.int16) == candidate.view(torch.int16)
    vector_exact = equal.all(dim=1)
    row_exact = equal.float().mean(dim=1)
    mismatched = (~equal).sum(dim=1)
    return {
        "vector_exact_fraction": float(vector_exact.float().mean().item()),
        "exact_vectors": int(vector_exact.sum().item()),
        "count": int(reference.shape[0]),
        "row_exact_fraction_p05": percentile(row_exact.tolist(), 0.05),
        "row_exact_fraction_p50": percentile(row_exact.tolist(), 0.50),
        "row_exact_fraction_p95": percentile(row_exact.tolist(), 0.95),
        "mismatched_rows_p50": percentile(mismatched.tolist(), 0.50),
        "mismatched_rows_p95": percentile(mismatched.tolist(), 0.95),
        "mismatched_rows_max": int(mismatched.max().item()),
    }


def balanced_partition(x: torch.Tensor, program_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = _require_bf16_2d(x, name="partition input")
    count = int(x.shape[0])
    k = int(program_count)
    if k < 1 or count < 2 * k:
        raise QuadraticResidualError("insufficient states for balanced programs")
    normalized = normalize_rows(x)
    centered = normalized - normalized.mean(dim=0)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    scores = centered @ direction
    order = sorted(range(count), key=lambda index: (float(scores[index]), index))
    assignments = torch.empty(count, dtype=torch.long)
    base, remainder = divmod(count, k)
    start = 0
    for program_index in range(k):
        size = base + (1 if program_index < remainder else 0)
        indices = order[start : start + size]
        assignments[torch.tensor(indices, dtype=torch.long)] = program_index
        start += size
    centroids = []
    for program_index in range(k):
        centroid = normalized[assignments == program_index].mean(dim=0)
        centroid = centroid / torch.linalg.vector_norm(centroid).clamp_min(1e-30)
        centroids.append(centroid)
    return assignments, torch.stack(centroids)


def gf2_rank(matrix: torch.Tensor) -> int:
    if matrix.ndim != 2:
        raise QuadraticResidualError("GF(2) rank requires a matrix")
    work = torch.bitwise_and(matrix.to(torch.uint8).clone(), 1)
    rows, columns = map(int, work.shape)
    pivot_row = 0
    for column in range(columns):
        candidates = torch.nonzero(work[pivot_row:, column], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        selected = pivot_row + int(candidates[0].item())
        if selected != pivot_row:
            temporary = work[pivot_row].clone()
            work[pivot_row] = work[selected]
            work[selected] = temporary
        for row in range(rows):
            if row != pivot_row and int(work[row, column].item()) == 1:
                work[row] = torch.bitwise_xor(work[row], work[pivot_row])
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def quadratic_features(bits: torch.Tensor) -> torch.Tensor:
    if bits.ndim != 2:
        raise QuadraticResidualError("context bits must be a matrix")
    values = torch.bitwise_and(bits.to(torch.uint8), 1)
    batch, dimensions = map(int, values.shape)
    features = [torch.ones((batch, 1), dtype=torch.uint8), values]
    pair_columns = []
    for left in range(dimensions):
        for right in range(left + 1, dimensions):
            pair_columns.append((values[:, left] & values[:, right]).unsqueeze(1))
    if pair_columns:
        features.append(torch.cat(pair_columns, dim=1))
    return torch.cat(features, dim=1)


def _candidate_direction_pool(
    x: torch.Tensor,
    mean: torch.Tensor,
    pool_size: int,
    seed: int,
) -> torch.Tensor:
    centered = x.double() - mean.double()
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    directions = [row.clone() for row in vh]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    while len(directions) < int(pool_size):
        value = torch.randn(x.shape[1], generator=generator, dtype=torch.float64)
        for existing in directions[: min(len(directions), int(x.shape[1]))]:
            value = value - torch.dot(value, existing) * existing
        norm = torch.linalg.vector_norm(value)
        if float(norm) > 1e-12:
            directions.append(value / norm)
    return torch.stack(directions[: int(pool_size)])


def select_context_basis(
    x: torch.Tensor,
    context_bits: int,
    pool_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x = _require_bf16_2d(x, name="context build input")
    d = int(context_bits)
    if d < 1 or pool_size < d:
        raise QuadraticResidualError("invalid context basis dimensions")
    mean = x.double().mean(dim=0)
    pool = _candidate_direction_pool(x, mean, pool_size, seed)
    pool_bits = ((x.double() - mean) @ pool.T >= 0).to(torch.uint8)
    selected: list[int] = []
    best_rank = 0
    remaining = list(range(int(pool.shape[0])))
    for _ in range(d):
        choice = None
        choice_rank = -1
        for candidate in remaining:
            columns = selected + [candidate]
            rank = gf2_rank(quadratic_features(pool_bits[:, columns]))
            if rank > choice_rank or (rank == choice_rank and (choice is None or candidate < choice)):
                choice = candidate
                choice_rank = rank
        if choice is None:
            raise QuadraticResidualError("context basis selection failed")
        selected.append(choice)
        remaining.remove(choice)
        best_rank = choice_rank
    basis = pool[selected].T.contiguous()
    bits = pool_bits[:, selected].contiguous()
    return mean.float(), basis.float(), bits, best_rank


def _modal_words(words: torch.Tensor) -> torch.Tensor:
    if words.ndim != 2:
        raise QuadraticResidualError("modal baseline requires [N,H] words")
    output = torch.empty(words.shape[1], dtype=torch.int32)
    for column in range(words.shape[1]):
        values, counts = torch.unique(words[:, column], return_counts=True)
        maximum = int(counts.max().item())
        choices = values[counts == maximum]
        output[column] = int(choices.min().item())
    return output


def unpack_word_bits(words: torch.Tensor) -> torch.Tensor:
    if words.ndim != 2:
        raise QuadraticResidualError("word bit unpack requires a matrix")
    columns = [torch.bitwise_and(words >> bit, 1).to(torch.uint8) for bit in range(BF16_WORD_BITS)]
    return torch.stack(columns, dim=2).reshape(words.shape[0], -1)


def pack_word_coefficients(coefficients: torch.Tensor, output_size: int) -> torch.Tensor:
    if coefficients.ndim != 2 or coefficients.shape[1] != output_size * BF16_WORD_BITS:
        raise QuadraticResidualError("coefficient bit shape mismatch")
    reshaped = coefficients.reshape(coefficients.shape[0], output_size, BF16_WORD_BITS).to(torch.int32)
    packed = torch.zeros((coefficients.shape[0], output_size), dtype=torch.int32)
    for bit in range(BF16_WORD_BITS):
        packed = torch.bitwise_or(packed, reshaped[:, :, bit] << bit)
    return packed


def solve_gf2(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise QuadraticResidualError("invalid GF(2) solve shapes")
    a = torch.bitwise_and(features.to(torch.uint8).clone(), 1)
    b = torch.bitwise_and(targets.to(torch.uint8).clone(), 1)
    rows, columns = map(int, a.shape)
    pivot_row = 0
    pivots: list[int] = []
    for column in range(columns):
        candidates = torch.nonzero(a[pivot_row:, column], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        selected = pivot_row + int(candidates[0].item())
        if selected != pivot_row:
            row_copy = a[pivot_row].clone()
            target_copy = b[pivot_row].clone()
            a[pivot_row] = a[selected]
            b[pivot_row] = b[selected]
            a[selected] = row_copy
            b[selected] = target_copy
        for row in range(rows):
            if row != pivot_row and int(a[row, column].item()) == 1:
                a[row] = torch.bitwise_xor(a[row], a[pivot_row])
                b[row] = torch.bitwise_xor(b[row], b[pivot_row])
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    consistent = True
    for row in range(pivot_row, rows):
        if not bool(a[row].any()) and bool(b[row].any()):
            consistent = False
            break
    coefficients = torch.zeros((columns, targets.shape[1]), dtype=torch.uint8)
    if consistent:
        for row, column in enumerate(pivots):
            coefficients[column] = b[row]
        replay = torch.remainder(features.to(torch.int16) @ coefficients.to(torch.int16), 2).to(torch.uint8)
        consistent = bool(torch.equal(replay, torch.bitwise_and(targets.to(torch.uint8), 1)))
    return coefficients, pivot_row, consistent


@dataclass(frozen=True)
class QuadraticResidualProgram:
    input_mean: torch.Tensor
    context_basis: torch.Tensor
    router_centroid: torch.Tensor
    baseline_words: torch.Tensor
    coefficient_words: torch.Tensor
    build_count: int
    feature_rank: int
    feature_count: int
    context_bits: int
    fingerprint: str

    @property
    def input_size(self) -> int:
        return int(self.input_mean.numel())

    @property
    def output_size(self) -> int:
        return int(self.baseline_words.numel())

    def context(self, x: torch.Tensor) -> torch.Tensor:
        x = _require_bf16_2d(x, name="program query")
        if x.shape[1] != self.input_size:
            raise QuadraticResidualError("program query input width mismatch")
        return ((x.float() - self.input_mean) @ self.context_basis >= 0).to(torch.uint8)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        features = quadratic_features(self.context(x))
        if features.shape[1] != self.feature_count:
            raise QuadraticResidualError("feature count mismatch")
        residual = torch.zeros((features.shape[0], self.output_size), dtype=torch.int32)
        for feature_index in range(self.feature_count):
            mask = features[:, feature_index].bool()
            if bool(mask.any()):
                residual[mask] = torch.bitwise_xor(
                    residual[mask], self.coefficient_words[feature_index]
                )
        words = torch.bitwise_xor(residual, self.baseline_words)
        return words_to_bf16(words)


@dataclass(frozen=True)
class QuadraticResidualLibrary:
    programs: tuple[QuadraticResidualProgram, ...]
    fingerprint: str

    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([program.query(x) for program in self.programs], dim=1)

    def query(self, x: torch.Tensor, reference: torch.Tensor, *, oracle: bool) -> dict[str, Any]:
        x = _require_bf16_2d(x, name="library query")
        reference = _require_bf16_2d(reference, name="library reference")
        predictions = self.predictions(x)
        if oracle:
            word_matches = (
                predictions.view(torch.int16)
                == reference[:, None, :].view(torch.int16)
            ).sum(dim=2)
            selected = torch.argmax(word_matches, dim=1)
            mode = "oracle"
        else:
            normalized = normalize_rows(x)
            centroids = torch.stack([program.router_centroid for program in self.programs])
            selected = torch.argmin(
                ((normalized[:, None] - centroids[None]) ** 2).sum(dim=2), dim=1
            )
            mode = "router"
        candidate = predictions[torch.arange(x.shape[0]), selected]
        return {
            "mode": mode,
            "selected": [int(value) for value in selected.tolist()],
            "use_counts": [
                int(value)
                for value in torch.bincount(selected, minlength=len(self.programs)).tolist()
            ],
            "candidate": candidate,
            "report": exact_report(reference, candidate),
        }


def fit_program(
    x: torch.Tensor,
    y: torch.Tensor,
    router_centroid: torch.Tensor,
    *,
    context_bits: int,
    context_pool: int,
    seed: int,
) -> tuple[QuadraticResidualProgram, bool]:
    x = _require_bf16_2d(x, name="program build x")
    y = _require_bf16_2d(y, name="program build y")
    if x.shape[0] != y.shape[0]:
        raise QuadraticResidualError("program build pair count mismatch")
    mean, basis, bits, feature_rank = select_context_basis(
        x, context_bits, context_pool, seed
    )
    features = quadratic_features(bits)
    words = bf16_words(y)
    baseline = _modal_words(words)
    residual_words = torch.bitwise_xor(words, baseline)
    target_bits = unpack_word_bits(residual_words)
    coefficients, solved_rank, consistent = solve_gf2(features, target_bits)
    coefficient_words = pack_word_coefficients(coefficients, int(y.shape[1]))
    payload = {
        "mean": tensor_sha256(mean),
        "basis": tensor_sha256(basis),
        "centroid": tensor_sha256(router_centroid),
        "baseline": tensor_sha256(baseline),
        "coefficients": tensor_sha256(coefficient_words),
        "context_bits": int(context_bits),
        "feature_rank": int(feature_rank),
        "solved_rank": int(solved_rank),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    program = QuadraticResidualProgram(
        mean,
        basis,
        router_centroid.double(),
        baseline,
        coefficient_words,
        int(x.shape[0]),
        int(solved_rank),
        int(features.shape[1]),
        int(context_bits),
        fingerprint,
    )
    build_exact = consistent and exact_report(y, program.query(x))["vector_exact_fraction"] == 1.0
    return program, build_exact


def fit_library(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    program_count: int,
    context_bits: int,
    context_pool: int,
    seed: int,
) -> tuple[QuadraticResidualLibrary, torch.Tensor, list[bool]]:
    x = _require_bf16_2d(x, name="library build x")
    y = _require_bf16_2d(y, name="library build y")
    assignments, centroids = balanced_partition(x, program_count)
    programs = []
    build_exact = []
    for program_index in range(program_count):
        mask = assignments == program_index
        program, exact = fit_program(
            x[mask],
            y[mask],
            centroids[program_index],
            context_bits=context_bits,
            context_pool=context_pool,
            seed=seed + program_index * 1009,
        )
        programs.append(program)
        build_exact.append(exact)
    fingerprint = hashlib.sha256(
        "|".join(program.fingerprint for program in programs).encode()
    ).hexdigest()
    return QuadraticResidualLibrary(tuple(programs), fingerprint), assignments, build_exact


def synthetic_quadratic_control(seed: int = 20260820) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(0, 2, (32, 6), generator=generator, dtype=torch.uint8)
    features = quadratic_features(bits)
    coefficients = torch.randint(
        0, 2, (features.shape[1], 64), generator=generator, dtype=torch.uint8
    )
    targets = torch.remainder(
        features.to(torch.int16) @ coefficients.to(torch.int16), 2
    ).to(torch.uint8)
    fitted, rank, consistent = solve_gf2(features, targets)
    replay = torch.remainder(features.to(torch.int16) @ fitted.to(torch.int16), 2).to(torch.uint8)
    return {
        "consistent": bool(consistent),
        "rank": int(rank),
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "mismatches": int(torch.count_nonzero(replay != targets).item()),
    }


def target_resources(
    *,
    program_count: int,
    context_bits: int,
    scalar_bytes: int = 4,
) -> dict[str, Any]:
    d = int(context_bits)
    feature_count = 1 + d + d * (d - 1) // 2
    per_program_layer_bytes = (
        LLAMA405_HIDDEN_SIZE * scalar_bytes  # input mean
        + LLAMA405_HIDDEN_SIZE * d * scalar_bytes  # context basis
        + LLAMA405_HIDDEN_SIZE * scalar_bytes  # router centroid
        + LLAMA405_HIDDEN_SIZE * 2  # BF16 baseline words
        + feature_count * LLAMA405_HIDDEN_SIZE * 2  # packed 16-bit coefficients
    )
    sidecar_bytes = program_count * LLAMA405_LAYER_COUNT * per_program_layer_bytes
    router_ops = 3 * program_count * LLAMA405_HIDDEN_SIZE
    context_ops = 2 * LLAMA405_HIDDEN_SIZE * d
    feature_ops = d * (d - 1) // 2
    packed_word_xors = feature_count * (LLAMA405_HIDDEN_SIZE * BF16_WORD_BITS // 64)
    assembly_ops = 2 * LLAMA405_HIDDEN_SIZE
    favorable_ops = router_ops + context_ops + feature_ops + packed_word_xors + assembly_ops
    dense_mlp_macs = 3 * LLAMA405_HIDDEN_SIZE * LLAMA405_INTERMEDIATE_SIZE
    compiled_fraction = favorable_ops / dense_mlp_macs
    whole_model_fraction = 1.0 - LLAMA405_MLP_PARAMETER_SHARE + LLAMA405_MLP_PARAMETER_SHARE * compiled_fraction
    return {
        "feature_count": feature_count,
        "sidecar_bytes": int(sidecar_bytes),
        "sidecar_gib": sidecar_bytes / 1024**3,
        "favorable_packed_instruction_count_per_layer": int(favorable_ops),
        "compiled_mlp_operation_fraction": float(compiled_fraction),
        "whole_model_operation_fraction_if_only_mlp_replaced": float(whole_model_fraction),
        "uncompiled_non_mlp_fraction": float(1.0 - LLAMA405_MLP_PARAMETER_SHARE),
    }
