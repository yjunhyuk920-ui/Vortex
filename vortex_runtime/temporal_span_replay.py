"""Exact dyadic temporal-span certificates for causal dense projections.

The module treats every captured float32 scalar as its exact dyadic rational
value.  Odd-prime finite-field images provide sound rational-independence
certificates: if a new vector raises rank modulo any registered prime, it cannot
be in the exact rational span of the prior vectors.

Non-increase modulo the registered primes is intentionally *not* called an exact
replay hit.  A separately verified exact coefficient witness is required before
any replay credit is granted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.activation_sparsity import (
    HookContext,
    ProjectionRegistration,
    register_linear_projections,
)


class TemporalSpanError(ValueError):
    """Raised when exact temporal-span evidence is malformed."""


def _validate_prime(prime: int) -> int:
    value = int(prime)
    if value <= 2 or value % 2 == 0:
        raise TemporalSpanError("registered modulus must be an odd prime")
    # The experiment deliberately uses small primes so int64 vectorized products
    # cannot overflow at the registered trace widths.
    if value >= 1 << 20:
        raise TemporalSpanError("registered modulus is too large for safe accounting")
    return value


def float32_to_field(values: Any, prime: int) -> np.ndarray:
    """Map exact IEEE-754 binary32 values into an odd prime field.

    Normal values are ``significand * 2**(exponent-150)`` and subnormals are
    ``mantissa * 2**-149``.  Powers of two are invertible for odd primes.
    """

    modulus = _validate_prime(prime)
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = array.view(np.uint32)
    exponent_bits = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mantissa = (bits & np.uint32(0x7FFFFF)).astype(np.int64)
    if np.any(exponent_bits == 0xFF):
        raise TemporalSpanError("non-finite float32 cannot enter an exact span")

    normal = exponent_bits != 0
    significand = mantissa.copy()
    significand[normal] += 1 << 23
    dyadic_exponent = np.where(normal, exponent_bits - 150, -149)
    result = np.mod(significand, modulus).astype(np.int64, copy=False)

    for exponent in np.unique(dyadic_exponent):
        mask = dyadic_exponent == exponent
        power = pow(2, int(exponent), modulus)
        result[mask] = np.mod(result[mask] * power, modulus)

    negative = ((bits >> np.uint32(31)) & np.uint32(1)).astype(bool)
    result[negative & (result != 0)] = modulus - result[negative & (result != 0)]
    return result


def exact_float32_fraction(value: float | np.float32) -> Fraction:
    scalar = np.float32(value)
    if not np.isfinite(scalar):
        raise TemporalSpanError("non-finite coefficient source")
    numerator, denominator = float(scalar).as_integer_ratio()
    return Fraction(numerator, denominator)


def exact_vector_fractions(vector: Any) -> tuple[Fraction, ...]:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return tuple(exact_float32_fraction(value) for value in array)


def verify_fraction_witness(
    prior_vectors: Sequence[Any],
    target_vector: Any,
    coefficients: Sequence[Fraction | int],
) -> bool:
    """Verify an exact rational span witness without floating-point tolerance."""

    if len(prior_vectors) != len(coefficients):
        raise TemporalSpanError("witness coefficient count mismatch")
    target = exact_vector_fractions(target_vector)
    sources = [exact_vector_fractions(vector) for vector in prior_vectors]
    if any(len(source) != len(target) for source in sources):
        raise TemporalSpanError("witness vector width mismatch")
    normalized = [Fraction(value) for value in coefficients]
    for column, expected in enumerate(target):
        observed = sum(
            coefficient * source[column]
            for coefficient, source in zip(normalized, sources, strict=True)
        )
        if observed != expected:
            return False
    return True


def canonical_float32_bytes(vector: Any) -> bytes:
    """Canonicalize signed zeros, then return exact float32 value bytes."""

    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(-1))
    if not np.all(np.isfinite(array)):
        raise TemporalSpanError("non-finite vector cannot be canonicalized")
    canonical = array.copy()
    canonical[canonical == 0] = np.float32(0.0)
    return canonical.tobytes()


def vector_sha256(vector: Any) -> str:
    return hashlib.sha256(canonical_float32_bytes(vector)).hexdigest()


@dataclass
class IncrementalModularSpan:
    width: int
    prime: int
    basis: np.ndarray = field(init=False)
    pivots: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.prime = _validate_prime(self.prime)
        if self.width <= 0:
            raise TemporalSpanError("span width must be positive")
        self.basis = np.empty((0, self.width), dtype=np.int64)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def add_float32(self, vector: Any) -> bool:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size != self.width:
            raise TemporalSpanError(
                f"span vector width mismatch: {array.size} != {self.width}"
            )
        reduced = float32_to_field(array, self.prime).reshape(-1)
        if self.rank:
            coefficients = reduced[np.asarray(self.pivots, dtype=np.int64)]
            # Registered primes/traces keep the int64 dot-product accumulation
            # comfortably below overflow.
            reduced = np.mod(reduced - coefficients @ self.basis, self.prime)
        nonzero = np.flatnonzero(reduced)
        if nonzero.size == 0:
            return False

        pivot = int(nonzero[0])
        inverse = pow(int(reduced[pivot]), self.prime - 2, self.prime)
        row = np.mod(reduced * inverse, self.prime)
        if self.rank:
            factors = self.basis[:, pivot].copy()
            self.basis = np.mod(
                self.basis - factors[:, None] * row[None, :], self.prime
            )
        self.basis = np.concatenate((self.basis, row.reshape(1, -1)), axis=0)
        self.pivots.append(pivot)
        return True

    def validate(self) -> None:
        if self.basis.shape != (len(self.pivots), self.width):
            raise TemporalSpanError("basis shape/pivot count mismatch")
        if len(set(self.pivots)) != len(self.pivots):
            raise TemporalSpanError("basis pivots are not unique")
        for index, pivot in enumerate(self.pivots):
            column = self.basis[:, pivot]
            expected = np.zeros(self.rank, dtype=np.int64)
            expected[index] = 1
            if not np.array_equal(column, expected):
                raise TemporalSpanError("basis is not reduced at pivot columns")


@dataclass(frozen=True)
class TemporalSpanCertificate:
    vector_count: int
    width: int
    primes: tuple[int, ...]
    rank_trajectories: dict[str, tuple[int, ...]]
    independent_flags: tuple[bool, ...]
    certified_independent_count: int
    uncertified_count: int
    exact_duplicate_hits: int
    first_exact_duplicate_position: int | None
    maximum_rank_lower_bound: int
    rank_disagreement_count: int

    @property
    def certified_independent_fraction(self) -> float:
        if self.vector_count == 0:
            return 0.0
        return self.certified_independent_count / self.vector_count

    @property
    def exact_duplicate_fraction(self) -> float:
        if self.vector_count == 0:
            return 0.0
        return self.exact_duplicate_hits / self.vector_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector_count": self.vector_count,
            "width": self.width,
            "primes": list(self.primes),
            "rank_trajectories": {
                key: list(value) for key, value in self.rank_trajectories.items()
            },
            "independent_flags": list(self.independent_flags),
            "certified_independent_count": self.certified_independent_count,
            "uncertified_count": self.uncertified_count,
            "certified_independent_fraction": self.certified_independent_fraction,
            "exact_duplicate_hits": self.exact_duplicate_hits,
            "exact_duplicate_fraction": self.exact_duplicate_fraction,
            "first_exact_duplicate_position": self.first_exact_duplicate_position,
            "maximum_rank_lower_bound": self.maximum_rank_lower_bound,
            "rank_disagreement_count": self.rank_disagreement_count,
        }


def certify_temporal_span(
    vectors: Sequence[Any], *, primes: Sequence[int]
) -> TemporalSpanCertificate:
    if not primes:
        raise TemporalSpanError("at least one prime is required")
    normalized_primes = tuple(_validate_prime(value) for value in primes)
    if len(set(normalized_primes)) != len(normalized_primes):
        raise TemporalSpanError("registered primes must be unique")
    if not vectors:
        raise TemporalSpanError("temporal span requires at least one vector")

    arrays = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    width = int(arrays[0].size)
    if width <= 0 or any(array.size != width for array in arrays):
        raise TemporalSpanError("temporal vectors have inconsistent widths")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise TemporalSpanError("non-finite temporal vector")

    spans = {
        prime: IncrementalModularSpan(width=width, prime=prime)
        for prime in normalized_primes
    }
    trajectories: dict[int, list[int]] = {prime: [] for prime in normalized_primes}
    independent_flags: list[bool] = []
    rank_disagreements = 0
    seen: dict[str, list[bytes]] = {}
    duplicate_hits = 0
    first_duplicate: int | None = None

    for position, array in enumerate(arrays):
        increments: list[bool] = []
        for prime, span in spans.items():
            increments.append(span.add_float32(array))
            trajectories[prime].append(span.rank)
        independent_flags.append(any(increments))
        rank_disagreements += int(len(set(increments)) > 1)

        payload = canonical_float32_bytes(array)
        digest = hashlib.sha256(payload).hexdigest()
        candidates = seen.setdefault(digest, [])
        duplicate = any(candidate == payload for candidate in candidates)
        if duplicate:
            duplicate_hits += 1
            if first_duplicate is None:
                first_duplicate = position
        else:
            candidates.append(payload)

    for span in spans.values():
        span.validate()
    maximum_rank = max(span.rank for span in spans.values())
    independent_count = sum(int(value) for value in independent_flags)
    return TemporalSpanCertificate(
        vector_count=len(arrays),
        width=width,
        primes=normalized_primes,
        rank_trajectories={
            str(prime): tuple(values) for prime, values in trajectories.items()
        },
        independent_flags=tuple(independent_flags),
        certified_independent_count=independent_count,
        uncertified_count=len(arrays) - independent_count,
        exact_duplicate_hits=duplicate_hits,
        first_exact_duplicate_position=first_duplicate,
        maximum_rank_lower_bound=maximum_rank,
        rank_disagreement_count=rank_disagreements,
    )


@dataclass(frozen=True)
class CapturedProjectionCall:
    model_id: str
    prompt_family: str
    phase: str
    decode_step: int
    module_name: str
    module_aliases: tuple[str, ...]
    input_width: int
    output_width: int
    vector_index: int
    vector_sha256: str
    vector: np.ndarray = field(repr=False, compare=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_family": self.prompt_family,
            "phase": self.phase,
            "decode_step": self.decode_step,
            "module_name": self.module_name,
            "module_aliases": list(self.module_aliases),
            "input_width": self.input_width,
            "output_width": self.output_width,
            "vector_index": self.vector_index,
            "vector_sha256": self.vector_sha256,
        }


@dataclass
class TemporalSpanRecorder:
    model: Any
    registrations: tuple[ProjectionRegistration, ...]
    context: HookContext = field(default_factory=HookContext)
    calls: list[CapturedProjectionCall] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list, init=False)
    _call_counts: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_model(cls, model: Any) -> "TemporalSpanRecorder":
        return cls(model=model, registrations=register_linear_projections(model))

    def attach(self) -> None:
        if self._handles:
            raise TemporalSpanError("temporal hooks are already attached")
        by_identity = {item.object_identity: item for item in self.registrations}
        seen: set[int] = set()
        try:
            iterator = self.model.named_modules(remove_duplicate=False)
        except TypeError:  # pragma: no cover
            iterator = self.model.named_modules()
        for _, module in iterator:
            identity = id(module)
            if identity not in by_identity or identity in seen:
                continue
            seen.add(identity)
            registration = by_identity[identity]

            def hook(
                current_module: Any,
                arguments: tuple[Any, ...],
                *,
                registration: ProjectionRegistration = registration,
            ) -> None:
                if self.context.phase == "inactive":
                    return
                if not arguments:
                    raise TemporalSpanError(
                        f"projection {registration.canonical_name} has no input"
                    )
                tensor = arguments[0]
                if not hasattr(tensor, "shape") or tensor.ndim < 1:
                    raise TemporalSpanError("projection input is not a tensor")
                if int(tensor.shape[-1]) != registration.input_width:
                    raise TemporalSpanError(
                        f"projection input width mismatch for {registration.canonical_name}"
                    )
                flattened = (
                    tensor.detach()
                    .to(dtype=current_module.weight.dtype)
                    .to(dtype=getattr(current_module.weight, "dtype", tensor.dtype))
                    .cpu()
                    .contiguous()
                    .reshape(-1, registration.input_width)
                    .to(dtype=__import__("torch").float32)
                    .numpy()
                )
                for vector_index, vector in enumerate(flattened):
                    copied = np.ascontiguousarray(vector, dtype=np.float32).copy()
                    self.calls.append(
                        CapturedProjectionCall(
                            model_id=self.context.model_id,
                            prompt_family=self.context.prompt_family,
                            phase=self.context.phase,
                            decode_step=self.context.decode_step,
                            module_name=registration.canonical_name,
                            module_aliases=registration.aliases,
                            input_width=registration.input_width,
                            output_width=registration.output_width,
                            vector_index=vector_index,
                            vector_sha256=vector_sha256(copied),
                            vector=copied,
                        )
                    )
                self._call_counts[registration.canonical_name] = (
                    self._call_counts.get(registration.canonical_name, 0) + 1
                )

            self._handles.append(module.register_forward_pre_hook(hook))
        if len(seen) != len(self.registrations):
            raise TemporalSpanError("not every registered projection was hooked")

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.context.phase = "inactive"

    def set_context(
        self,
        *,
        model_id: str,
        prompt_family: str,
        phase: str,
        decode_step: int,
    ) -> None:
        if phase not in {"prefill", "first_decode", "warm_decode", "inactive"}:
            raise TemporalSpanError(f"unsupported causal phase: {phase}")
        self.context = HookContext(
            model_id=model_id,
            prompt_family=prompt_family,
            phase=phase,
            decode_step=decode_step,
        )

    def drain(self) -> list[CapturedProjectionCall]:
        rows = self.calls
        self.calls = []
        return rows

    def missing_called_modules(self) -> tuple[str, ...]:
        return tuple(
            item.canonical_name
            for item in self.registrations
            if self._call_counts.get(item.canonical_name, 0) == 0
        )


def q4_matrix_bytes(input_width: int, output_width: int) -> int:
    if input_width <= 0 or output_width <= 0:
        raise TemporalSpanError("matrix dimensions must be positive")
    return math.ceil(input_width * output_width * 4 / 8)


def dense_operation_terms(input_width: int, output_width: int) -> int:
    if input_width <= 0 or output_width <= 0:
        raise TemporalSpanError("matrix dimensions must be positive")
    return input_width * output_width


def favorable_basis_cache_bytes(
    *, input_width: int, output_width: int, rank_lower_bound: int
) -> int:
    if rank_lower_bound < 0 or rank_lower_bound > input_width:
        raise TemporalSpanError("invalid basis rank lower bound")
    # Exact captured input and output vectors in float32.  Coefficients, indexes,
    # allocator overhead and rank metadata are excluded to favor survival.
    return rank_lower_bound * (input_width + output_width) * 4


def grouped_calls(
    calls: Iterable[CapturedProjectionCall], *, phase: str
) -> dict[str, list[CapturedProjectionCall]]:
    result: dict[str, list[CapturedProjectionCall]] = {}
    for call in calls:
        if call.phase != phase:
            continue
        result.setdefault(call.module_name, []).append(call)
    for rows in result.values():
        rows.sort(key=lambda item: (item.decode_step, item.vector_index))
    return result
