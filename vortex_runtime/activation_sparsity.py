"""Exact activation-zero accounting for causal dense projections.

The instrumentation is observational: hooks never modify model inputs or
outputs.  Only IEEE values equal to positive or negative zero are counted.
Near-zero thresholds are intentionally unsupported because they would be an
approximation rather than an exact runtime skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Sequence


class ActivationSparsityError(ValueError):
    """Raised when a projection registration or accounting request is invalid."""


def unsigned_width(maximum_value: int) -> int:
    if maximum_value < 0:
        raise ActivationSparsityError("unsigned width requires nonnegative input")
    return max(1, math.ceil(max(1, maximum_value).bit_length() / 8))


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ProjectionRegistration:
    canonical_name: str
    aliases: tuple[str, ...]
    object_identity: int
    input_width: int
    output_width: int
    weight_shape: tuple[int, int]
    weight_sha256: str


@dataclass(frozen=True)
class ActivationCallAccounting:
    model_id: str
    prompt_family: str
    phase: str
    decode_step: int
    module_name: str
    module_aliases: tuple[str, ...]
    input_width: int
    output_width: int
    vector_count: int
    input_scalar_count: int
    exact_zero_count: int
    nonzero_count: int
    dense_operation_terms: int
    sparse_operation_terms: int
    zero_scan_terms: int
    fully_accounted_operation_terms: int
    dense_q4_weight_bytes: int
    sparse_q4_weight_bytes: int
    activation_metadata_bytes: int

    @property
    def exact_zero_fraction(self) -> float:
        return self.exact_zero_count / self.input_scalar_count

    @property
    def sparse_operation_fraction(self) -> float:
        return self.sparse_operation_terms / self.dense_operation_terms

    @property
    def fully_accounted_operation_fraction(self) -> float:
        return self.fully_accounted_operation_terms / self.dense_operation_terms

    @property
    def query_byte_fraction(self) -> float:
        return (
            self.sparse_q4_weight_bytes + self.activation_metadata_bytes
        ) / self.dense_q4_weight_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_family": self.prompt_family,
            "phase": self.phase,
            "decode_step": self.decode_step,
            "module_name": self.module_name,
            "module_aliases": list(self.module_aliases),
            "input_width": self.input_width,
            "output_width": self.output_width,
            "vector_count": self.vector_count,
            "input_scalar_count": self.input_scalar_count,
            "exact_zero_count": self.exact_zero_count,
            "nonzero_count": self.nonzero_count,
            "exact_zero_fraction": self.exact_zero_fraction,
            "dense_operation_terms": self.dense_operation_terms,
            "sparse_operation_terms": self.sparse_operation_terms,
            "zero_scan_terms": self.zero_scan_terms,
            "fully_accounted_operation_terms": self.fully_accounted_operation_terms,
            "sparse_operation_fraction": self.sparse_operation_fraction,
            "fully_accounted_operation_fraction": self.fully_accounted_operation_fraction,
            "dense_q4_weight_bytes": self.dense_q4_weight_bytes,
            "sparse_q4_weight_bytes": self.sparse_q4_weight_bytes,
            "activation_metadata_bytes": self.activation_metadata_bytes,
            "query_byte_fraction": self.query_byte_fraction,
        }


def account_activation_call(
    *,
    model_id: str,
    prompt_family: str,
    phase: str,
    decode_step: int,
    module_name: str,
    module_aliases: Sequence[str],
    input_width: int,
    output_width: int,
    vector_count: int,
    exact_zero_count: int,
) -> ActivationCallAccounting:
    if input_width <= 0 or output_width <= 0 or vector_count <= 0:
        raise ActivationSparsityError("projection dimensions must be positive")
    input_scalars = input_width * vector_count
    if exact_zero_count < 0 or exact_zero_count > input_scalars:
        raise ActivationSparsityError("exact-zero count outside input population")
    nonzero = input_scalars - exact_zero_count
    dense_terms = output_width * input_scalars
    sparse_terms = output_width * nonzero
    scan_terms = input_scalars
    dense_weight_bits = output_width * input_scalars * 4
    sparse_weight_bits = output_width * nonzero * 4
    dense_weight_bytes = max(1, math.ceil(dense_weight_bits / 8))
    sparse_weight_bytes = math.ceil(sparse_weight_bits / 8)
    index_width = unsigned_width(max(0, input_width - 1))
    pointer_width = unsigned_width(nonzero)
    metadata = nonzero * index_width + (vector_count + 1) * pointer_width
    return ActivationCallAccounting(
        model_id=model_id,
        prompt_family=prompt_family,
        phase=phase,
        decode_step=decode_step,
        module_name=module_name,
        module_aliases=tuple(module_aliases),
        input_width=input_width,
        output_width=output_width,
        vector_count=vector_count,
        input_scalar_count=input_scalars,
        exact_zero_count=exact_zero_count,
        nonzero_count=nonzero,
        dense_operation_terms=dense_terms,
        sparse_operation_terms=sparse_terms,
        zero_scan_terms=scan_terms,
        fully_accounted_operation_terms=sparse_terms + scan_terms,
        dense_q4_weight_bytes=dense_weight_bytes,
        sparse_q4_weight_bytes=sparse_weight_bytes,
        activation_metadata_bytes=metadata,
    )


def register_linear_projections(model: Any) -> tuple[ProjectionRegistration, ...]:
    """Register unique ``torch.nn.Linear`` objects while retaining aliases."""

    import torch

    grouped: dict[int, dict[str, Any]] = {}
    try:
        iterator = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        iterator = model.named_modules()
    for name, module in iterator:
        if not isinstance(module, torch.nn.Linear):
            continue
        identity = id(module)
        row = grouped.setdefault(
            identity,
            {
                "module": module,
                "aliases": [],
            },
        )
        row["aliases"].append(name)
    registrations: list[ProjectionRegistration] = []
    for identity, row in grouped.items():
        module = row["module"]
        aliases = tuple(sorted(set(str(alias) for alias in row["aliases"])))
        if not aliases:
            raise ActivationSparsityError("linear projection has no registered name")
        weight_shape = tuple(int(value) for value in module.weight.shape)
        expected = (int(module.out_features), int(module.in_features))
        if weight_shape != expected:
            raise ActivationSparsityError(
                f"linear weight shape mismatch for {aliases[0]}: {weight_shape} != {expected}"
            )
        registrations.append(
            ProjectionRegistration(
                canonical_name=aliases[0],
                aliases=aliases,
                object_identity=identity,
                input_width=int(module.in_features),
                output_width=int(module.out_features),
                weight_shape=weight_shape,
                weight_sha256=tensor_sha256(module.weight),
            )
        )
    return tuple(sorted(registrations, key=lambda item: item.canonical_name))


@dataclass
class HookContext:
    model_id: str = ""
    prompt_family: str = ""
    phase: str = "inactive"
    decode_step: int = -1


@dataclass
class ActivationSparsityRecorder:
    model: Any
    registrations: tuple[ProjectionRegistration, ...]
    context: HookContext = field(default_factory=HookContext)
    calls: list[ActivationCallAccounting] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list, init=False)
    _call_counts: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_model(cls, model: Any) -> "ActivationSparsityRecorder":
        return cls(model=model, registrations=register_linear_projections(model))

    def attach(self) -> None:
        if self._handles:
            raise ActivationSparsityError("hooks are already attached")
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
                    raise ActivationSparsityError(
                        f"projection {registration.canonical_name} received no positional input"
                    )
                tensor = arguments[0]
                if not hasattr(tensor, "shape") or tensor.ndim < 1:
                    raise ActivationSparsityError(
                        f"projection {registration.canonical_name} input is not a tensor"
                    )
                if int(tensor.shape[-1]) != registration.input_width:
                    raise ActivationSparsityError(
                        f"projection {registration.canonical_name} input width mismatch: "
                        f"{tensor.shape[-1]} != {registration.input_width}"
                    )
                vector_count = int(tensor.numel() // registration.input_width)
                zero_count = int((tensor == 0).sum().item())
                call = account_activation_call(
                    model_id=self.context.model_id,
                    prompt_family=self.context.prompt_family,
                    phase=self.context.phase,
                    decode_step=self.context.decode_step,
                    module_name=registration.canonical_name,
                    module_aliases=registration.aliases,
                    input_width=registration.input_width,
                    output_width=registration.output_width,
                    vector_count=vector_count,
                    exact_zero_count=zero_count,
                )
                self.calls.append(call)
                self._call_counts[registration.canonical_name] = (
                    self._call_counts.get(registration.canonical_name, 0) + 1
                )

            self._handles.append(module.register_forward_pre_hook(hook))
        if len(seen) != len(self.registrations):
            raise ActivationSparsityError("not every registered projection was hooked")

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
            raise ActivationSparsityError(f"unsupported causal phase: {phase}")
        self.context = HookContext(
            model_id=model_id,
            prompt_family=prompt_family,
            phase=phase,
            decode_step=decode_step,
        )

    def missing_called_modules(self) -> tuple[str, ...]:
        return tuple(
            item.canonical_name
            for item in self.registrations
            if self._call_counts.get(item.canonical_name, 0) == 0
        )


def exact_zero_skipped_dot(
    weights: Sequence[float], values: Sequence[float]
) -> tuple[float, float]:
    """Return scalar-loop dense and zero-skipped dot products in source order."""

    if len(weights) != len(values):
        raise ActivationSparsityError("dot-product lengths differ")
    dense = 0.0
    sparse = 0.0
    for weight, value in zip(weights, values):
        product = float(weight) * float(value)
        dense += product
        if float(value) != 0.0:
            sparse += product
    return dense, sparse


def weighted_percentile(
    rows: Iterable[ActivationCallAccounting],
    *,
    field_name: str,
    probability: float,
) -> float:
    items = sorted(
        (
            float(getattr(row, field_name)),
            int(row.dense_operation_terms),
        )
        for row in rows
    )
    if not items:
        raise ActivationSparsityError("weighted percentile requires calls")
    total = sum(weight for _, weight in items)
    target = max(1, math.ceil(probability * total))
    cumulative = 0
    for value, weight in items:
        cumulative += weight
        if cumulative >= target:
            return value
    raise AssertionError("unreachable weighted percentile")
