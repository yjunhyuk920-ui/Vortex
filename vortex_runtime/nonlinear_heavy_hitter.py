from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from vortex_runtime.mlp_heavy_hitter import OracleHeavyHitterSwiGLU


@dataclass(frozen=True)
class LayerDamagePoint:
    selected_neurons: int
    damage: float
    top1_rate: float
    top32_rate: float
    output_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class NonlinearAllocation:
    total_budget: int
    used_neurons: int
    layer_counts: tuple[int, ...]
    predicted_total_damage: float
    active_layers: int
    minimum_count: int
    maximum_count: int

    def to_dict(self) -> dict[str, int | float | list[int]]:
        payload = asdict(self)
        payload["layer_counts"] = list(self.layer_counts)
        return payload


def uniform_neuron_allocation(
    *,
    layers: int,
    intermediate_neurons: int,
    total_neurons: int,
) -> tuple[int, ...]:
    """Distribute an exact total neuron budget as evenly as possible."""

    if min(layers, intermediate_neurons, total_neurons) <= 0:
        raise ValueError("allocation dimensions must be positive")
    if total_neurons > layers * intermediate_neurons:
        raise ValueError("total_neurons exceeds model capacity")
    base, remainder = divmod(total_neurons, layers)
    if base > intermediate_neurons or (base == intermediate_neurons and remainder):
        raise ValueError("uniform allocation exceeds layer capacity")
    return tuple(base + int(index < remainder) for index in range(layers))


def replace_llama_mlp_with_count_allocation(
    model: nn.Module,
    *,
    layer_counts: tuple[int, ...] | list[int],
) -> list[OracleHeavyHitterSwiGLU]:
    """Replace every Llama MLP with an exact-activation original-neuron oracle.

    The helper lives in this module deliberately: research workflows must be
    branch-standalone and must not import execution primitives from a sibling
    experiment branch.
    """

    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")
    if len(layer_counts) != len(layers):
        raise ValueError("one neuron count is required per decoder layer")

    replacements: list[OracleHeavyHitterSwiGLU] = []
    for layer, requested_count in zip(layers, layer_counts):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("decoder layer has no mlp module")
        intermediate = int(mlp.gate_proj.out_features)
        count = int(requested_count)
        if count <= 0 or count > intermediate:
            raise ValueError("each active layer count must be in [1, intermediate]")
        replacement = OracleHeavyHitterSwiGLU(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            act_fn=mlp.act_fn,
            selected_fraction=count / intermediate,
        )
        layer.mlp = replacement
        replacements.append(replacement)
    return replacements


def normalize_damage_curves(
    curves: list[list[LayerDamagePoint]],
) -> list[list[LayerDamagePoint]]:
    if not curves:
        raise ValueError("at least one layer curve is required")
    normalized: list[list[LayerDamagePoint]] = []
    for layer_index, curve in enumerate(curves):
        if not curve:
            raise ValueError(f"layer {layer_index} has no damage points")
        by_count: dict[int, LayerDamagePoint] = {}
        for point in curve:
            if point.selected_neurons <= 0:
                raise ValueError("selected-neuron counts must be positive")
            if point.damage < 0:
                raise ValueError("damage values must be nonnegative")
            current = by_count.get(point.selected_neurons)
            if current is None or point.damage < current.damage:
                by_count[point.selected_neurons] = point
        ordered = [by_count[count] for count in sorted(by_count)]
        # At a measured cost c, the allocator may reuse the best option observed
        # at a cheaper or equal cost and leave extra budget unused. It may never
        # borrow the quality of a more expensive point.
        best_point: LayerDamagePoint | None = None
        envelope: list[LayerDamagePoint] = []
        for point in ordered:
            if best_point is None or point.damage < best_point.damage:
                best_point = point
            envelope.append(
                LayerDamagePoint(
                    selected_neurons=point.selected_neurons,
                    damage=best_point.damage,
                    top1_rate=best_point.top1_rate,
                    top32_rate=best_point.top32_rate,
                    output_error=best_point.output_error,
                )
            )
        normalized.append(envelope)
    return normalized


def solve_nonlinear_allocation(
    curves: list[list[LayerDamagePoint]],
    *,
    total_budget: int,
) -> NonlinearAllocation:
    """Solve the measured byte-constrained layer allocation exactly.

    Every layer chooses one measured neuron-count option. Costs are counts and
    losses are measured nonlinear final-logit damages. Dynamic programming finds
    the minimum predicted total damage using no more than ``total_budget``.
    """

    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    normalized = normalize_damage_curves(curves)
    minimum_required = sum(curve[0].selected_neurons for curve in normalized)
    if minimum_required > total_budget:
        raise ValueError("total budget is below the minimum measured allocation")

    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for curve in normalized:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (damage, counts) in states.items():
            for point in curve:
                new_used = used + point.selected_neurons
                if new_used > total_budget:
                    continue
                candidate = (damage + point.damage, counts + (point.selected_neurons,))
                current = next_states.get(new_used)
                if current is None or candidate[0] < current[0]:
                    next_states[new_used] = candidate
        if not next_states:
            raise RuntimeError("damage curves produced no feasible allocation")
        states = next_states

    used, (damage, counts) = min(
        states.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    return NonlinearAllocation(
        total_budget=total_budget,
        used_neurons=used,
        layer_counts=counts,
        predicted_total_damage=damage,
        active_layers=sum(count > 0 for count in counts),
        minimum_count=min(counts),
        maximum_count=max(counts),
    )
