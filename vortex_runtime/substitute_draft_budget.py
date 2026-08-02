from __future__ import annotations

from dataclasses import asdict, dataclass

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class SubstituteDraftBudget:
    retained_layers: int
    total_layers: int
    weight_bits: int
    tie_word_embeddings: bool
    per_layer_parameters: int
    io_parameters: int
    retained_parameters: int
    weight_gib: float
    workspace_gib: float
    total_gib: float
    memory_limit_gib: float
    fits_memory: bool
    average_bits_per_target_parameter: float

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def llama_layer_parameter_count(model: ModelSpec) -> int:
    """Return the dense decoder-layer parameter count for a Llama-like model."""

    hidden = model.hidden_size
    intermediate = model.intermediate_size
    kv = model.kv_dim
    attention = hidden * hidden + 2 * kv * hidden + hidden * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    return attention + mlp + norms


def llama_io_parameter_count(
    model: ModelSpec,
    *,
    tie_word_embeddings: bool,
) -> int:
    embedding = model.vocab_size * model.hidden_size
    lm_head = 0 if tie_word_embeddings else embedding
    final_norm = model.hidden_size
    return embedding + lm_head + final_norm


def substitute_draft_budget(
    *,
    model: ModelSpec,
    retained_layers: int,
    weight_bits: int = 4,
    tie_word_embeddings: bool = False,
    workspace_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
) -> SubstituteDraftBudget:
    """Budget a target-derived layer-thinned draft kept entirely on the GPU.

    The draft retains the target tokenizer, embeddings, final normalization and
    LM head, but executes only a selected subset of decoder layers. No learned
    adapter or retraining is assumed. This is a weight-memory floor; KV cache,
    allocator fragmentation and kernel metadata must fit inside `workspace_gib`.
    """

    if not 0 < retained_layers <= model.layers:
        raise ValueError("retained_layers must be in [1, model.layers]")
    if weight_bits <= 0:
        raise ValueError("weight_bits must be positive")
    if workspace_gib < 0 or memory_limit_gib <= 0:
        raise ValueError("workspace must be non-negative and memory limit positive")

    per_layer = llama_layer_parameter_count(model)
    io_parameters = llama_io_parameter_count(
        model,
        tie_word_embeddings=tie_word_embeddings,
    )
    retained_parameters = io_parameters + retained_layers * per_layer
    weight_gib = retained_parameters * weight_bits / 8 / GIB
    total_gib = weight_gib + workspace_gib
    average_bits = retained_parameters * weight_bits / model.parameters

    return SubstituteDraftBudget(
        retained_layers=retained_layers,
        total_layers=model.layers,
        weight_bits=weight_bits,
        tie_word_embeddings=tie_word_embeddings,
        per_layer_parameters=per_layer,
        io_parameters=io_parameters,
        retained_parameters=retained_parameters,
        weight_gib=weight_gib,
        workspace_gib=workspace_gib,
        total_gib=total_gib,
        memory_limit_gib=memory_limit_gib,
        fits_memory=total_gib <= memory_limit_gib,
        average_bits_per_target_parameter=average_bits,
    )


def maximum_retained_layers(
    *,
    model: ModelSpec,
    weight_bits: int = 4,
    tie_word_embeddings: bool = False,
    workspace_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
) -> int:
    feasible = [
        layers
        for layers in range(1, model.layers + 1)
        if substitute_draft_budget(
            model=model,
            retained_layers=layers,
            weight_bits=weight_bits,
            tie_word_embeddings=tie_word_embeddings,
            workspace_gib=workspace_gib,
            memory_limit_gib=memory_limit_gib,
        ).fits_memory
    ]
    return max(feasible, default=0)


def select_layer_indices(
    *,
    total_layers: int,
    retained_layers: int,
    strategy: str,
) -> tuple[int, ...]:
    """Select deterministic target layers for a training-free thinned draft."""

    if not 0 < retained_layers <= total_layers:
        raise ValueError("retained_layers must be in [1, total_layers]")
    if strategy == "front":
        return tuple(range(retained_layers))
    if strategy == "uniform":
        if retained_layers == 1:
            return (total_layers - 1,)
        indices = {
            round(index * (total_layers - 1) / (retained_layers - 1))
            for index in range(retained_layers)
        }
        if len(indices) != retained_layers:
            raise RuntimeError("uniform layer selection produced duplicate indices")
        return tuple(sorted(indices))
    if strategy == "edge":
        front_count = (retained_layers + 1) // 2
        back_count = retained_layers - front_count
        front = list(range(front_count))
        back = list(range(total_layers - back_count, total_layers))
        result = tuple(front + back)
        if len(set(result)) != retained_layers:
            raise RuntimeError("edge layer selection produced duplicate indices")
        return result
    raise ValueError(f"unsupported layer selection strategy: {strategy}")
