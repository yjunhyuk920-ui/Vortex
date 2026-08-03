from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

GIB = 1024**3


@dataclass(frozen=True)
class MicroRoutingConfig:
    hidden_size: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 2
    total_layers: int = 4
    loader_layers: int = 2
    groups_per_layer: int = 1
    neurons_per_group: int = 1
    payload_coordinates: int = 1
    code_levels: int = 16
    rms_eps: float = 1e-6
    gate_value: float = 1.0

    @property
    def variable_layers(self) -> int:
        return self.total_layers - self.loader_layers

    @property
    def kv_dimension(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def checkpoint_coefficients(self) -> int:
        return (
            self.variable_layers
            * self.groups_per_layer
            * self.neurons_per_group
            * self.payload_coordinates
        )

    @property
    def checkpoint_information_bits(self) -> int:
        bits_per_code = (self.code_levels - 1).bit_length()
        return self.checkpoint_coefficients * bits_per_code

    @property
    def expected_function_count(self) -> int:
        return self.code_levels**self.checkpoint_coefficients


@dataclass(frozen=True)
class TargetRoutingProjection:
    hidden_size: int
    intermediate_size: int
    total_layers: int
    loader_layers: int
    variable_layers: int
    kv_dimension: int
    groups_per_layer: int
    neurons_per_group: int
    payload_coordinates: int
    active_intermediate_neurons: int
    control_coordinates: int
    answer_coordinates: int
    code_levels: int
    bits_per_code: int
    checkpoint_coefficients: int
    metadata_bits: int
    metadata_gib: float
    vocabulary_rows: int
    vocabulary_limit: int
    resident_limit_gib: float
    exceeds_resident_limit: bool
    vocabulary_pass: bool
    hidden_layout_pass: bool
    intermediate_pass: bool
    loader_capacity_pass: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FamilyEnumeration:
    checkpoint_coefficients: int
    information_bits: int
    expected_functions: int
    observed_functions: int
    minimum_winner_margin: float
    exact_code_recovery: bool
    passes: bool

    def to_dict(self) -> dict:
        return asdict(self)


def signed_q4_level(code: int, code_levels: int = 16) -> float:
    if code_levels != 16:
        raise ValueError("the current certificate uses the 16 signed Q4 levels")
    if not 0 <= code < code_levels:
        raise ValueError("code is outside the Q4 codebook")
    return float(code - 8)


def target_routing_projection(
    *,
    hidden_size: int = 16384,
    intermediate_size: int = 53248,
    total_layers: int = 126,
    kv_dimension: int = 1024,
    code_levels: int = 16,
    vocabulary_limit: int = 128256,
    resident_limit_gib: float = 8.0,
) -> TargetRoutingProjection:
    """Find the strongest certificate inside the Llama-shaped envelope."""
    if min(
        hidden_size,
        intermediate_size,
        total_layers,
        kv_dimension,
        code_levels,
        vocabulary_limit,
    ) <= 0:
        raise ValueError("projection dimensions must be positive")
    bits_per_code = (code_levels - 1).bit_length()
    best: TargetRoutingProjection | None = None

    for neurons_per_group in range(
        1,
        min(intermediate_size, hidden_size // 2) + 1,
    ):
        control_coordinates = hidden_size - neurons_per_group - 1
        loader_layers = ceil(control_coordinates / kv_dimension)
        variable_layers = total_layers - loader_layers
        if variable_layers <= 0:
            continue

        remaining_after_outputs = (
            hidden_size - 1 - 2 * neurons_per_group
        )
        if remaining_after_outputs <= variable_layers:
            continue

        maximum_groups = min(
            intermediate_size // neurons_per_group,
            (remaining_after_outputs - 1) // variable_layers,
        )
        if maximum_groups <= 0:
            continue

        ideal_groups = remaining_after_outputs // (
            2 * variable_layers
        )
        candidates = {
            1,
            maximum_groups,
            ideal_groups - 1,
            ideal_groups,
            ideal_groups + 1,
        }
        for groups_per_layer in candidates:
            if not 1 <= groups_per_layer <= maximum_groups:
                continue
            payload_coordinates = (
                remaining_after_outputs
                - variable_layers * groups_per_layer
            )
            if payload_coordinates <= 0:
                continue

            active_intermediate = (
                groups_per_layer * neurons_per_group
            )
            checkpoint_coefficients = (
                variable_layers
                * groups_per_layer
                * neurons_per_group
                * payload_coordinates
            )
            metadata_bits = checkpoint_coefficients * bits_per_code
            metadata_gib = metadata_bits / 8 / GIB
            vocabulary_rows = (
                code_levels * neurons_per_group
                + variable_layers * groups_per_layer
                + payload_coordinates
                + neurons_per_group
                + 1
            )
            candidate = TargetRoutingProjection(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                total_layers=total_layers,
                loader_layers=loader_layers,
                variable_layers=variable_layers,
                kv_dimension=kv_dimension,
                groups_per_layer=groups_per_layer,
                neurons_per_group=neurons_per_group,
                payload_coordinates=payload_coordinates,
                active_intermediate_neurons=active_intermediate,
                control_coordinates=control_coordinates,
                answer_coordinates=neurons_per_group,
                code_levels=code_levels,
                bits_per_code=bits_per_code,
                checkpoint_coefficients=checkpoint_coefficients,
                metadata_bits=metadata_bits,
                metadata_gib=metadata_gib,
                vocabulary_rows=vocabulary_rows,
                vocabulary_limit=vocabulary_limit,
                resident_limit_gib=resident_limit_gib,
                exceeds_resident_limit=(
                    metadata_gib > resident_limit_gib
                ),
                vocabulary_pass=(
                    vocabulary_rows <= vocabulary_limit
                ),
                hidden_layout_pass=(
                    variable_layers * groups_per_layer
                    + payload_coordinates
                    + 2 * neurons_per_group
                    + 1
                    <= hidden_size
                ),
                intermediate_pass=(
                    active_intermediate <= intermediate_size
                ),
                loader_capacity_pass=(
                    loader_layers * kv_dimension
                    >= control_coordinates
                ),
            )
            if not candidate.vocabulary_pass:
                continue
            if best is None or (
                candidate.metadata_bits > best.metadata_bits
            ):
                best = candidate

    if best is None:
        raise RuntimeError("no valid routing projection fits the envelope")
    return best


def codes_from_index(
    index: int,
    config: MicroRoutingConfig,
) -> list[list[list[list[int]]]]:
    if index < 0:
        raise ValueError("index must be nonnegative")
    codes: list[list[list[list[int]]]] = []
    remaining = index
    for _layer in range(config.variable_layers):
        layer: list[list[list[int]]] = []
        for _group in range(config.groups_per_layer):
            group: list[list[int]] = []
            for _neuron in range(config.neurons_per_group):
                payload: list[int] = []
                for _coordinate in range(
                    config.payload_coordinates
                ):
                    payload.append(
                        remaining % config.code_levels
                    )
                    remaining //= config.code_levels
                group.append(payload)
            layer.append(group)
        codes.append(layer)
    if remaining:
        raise ValueError(
            "index exceeds the configured checkpoint family"
        )
    return codes


def flatten_codes(codes: Sequence) -> tuple[int, ...]:
    flattened: list[int] = []
    for layer in codes:
        for group in layer:
            for neuron in group:
                flattened.extend(int(code) for code in neuron)
    return tuple(flattened)


def _validate_codes(
    codes: Sequence,
    config: MicroRoutingConfig,
) -> None:
    if len(codes) != config.variable_layers:
        raise ValueError(
            "code table has the wrong number of variable layers"
        )
    for layer in codes:
        if len(layer) != config.groups_per_layer:
            raise ValueError("code table has the wrong group count")
        for group in layer:
            if len(group) != config.neurons_per_group:
                raise ValueError(
                    "code table has the wrong neuron count"
                )
            for neuron in group:
                if len(neuron) != config.payload_coordinates:
                    raise ValueError(
                        "code table has the wrong payload count"
                    )
                for code in neuron:
                    if not 0 <= int(code) < config.code_levels:
                        raise ValueError(
                            "code table contains an invalid Q4 code"
                        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "weight",
            torch.ones(hidden_size, dtype=torch.float64),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.square().mean(
            dim=-1,
            keepdim=True,
        )
        return (
            hidden_states
            * torch.rsqrt(variance + self.eps)
            * self.weight
        )


class CausalGQAAttention(nn.Module):
    """Bias-free causal GQA with zero Q/K and chunked V/O loading."""

    def __init__(
        self,
        config: MicroRoutingConfig,
        assigned_coordinates: Iterable[int],
    ) -> None:
        super().__init__()
        if (
            config.num_attention_heads * config.head_dim
            != config.hidden_size
        ):
            raise ValueError(
                "attention heads do not cover the hidden size"
            )
        if (
            config.num_attention_heads
            % config.num_key_value_heads
        ):
            raise ValueError("GQA head ratio must be integral")

        coordinates = tuple(
            int(value) for value in assigned_coordinates
        )
        if len(coordinates) > config.kv_dimension:
            raise ValueError(
                "one loader exceeds the KV projection dimension"
            )
        self.config = config
        self.assigned_coordinates = coordinates

        hidden = config.hidden_size
        kv_dimension = config.kv_dimension
        q_proj = torch.zeros(
            hidden,
            hidden,
            dtype=torch.float64,
        )
        k_proj = torch.zeros(
            kv_dimension,
            hidden,
            dtype=torch.float64,
        )
        v_proj = torch.zeros(
            kv_dimension,
            hidden,
            dtype=torch.float64,
        )
        o_proj = torch.zeros(
            hidden,
            hidden,
            dtype=torch.float64,
        )

        for value_index, coordinate in enumerate(coordinates):
            v_proj[value_index, coordinate] = 1.0
            o_proj[coordinate, value_index] = 1.0

        self.register_buffer("q_proj", q_proj)
        self.register_buffer("k_proj", k_proj)
        self.register_buffer("v_proj", v_proj)
        self.register_buffer("o_proj", o_proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = hidden_states.shape
        config = self.config

        query = (hidden_states @ self.q_proj.T).view(
            batch,
            sequence,
            config.num_attention_heads,
            config.head_dim,
        ).transpose(1, 2)
        key = (hidden_states @ self.k_proj.T).view(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dim,
        ).transpose(1, 2)
        value = (hidden_states @ self.v_proj.T).view(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dim,
        ).transpose(1, 2)

        repeats = (
            config.num_attention_heads
            // config.num_key_value_heads
        )
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

        scores = query @ key.transpose(-1, -2) / sqrt(
            config.head_dim
        )
        mask = torch.triu(
            torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        context = (
            probabilities @ value
        ).transpose(1, 2).reshape(batch, sequence, hidden)
        return context @ self.o_proj.T


class ZeroMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class CodeSwiGLU(nn.Module):
    def __init__(
        self,
        *,
        config: MicroRoutingConfig,
        logical_layer: int,
        selector_start: int,
        payload_start: int,
        bit_output_start: int,
        layer_codes: Sequence,
        selector_amplitudes: Sequence[float],
        payload_amplitudes: Sequence[Sequence[float]],
    ) -> None:
        super().__init__()
        groups = config.groups_per_layer
        neurons = config.neurons_per_group
        payloads = config.payload_coordinates
        intermediate = groups * neurons
        hidden = config.hidden_size

        gate_proj = torch.zeros(
            intermediate,
            hidden,
            dtype=torch.float64,
        )
        up_proj = torch.zeros(
            intermediate,
            hidden,
            dtype=torch.float64,
        )
        down_proj = torch.zeros(
            hidden,
            intermediate,
            dtype=torch.float64,
        )

        for group_index in range(groups):
            selector_coordinate = (
                selector_start
                + logical_layer * groups
                + group_index
            )
            selector_amplitude = float(
                selector_amplitudes[group_index]
            )
            if selector_amplitude <= 0:
                raise ValueError(
                    "selector calibration must be positive"
                )

            for neuron_index in range(neurons):
                intermediate_index = (
                    group_index * neurons + neuron_index
                )
                gate_proj[
                    intermediate_index,
                    selector_coordinate,
                ] = config.gate_value / selector_amplitude

                for payload_index in range(payloads):
                    payload_amplitude = float(
                        payload_amplitudes[group_index][
                            payload_index
                        ]
                    )
                    if payload_amplitude <= 0:
                        raise ValueError(
                            "payload calibration must be positive"
                        )
                    level = signed_q4_level(
                        int(
                            layer_codes[group_index][
                                neuron_index
                            ][payload_index]
                        ),
                        config.code_levels,
                    )
                    up_proj[
                        intermediate_index,
                        payload_start + payload_index,
                    ] = level / payload_amplitude

                down_proj[
                    bit_output_start + neuron_index,
                    intermediate_index,
                ] = 1.0

        self.register_buffer("gate_proj", gate_proj)
        self.register_buffer("up_proj", up_proj)
        self.register_buffer("down_proj", down_proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = hidden_states @ self.gate_proj.T
        up = hidden_states @ self.up_proj.T
        return (F.silu(gate) * up) @ self.down_proj.T


class RoutingBlock(nn.Module):
    def __init__(
        self,
        *,
        config: MicroRoutingConfig,
        attention: CausalGQAAttention,
        mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_eps,
        )
        self.attention = attention
        self.mlp_norm = RMSNorm(
            config.hidden_size,
            config.rms_eps,
        )
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.mlp(
            self.mlp_norm(hidden_states)
        )
        return hidden_states


class LlamaFinalDecisionRoutingModel(nn.Module):
    """Small deterministic Llama-style final-token routing family."""

    def __init__(
        self,
        codes: Sequence,
        config: MicroRoutingConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MicroRoutingConfig()
        config = self.config
        _validate_codes(codes, config)

        if config.variable_layers <= 0:
            raise ValueError(
                "at least one variable layer is required"
            )
        if config.code_levels != 16:
            raise ValueError(
                "the executable certificate requires Q4"
            )

        selector_start = 0
        selector_count = (
            config.variable_layers * config.groups_per_layer
        )
        payload_start = selector_start + selector_count
        output_selector_start = (
            payload_start + config.payload_coordinates
        )
        bit_output_start = (
            output_selector_start + config.neurons_per_group
        )
        carrier_coordinate = (
            bit_output_start + config.neurons_per_group
        )
        if carrier_coordinate + 1 > config.hidden_size:
            raise ValueError(
                "routing coordinates exceed hidden size"
            )

        self.selector_start = selector_start
        self.payload_start = payload_start
        self.output_selector_start = output_selector_start
        self.bit_output_start = bit_output_start
        self.carrier_coordinate = carrier_coordinate

        control_order = list(
            range(
                payload_start,
                output_selector_start
                + config.neurons_per_group,
            )
        ) + list(range(selector_start, payload_start))
        required_loaders = ceil(
            len(control_order) / config.kv_dimension
        )
        if required_loaders > config.loader_layers:
            raise ValueError(
                "loader layers cannot cover all controls"
            )
        chunks = [
            control_order[
                index : index + config.kv_dimension
            ]
            for index in range(
                0,
                len(control_order),
                config.kv_dimension,
            )
        ]
        chunks.extend(
            []
            for _ in range(
                config.loader_layers - len(chunks)
            )
        )
        self.loader_chunks = tuple(
            tuple(chunk) for chunk in chunks
        )

        answer_rows = (
            config.code_levels * config.neurons_per_group
        )
        selector_token_count = selector_count
        payload_token_count = config.payload_coordinates
        output_token_count = config.neurons_per_group

        self.answer_rows = answer_rows
        self.selector_token_start = answer_rows
        self.payload_token_start = (
            self.selector_token_start + selector_token_count
        )
        self.output_token_start = (
            self.payload_token_start + payload_token_count
        )
        self.query_token_id = (
            self.output_token_start + output_token_count
        )
        self.vocabulary_size = self.query_token_id + 1

        embedding = torch.zeros(
            self.vocabulary_size,
            config.hidden_size,
            dtype=torch.float64,
        )
        for selector_index in range(selector_token_count):
            embedding[
                self.selector_token_start + selector_index,
                selector_start + selector_index,
            ] = 1.0
        for payload_index in range(payload_token_count):
            embedding[
                self.payload_token_start + payload_index,
                payload_start + payload_index,
            ] = 1.0
        for output_index in range(output_token_count):
            embedding[
                self.output_token_start + output_index,
                output_selector_start + output_index,
            ] = 1.0
        embedding[
            self.query_token_id,
            carrier_coordinate,
        ] = 1.0
        self.register_buffer("embedding", embedding)

        self.loader_blocks = nn.ModuleList(
            [
                RoutingBlock(
                    config=config,
                    attention=CausalGQAAttention(
                        config,
                        chunk,
                    ),
                    mlp=ZeroMLP(),
                )
                for chunk in chunks
            ]
        )

        calibrations = self._collect_calibrations()
        selector_calibrations = calibrations["selectors"]
        payload_calibrations = calibrations["payloads"]
        output_calibrations = calibrations["outputs"]
        carrier_amplitude = float(calibrations["carrier"])

        variable_blocks: list[RoutingBlock] = []
        for logical_layer in range(config.variable_layers):
            layer_selectors: list[float] = []
            layer_payloads: list[list[float]] = []
            for group_index in range(
                config.groups_per_layer
            ):
                key = (logical_layer, group_index)
                layer_selectors.append(
                    float(selector_calibrations[key])
                )
                layer_payloads.append(
                    [
                        float(
                            payload_calibrations[
                                (key, payload)
                            ]
                        )
                        for payload in range(
                            config.payload_coordinates
                        )
                    ]
                )
            variable_blocks.append(
                RoutingBlock(
                    config=config,
                    attention=CausalGQAAttention(
                        config,
                        (),
                    ),
                    mlp=CodeSwiGLU(
                        config=config,
                        logical_layer=logical_layer,
                        selector_start=selector_start,
                        payload_start=payload_start,
                        bit_output_start=bit_output_start,
                        layer_codes=codes[logical_layer],
                        selector_amplitudes=layer_selectors,
                        payload_amplitudes=layer_payloads,
                    ),
                )
            )
        self.variable_blocks = nn.ModuleList(variable_blocks)
        self.final_norm = RMSNorm(
            config.hidden_size,
            config.rms_eps,
        )

        gate_scale = float(
            F.silu(torch.tensor(config.gate_value))
        )
        centers = torch.tensor(
            [
                gate_scale
                * signed_q4_level(
                    code,
                    config.code_levels,
                )
                for code in range(config.code_levels)
            ],
            dtype=torch.float64,
        )
        maximum_center = float(centers.abs().max())
        baseline = 2.0 * maximum_center**2 + 1.0

        lm_head = torch.zeros(
            self.vocabulary_size,
            config.hidden_size,
            dtype=torch.float64,
        )
        for output_index in range(
            config.neurons_per_group
        ):
            output_amplitude = float(
                output_calibrations[output_index]
            )
            if output_amplitude <= 0:
                raise ValueError(
                    "output-selector calibration must be positive"
                )
            for code, center in enumerate(
                centers.tolist()
            ):
                row = (
                    output_index * config.code_levels + code
                )
                lm_head[
                    row,
                    output_selector_start + output_index,
                ] = baseline / output_amplitude
                lm_head[
                    row,
                    bit_output_start + output_index,
                ] = 2.0 * center
                lm_head[
                    row,
                    carrier_coordinate,
                ] = -(center**2) / carrier_amplitude

        self.register_buffer("lm_head", lm_head)
        self.register_buffer("code_centers", centers)
        self.baseline = baseline
        self.calibrations = calibrations

    def prompt_ids(
        self,
        logical_layer: int,
        group_index: int,
        payload_index: int,
        output_index: int,
    ) -> torch.Tensor:
        config = self.config
        if not 0 <= logical_layer < config.variable_layers:
            raise ValueError(
                "logical layer is outside the variable stack"
            )
        if not 0 <= group_index < config.groups_per_layer:
            raise ValueError("group index is invalid")
        if not 0 <= payload_index < config.payload_coordinates:
            raise ValueError("payload index is invalid")
        if not 0 <= output_index < config.neurons_per_group:
            raise ValueError("output index is invalid")

        selector_index = (
            logical_layer * config.groups_per_layer
            + group_index
        )
        return torch.tensor(
            [
                [
                    self.selector_token_start + selector_index,
                    self.payload_token_start + payload_index,
                    self.output_token_start + output_index,
                    self.query_token_id,
                ]
            ],
            dtype=torch.long,
        )

    def _run_loaders(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embedding[input_ids]
        for block in self.loader_blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def _collect_calibrations(self) -> dict:
        config = self.config
        norm = RMSNorm(config.hidden_size, config.rms_eps)
        selectors: dict[tuple[int, int], float] = {}
        payloads: dict[
            tuple[tuple[int, int], int],
            float,
        ] = {}
        outputs: dict[int, float] = {}
        carrier_values: list[float] = []

        with torch.no_grad():
            for logical_layer in range(
                config.variable_layers
            ):
                for group_index in range(
                    config.groups_per_layer
                ):
                    key = (logical_layer, group_index)
                    for payload_index in range(
                        config.payload_coordinates
                    ):
                        input_ids = self.prompt_ids(
                            logical_layer,
                            group_index,
                            payload_index,
                            0,
                        )
                        final_state = self._run_loaders(
                            input_ids
                        )[0, -1]
                        normalized = norm(final_state)
                        selector_coordinate = (
                            self.selector_start
                            + logical_layer
                            * config.groups_per_layer
                            + group_index
                        )
                        selectors[key] = float(
                            normalized[selector_coordinate]
                        )
                        payloads[
                            (key, payload_index)
                        ] = float(
                            normalized[
                                self.payload_start
                                + payload_index
                            ]
                        )
                        carrier_values.append(
                            float(
                                final_state[
                                    self.carrier_coordinate
                                ]
                            )
                        )

            for output_index in range(
                config.neurons_per_group
            ):
                input_ids = self.prompt_ids(
                    0,
                    0,
                    0,
                    output_index,
                )
                final_state = self._run_loaders(
                    input_ids
                )[0, -1]
                outputs[output_index] = float(
                    final_state[
                        self.output_selector_start
                        + output_index
                    ]
                )

        carrier = carrier_values[0]
        if any(
            abs(value - carrier) > 1e-12
            for value in carrier_values
        ):
            raise RuntimeError(
                "carrier calibration changed across prompts"
            )
        return {
            "selectors": selectors,
            "payloads": payloads,
            "outputs": outputs,
            "carrier": carrier,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._run_loaders(input_ids)
        for block in self.variable_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states) @ self.lm_head.T

    def winner(
        self,
        logical_layer: int,
        group_index: int,
        payload_index: int,
        output_index: int,
    ) -> tuple[int, float]:
        logits = self.forward(
            self.prompt_ids(
                logical_layer,
                group_index,
                payload_index,
                output_index,
            )
        )[0, -1]
        top = torch.topk(logits, 2)
        return (
            int(top.indices[0]),
            float(
                (top.values[0] - top.values[1]).item()
            ),
        )

    def decision_signature(self) -> tuple[int, ...]:
        config = self.config
        signature: list[int] = []
        for logical_layer in range(
            config.variable_layers
        ):
            for group_index in range(
                config.groups_per_layer
            ):
                for output_index in range(
                    config.neurons_per_group
                ):
                    for payload_index in range(
                        config.payload_coordinates
                    ):
                        winner, _ = self.winner(
                            logical_layer,
                            group_index,
                            payload_index,
                            output_index,
                        )
                        signature.append(
                            winner
                            - output_index
                            * config.code_levels
                        )
        return tuple(signature)

    def minimum_margin(self) -> float:
        config = self.config
        margins: list[float] = []
        for logical_layer in range(
            config.variable_layers
        ):
            for group_index in range(
                config.groups_per_layer
            ):
                for output_index in range(
                    config.neurons_per_group
                ):
                    for payload_index in range(
                        config.payload_coordinates
                    ):
                        _, margin = self.winner(
                            logical_layer,
                            group_index,
                            payload_index,
                            output_index,
                        )
                        margins.append(margin)
        return min(margins)


def enumerate_micro_family(
    config: MicroRoutingConfig | None = None,
) -> FamilyEnumeration:
    config = config or MicroRoutingConfig()
    expected = config.expected_function_count
    signatures: set[tuple[int, ...]] = set()
    minimum_margin = float("inf")
    exact_recovery = True

    for checkpoint_index in range(expected):
        codes = codes_from_index(checkpoint_index, config)
        model = LlamaFinalDecisionRoutingModel(
            codes,
            config,
        )
        signature = model.decision_signature()
        signatures.add(signature)
        minimum_margin = min(
            minimum_margin,
            model.minimum_margin(),
        )
        exact_recovery = exact_recovery and (
            signature == flatten_codes(codes)
        )

    observed = len(signatures)
    passes = (
        observed == expected
        and exact_recovery
        and minimum_margin > 0
    )
    return FamilyEnumeration(
        checkpoint_coefficients=(
            config.checkpoint_coefficients
        ),
        information_bits=config.checkpoint_information_bits,
        expected_functions=expected,
        observed_functions=observed,
        minimum_winner_margin=minimum_margin,
        exact_code_recovery=exact_recovery,
        passes=passes,
    )
