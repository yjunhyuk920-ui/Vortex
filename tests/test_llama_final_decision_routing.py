from __future__ import annotations

import math

import torch

from vortex_runtime.llama_final_decision_routing import (
    LlamaFinalDecisionRoutingModel,
    MicroRoutingConfig,
    codes_from_index,
    enumerate_micro_family,
    flatten_codes,
    signed_q4_level,
    target_routing_projection,
)


def test_signed_q4_codebook_is_complete_and_distinct() -> None:
    levels = [signed_q4_level(code) for code in range(16)]
    assert levels == [float(value) for value in range(-8, 8)]
    assert len(set(levels)) == 16


def test_causal_gqa_loaders_copy_disjoint_control_chunks() -> None:
    config = MicroRoutingConfig()
    model = LlamaFinalDecisionRoutingModel(
        codes_from_index(0, config),
        config,
    )
    assert model.loader_chunks == ((2, 3), (0, 1))

    input_ids = model.prompt_ids(0, 0, 0, 0)
    hidden = model.embedding[input_ids]

    first = model.loader_blocks[0](hidden)
    first_final = first[0, -1]
    assert first_final[model.payload_start] > 0
    assert first_final[model.output_selector_start] > 0
    assert first_final[model.selector_start] == 0
    assert torch.count_nonzero(
        model.loader_blocks[0].attention.v_proj
    ) == 2
    assert torch.count_nonzero(
        model.loader_blocks[0].attention.q_proj
    ) == 0
    assert torch.count_nonzero(
        model.loader_blocks[0].attention.k_proj
    ) == 0

    second = model.loader_blocks[1](first)
    second_final = second[0, -1]
    assert second_final[model.selector_start] > 0
    assert second_final[model.payload_start] == first_final[
        model.payload_start
    ]
    assert second_final[model.output_selector_start] == first_final[
        model.output_selector_start
    ]


def test_final_winner_decodes_signed_q4_codes_across_two_layers() -> None:
    config = MicroRoutingConfig()
    checkpoint_index = 15 + 16 * 3
    codes = codes_from_index(checkpoint_index, config)
    model = LlamaFinalDecisionRoutingModel(codes, config)

    signature = model.decision_signature()
    assert signature == (15, 3)
    assert signature == flatten_codes(codes)
    assert model.minimum_margin() > 0


def test_unselected_variable_layer_is_exactly_inactive() -> None:
    config = MicroRoutingConfig()
    codes = codes_from_index(15 + 16 * 7, config)
    model = LlamaFinalDecisionRoutingModel(codes, config)

    hidden = model._run_loaders(model.prompt_ids(1, 0, 0, 0))
    bit_coordinate = model.bit_output_start
    before = hidden[0, -1, bit_coordinate].clone()
    assert before == 0

    after_first = model.variable_blocks[0](hidden)
    assert after_first[0, -1, bit_coordinate] == before

    after_second = model.variable_blocks[1](after_first)
    assert after_second[0, -1, bit_coordinate] != before


def test_two_layer_end_to_end_family_is_injective() -> None:
    result = enumerate_micro_family(MicroRoutingConfig())
    assert result.checkpoint_coefficients == 2
    assert result.information_bits == 8
    assert result.expected_functions == 256
    assert result.observed_functions == 256
    assert result.exact_code_recovery
    assert result.minimum_winner_margin > 0
    assert result.passes


def test_target_llama_shaped_projection_exceeds_8gib() -> None:
    projection = target_routing_projection()
    assert projection.loader_layers == 15
    assert projection.variable_layers == 111
    assert projection.groups_per_layer == 31
    assert projection.neurons_per_group == 1717
    assert projection.payload_coordinates == 9508
    assert projection.active_intermediate_neurons == 53227
    assert projection.control_coordinates == 14666
    assert projection.checkpoint_coefficients == 56175137076
    assert projection.metadata_bits == 224700548304
    assert math.isclose(
        projection.metadata_gib,
        26.158586645498872,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert projection.vocabulary_rows == 42139
    assert projection.exceeds_resident_limit
    assert projection.vocabulary_pass
    assert projection.hidden_layout_pass
    assert projection.intermediate_pass
    assert projection.loader_capacity_pass
