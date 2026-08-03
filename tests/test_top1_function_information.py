from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.top1_function_information import (
    bit_table_from_index,
    decode_signature_bits,
    encode_selector_payload_classifier,
    enumerate_top1_function_family,
    llama_operator_collection_bound,
    top1_family_shape,
    top1_signature,
)


def test_selector_payload_signature_decodes_exact_bit_table() -> None:
    rows, columns = 4, 5
    shape = top1_family_shape(rows=rows, columns=columns)
    assert shape.row_pairs == 2
    assert shape.payload_columns == 3
    assert shape.decision_bits == 6
    table = bit_table_from_index(
        0b101101,
        row_pairs=shape.row_pairs,
        payload_columns=shape.payload_columns,
    )
    weight = encode_selector_payload_classifier(
        table,
        rows=rows,
        columns=columns,
    )
    signature, margin = top1_signature(weight)
    decoded = decode_signature_bits(
        signature,
        rows=rows,
        columns=columns,
    )
    assert torch.equal(decoded, table)
    assert margin == 1.0


@pytest.mark.parametrize(
    "rows,columns,expected_bits,expected_functions",
    [
        (2, 2, 1, 2),
        (4, 4, 4, 16),
        (4, 5, 6, 64),
        (6, 6, 9, 512),
    ],
)
def test_exhaustive_top1_family_is_injective(
    rows: int,
    columns: int,
    expected_bits: int,
    expected_functions: int,
) -> None:
    result = enumerate_top1_function_family(
        rows=rows,
        columns=columns,
        maximum_bits=10,
    )
    assert result.shape.decision_bits == expected_bits
    assert result.expected_functions == expected_functions
    assert result.enumerated_checkpoints == expected_functions
    assert result.observed_functions == expected_functions
    assert result.decoded_tables_match
    assert result.all_winners_unique
    assert result.minimum_winner_margin == 1.0
    assert result.injective
    assert result.passes


def test_square_family_encodes_one_quarter_bit_per_weight_coordinate() -> None:
    shape = top1_family_shape(rows=16_384, columns=16_384)
    assert shape.row_pairs == 8_192
    assert shape.payload_columns == 8_192
    assert shape.decision_bits == 67_108_864
    assert shape.matrix_parameters == 268_435_456
    assert shape.decision_bits_per_parameter == 0.25
    assert shape.metadata_mib == 8.0


def test_llama_405b_operator_collection_constants() -> None:
    bound = llama_operator_collection_bound()
    by_name = {item.name: item for item in bound.matrix_bounds}
    assert by_name["q_proj"].bits_per_copy == 67_108_864
    assert by_name["k_proj"].bits_per_copy == 8_126_464
    assert by_name["v_proj"].bits_per_copy == 8_126_464
    assert by_name["o_proj"].bits_per_copy == 67_108_864
    assert by_name["gate_proj"].bits_per_copy == 67_108_864
    assert by_name["up_proj"].bits_per_copy == 67_108_864
    assert by_name["down_proj"].bits_per_copy == 369_098_752
    assert bound.decoder_layer_bits == 653_787_136
    assert bound.decoder_layer_mib == 77.9375
    assert bound.decoder_stack_bits == 82_377_179_136
    assert math.isclose(
        bound.decoder_stack_gib,
        9.5899658203125,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert bound.lm_head_bits == 67_108_864
    assert bound.lm_head_mib == 8.0
    assert math.isclose(
        bound.total_gib,
        9.5977783203125,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert bound.exceeds_resident_limit
    assert bound.direct_classifier_bound_proven
    assert bound.independent_operator_collection_bound_proven
    assert not bound.full_transformer_top1_bound_proven


def test_exhaustive_enumeration_rejects_oversized_family() -> None:
    with pytest.raises(ValueError):
        enumerate_top1_function_family(
            rows=8,
            columns=8,
            maximum_bits=10,
        )


def test_family_requires_two_rows_and_columns() -> None:
    with pytest.raises(ValueError):
        top1_family_shape(rows=1, columns=4)
    with pytest.raises(ValueError):
        top1_family_shape(rows=4, columns=1)
