from __future__ import annotations

import torch

from vortex_runtime.feasibility import default_specs
from vortex_runtime.nested_precision import (
    nested_bitplane_budget,
    nested_symmetric_per_row_fake_quantize,
)


def test_nested_levels_add_low_bitplanes_without_changing_scale_family() -> None:
    weight = torch.tensor(
        [[1.0, -0.72, 0.31, -0.09], [0.5, -0.4, 0.2, -0.1]],
        dtype=torch.float32,
    )
    q6, stats6 = nested_symmetric_per_row_fake_quantize(
        weight,
        bits=6,
        maximum_bits=8,
        row_chunk=1,
    )
    q7, stats7 = nested_symmetric_per_row_fake_quantize(
        weight,
        bits=7,
        maximum_bits=8,
        row_chunk=1,
    )
    q8, stats8 = nested_symmetric_per_row_fake_quantize(
        weight,
        bits=8,
        maximum_bits=8,
        row_chunk=1,
    )
    assert q6.shape == q7.shape == q8.shape == weight.shape
    assert stats8.relative_l2_error <= stats7.relative_l2_error
    assert stats7.relative_l2_error <= stats6.relative_l2_error
    assert torch.linalg.vector_norm(q8 - weight) <= torch.linalg.vector_norm(q6 - weight)


def test_nested_budget_charges_full_stream_only_to_highest_reached_bit() -> None:
    target, baseline = default_specs()
    point = nested_bitplane_budget(
        target=target,
        baseline=baseline,
        base_bits=6,
        maximum_bits=8,
        block_positions=4096,
        fractions_reaching_bits={7: 0.25, 8: 0.1},
    )
    assert point.maximum_reached_bits == 8
    assert point.weight_stream_gib > 370
    assert point.incremental_compute_seconds_per_block > 0
    assert point.fractions_reaching_bits[8] <= point.fractions_reaching_bits[7]


def test_nested_budget_rejects_increasing_later_fraction() -> None:
    target, baseline = default_specs()
    try:
        nested_bitplane_budget(
            target=target,
            baseline=baseline,
            base_bits=6,
            maximum_bits=8,
            block_positions=4096,
            fractions_reaching_bits={7: 0.1, 8: 0.2},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("later precision fraction must not increase")
