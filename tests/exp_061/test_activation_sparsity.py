from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.activation_sparsity import (
    ActivationSparsityError,
    ActivationSparsityRecorder,
    account_activation_call,
    exact_zero_skipped_dot,
    register_linear_projections,
    weighted_percentile,
)


def test_accounting_counts_scan_weight_columns_and_metadata() -> None:
    row = account_activation_call(
        model_id="model",
        prompt_family="control",
        phase="warm_decode",
        decode_step=2,
        module_name="linear",
        module_aliases=("linear",),
        input_width=8,
        output_width=16,
        vector_count=1,
        exact_zero_count=6,
    )
    assert row.input_scalar_count == 8
    assert row.nonzero_count == 2
    assert row.dense_operation_terms == 128
    assert row.sparse_operation_terms == 32
    assert row.zero_scan_terms == 8
    assert row.fully_accounted_operation_terms == 40
    assert row.sparse_operation_fraction == 0.25
    assert row.fully_accounted_operation_fraction == 0.3125
    assert row.dense_q4_weight_bytes == 64
    assert row.sparse_q4_weight_bytes == 16
    assert row.activation_metadata_bytes == 4
    assert row.query_byte_fraction == pytest.approx(20 / 64)


def test_invalid_accounting_is_rejected() -> None:
    with pytest.raises(ActivationSparsityError):
        account_activation_call(
            model_id="m",
            prompt_family="p",
            phase="prefill",
            decode_step=-1,
            module_name="x",
            module_aliases=("x",),
            input_width=0,
            output_width=1,
            vector_count=1,
            exact_zero_count=0,
        )


def test_positive_and_negative_zero_are_counted_identically() -> None:
    layer = torch.nn.Linear(4, 3, bias=False)
    recorder = ActivationSparsityRecorder.from_model(layer)
    recorder.attach()
    recorder.set_context(
        model_id="m", prompt_family="zero", phase="warm_decode", decode_step=2
    )
    layer(torch.tensor([[0.0, -0.0, 1.0, -2.0]]))
    recorder.detach()
    assert len(recorder.calls) == 1
    assert recorder.calls[0].exact_zero_count == 2


def test_all_zero_vector_has_only_scan_operations() -> None:
    row = account_activation_call(
        model_id="m",
        prompt_family="all_zero",
        phase="warm_decode",
        decode_step=3,
        module_name="linear",
        module_aliases=("linear",),
        input_width=32,
        output_width=64,
        vector_count=1,
        exact_zero_count=32,
    )
    assert row.sparse_operation_terms == 0
    assert row.fully_accounted_operation_fraction == pytest.approx(1 / 64)
    assert row.sparse_q4_weight_bytes == 0


def test_relu_control_creates_registered_exact_zeros() -> None:
    values = torch.tensor([-2.0, -0.5, 0.0, 3.0])
    result = torch.relu(values)
    assert int((result == 0).sum().item()) == 3


def test_gelu_random_control_does_not_create_false_zeros() -> None:
    values = torch.tensor([-2.0, -0.5, 0.5, 3.0])
    result = torch.nn.functional.gelu(values)
    assert int((result == 0).sum().item()) == 0


def test_zero_skipped_scalar_dot_matches_dense_reference_exactly() -> None:
    dense, sparse = exact_zero_skipped_dot(
        weights=[1.5, -2.0, 4.0, 0.25],
        values=[3.0, 0.0, -0.0, 8.0],
    )
    assert dense == sparse
    assert dense == 6.5


def test_registration_deduplicates_shared_module_and_retains_aliases() -> None:
    class Shared(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            linear = torch.nn.Linear(4, 5, bias=False)
            self.a = linear
            self.b = linear

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.a(value) + self.b(value)

    registrations = register_linear_projections(Shared())
    assert len(registrations) == 1
    assert registrations[0].aliases == ("a", "b")
    assert registrations[0].weight_shape == (5, 4)


def test_recorder_registration_calls_and_detach_are_deterministic() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 6),
        torch.nn.GELU(),
        torch.nn.Linear(6, 2),
    )
    recorder = ActivationSparsityRecorder.from_model(model)
    assert [item.canonical_name for item in recorder.registrations] == ["0", "2"]
    recorder.attach()
    recorder.set_context(
        model_id="m", prompt_family="p", phase="prefill", decode_step=-1
    )
    output_with_hooks = model(torch.ones((1, 4)))
    assert not recorder.missing_called_modules()
    assert len(recorder.calls) == 2
    recorder.detach()
    call_count = len(recorder.calls)
    output_without_hooks = model(torch.ones((1, 4)))
    assert len(recorder.calls) == call_count
    assert torch.equal(output_with_hooks, output_without_hooks)


def test_weighted_percentile_uses_dense_operation_weight() -> None:
    small = account_activation_call(
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=2,
        module_name="small",
        module_aliases=("small",),
        input_width=2,
        output_width=1,
        vector_count=1,
        exact_zero_count=2,
    )
    large = account_activation_call(
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=2,
        module_name="large",
        module_aliases=("large",),
        input_width=100,
        output_width=100,
        vector_count=1,
        exact_zero_count=0,
    )
    value = weighted_percentile(
        [small, large],
        field_name="fully_accounted_operation_fraction",
        probability=0.5,
    )
    assert math.isclose(value, large.fully_accounted_operation_fraction)
