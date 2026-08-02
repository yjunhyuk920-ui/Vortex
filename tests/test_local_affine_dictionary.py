import torch
from torch import nn

from vortex_runtime.local_affine_dictionary import (
    LocalAffineDictionaryLinearModule,
    build_local_affine_dictionary,
    quantize_local_affine_dictionary,
)


def clustered_inputs() -> torch.Tensor:
    left_center = torch.tensor([-5.0, 0.0, 0.0, 0.0])
    right_center = torch.tensor([5.0, 0.0, 0.0, 0.0])
    offsets = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    return torch.cat((left_center + offsets, right_center + offsets), dim=0)


def test_local_affine_dictionary_reconstructs_cluster_lines() -> None:
    torch.manual_seed(503)
    linear = nn.Linear(4, 3, bias=True)
    inputs = clustered_inputs()
    outputs = linear(inputs)

    dictionary, stats = build_local_affine_dictionary(
        input_tensor=inputs,
        output_tensor=outputs,
        clusters=2,
        local_rank=1,
    )
    estimate = dictionary.apply(inputs)

    torch.testing.assert_close(estimate, outputs, atol=2e-5, rtol=2e-5)
    assert stats.stored_response_columns == 4
    assert stats.active_rank_maximum == 2
    assert sorted(stats.cluster_counts) == [3, 3]
    assert stats.training_output_reconstruction_relative_error < 1e-5


def test_wrapper_switches_between_exact_and_dictionary() -> None:
    torch.manual_seed(509)
    wrapper = LocalAffineDictionaryLinearModule(nn.Linear(4, 3, bias=True))
    inputs = clustered_inputs()
    outputs = wrapper.exact(inputs)
    dictionary, _ = build_local_affine_dictionary(
        input_tensor=inputs,
        output_tensor=outputs,
        clusters=2,
        local_rank=1,
    )
    wrapper.configure_dictionary(dictionary)
    wrapper.set_mode("dictionary")

    torch.testing.assert_close(wrapper(inputs), outputs, atol=2e-5, rtol=2e-5)
    wrapper.set_mode("exact")
    torch.testing.assert_close(wrapper(inputs), outputs)


def test_quantized_dictionary_preserves_shape_and_reports_payload() -> None:
    torch.manual_seed(521)
    linear = nn.Linear(4, 3, bias=False)
    inputs = clustered_inputs()
    dictionary, _ = build_local_affine_dictionary(
        input_tensor=inputs,
        output_tensor=linear(inputs),
        clusters=2,
        local_rank=1,
    )

    stats = quantize_local_affine_dictionary(dictionary, bits=8)
    estimate = dictionary.apply(inputs)

    assert estimate.shape == (6, 3)
    assert stats.logical_payload_bytes > 0
    assert stats.scale_bytes > 0
    assert stats.logical_total_bytes == (
        stats.logical_payload_bytes + stats.scale_bytes
    )
    assert stats.maximum_relative_l2_error < 0.02
