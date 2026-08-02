from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


@dataclass(frozen=True)
class HybridBasisStats:
    input_vectors: int
    global_rank: int
    requested_total_rank: int
    available_session_rank: int
    residual_numerical_rank: int
    added_session_rank: int
    final_rank: int
    global_input_reconstruction_relative_error: float
    final_input_reconstruction_relative_error: float
    global_output_reconstruction_relative_error: float
    final_output_reconstruction_relative_error: float
    capsule_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def augment_response_basis_from_prompt_io(
    module: DecisionResidualTileAtlasLinearModule,
    *,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    total_rank: int,
    rank_rtol: float = 1e-6,
) -> HybridBasisStats:
    """Add prompt-residual directions to an existing response capsule.

    Existing columns ``U_g`` and ``W U_g`` form the global prior. Exact prompt
    inputs ``X`` and bias-free outputs ``Y`` are decomposed into the part already
    represented by the global capsule and a residual part. Right singular
    vectors of the input residual provide session directions ``Q`` orthogonal
    to ``U_g``. Their exact response images are recovered by solving

    ``Y - (X U_g)(W U_g).T = (X Q)(W Q).T``.

    The function uses no continuation token, continuation gradient, or extra
    exact-weight read. It reuses exact prompt-prefill module inputs and outputs.
    """

    if total_rank <= 0:
        raise ValueError("total_rank must be positive")
    if rank_rtol < 0:
        raise ValueError("rank_rtol must be non-negative")
    if input_tensor.shape[:-1] != output_tensor.shape[:-1]:
        raise ValueError("input and output leading dimensions must match")
    if input_tensor.shape[-1] != module.exact.in_features:
        raise ValueError("captured input feature dimension mismatch")
    if output_tensor.shape[-1] != module.exact.out_features:
        raise ValueError("captured output feature dimension mismatch")

    inputs = input_tensor.detach().to("cpu", torch.float32).reshape(
        -1, module.exact.in_features
    )
    outputs = output_tensor.detach().to("cpu", torch.float32).reshape(
        -1, module.exact.out_features
    )
    if inputs.shape[0] == 0:
        raise ValueError("at least one prompt activation vector is required")
    if module.exact.bias is not None:
        outputs = outputs - module.exact.bias.detach().to(
            "cpu", torch.float32
        ).reshape(1, -1)

    global_basis = module.atlas.input_basis.detach().to(
        "cpu", torch.float32
    ).contiguous()
    global_image = module.atlas.output_image.detach().to(
        "cpu", torch.float32
    ).contiguous()
    global_rank = int(global_basis.shape[1])
    if global_image.shape[1] != global_rank:
        raise RuntimeError("global basis and response image ranks differ")
    if total_rank < global_rank:
        raise ValueError("total_rank cannot be smaller than the global rank")

    if global_rank:
        global_coordinates = inputs @ global_basis
        global_input = global_coordinates @ global_basis.T
        global_output = global_coordinates @ global_image.T
    else:
        global_input = torch.zeros_like(inputs)
        global_output = torch.zeros_like(outputs)

    input_residual = inputs - global_input
    output_residual = outputs - global_output
    available = total_rank - global_rank

    _, singular_values, vh = torch.linalg.svd(
        input_residual,
        full_matrices=False,
    )
    if singular_values.numel():
        threshold = float(singular_values[0].item()) * rank_rtol
        numerical_rank = int(
            torch.count_nonzero(singular_values > threshold).item()
        )
    else:
        numerical_rank = 0
    added_rank = min(available, numerical_rank)

    if added_rank:
        session_basis = vh[:added_rank].T.contiguous()
        if global_rank:
            session_basis = session_basis - global_basis @ (
                global_basis.T @ session_basis
            )
        session_basis, _ = torch.linalg.qr(
            session_basis,
            mode="reduced",
        )
        session_coordinates = inputs @ session_basis
        session_image = torch.linalg.lstsq(
            session_coordinates,
            output_residual,
        ).solution.T.contiguous()
        final_basis = torch.cat((global_basis, session_basis), dim=1)
        final_image = torch.cat((global_image, session_image), dim=1)
    else:
        final_basis = global_basis
        final_image = global_image

    final_coordinates = inputs @ final_basis
    final_input = final_coordinates @ final_basis.T
    final_output = final_coordinates @ final_image.T

    module.atlas.input_basis = final_basis.to(module.atlas.basis_dtype)
    module.atlas.output_image = final_image.to(module.atlas.basis_dtype)
    module.atlas.max_rank = max(module.atlas.max_rank, total_rank)

    return HybridBasisStats(
        input_vectors=int(inputs.shape[0]),
        global_rank=global_rank,
        requested_total_rank=total_rank,
        available_session_rank=available,
        residual_numerical_rank=numerical_rank,
        added_session_rank=added_rank,
        final_rank=int(final_basis.shape[1]),
        global_input_reconstruction_relative_error=_relative_error(
            inputs,
            global_input,
        ),
        final_input_reconstruction_relative_error=_relative_error(
            inputs,
            final_input,
        ),
        global_output_reconstruction_relative_error=_relative_error(
            outputs,
            global_output,
        ),
        final_output_reconstruction_relative_error=_relative_error(
            outputs,
            final_output,
        ),
        capsule_bytes=module.atlas.capsule_bytes,
    )


def augment_response_bases_from_prompt_io(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    captured_inputs: Mapping[str, torch.Tensor],
    captured_outputs: Mapping[str, torch.Tensor],
    total_rank: int,
    rank_rtol: float = 1e-6,
) -> dict[str, HybridBasisStats]:
    if not replacements:
        raise ValueError("at least one response capsule is required")
    result: dict[str, HybridBasisStats] = {}
    for name, module in replacements.items():
        if name not in captured_inputs or name not in captured_outputs:
            raise RuntimeError(f"missing exact prompt capture for {name}")
        result[name] = augment_response_basis_from_prompt_io(
            module,
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            total_rank=total_rank,
            rank_rtol=rank_rtol,
        )
    return result
