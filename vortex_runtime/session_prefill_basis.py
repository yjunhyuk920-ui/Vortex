from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


@dataclass(frozen=True)
class SessionBasisStats:
    input_vectors: int
    requested_rank: int
    numerical_rank: int
    compiled_rank: int
    input_features: int
    output_features: int
    input_reconstruction_relative_error: float
    output_reconstruction_relative_error: float
    capsule_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def compile_session_response_basis(
    module: DecisionResidualTileAtlasLinearModule,
    *,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    max_rank: int,
    rank_rtol: float = 1e-6,
) -> SessionBasisStats:
    """Compile ``U`` and ``WU`` using only exact prompt inputs and outputs.

    Let prompt activation rows form ``X`` and exact linear outputs without bias
    form ``Y = X W.T``. An orthonormal row-space basis ``U`` is obtained from
    the right singular vectors of ``X``. Coordinates ``C = X U`` then satisfy
    ``Y = C (W U).T`` on the captured prompt span, so ``WU`` can be recovered
    by a least-squares solve without reading the exact weight matrix again.

    The compiled capsule is exact on the retained prompt activation span up to
    floating-point solve error. No continuation token or continuation gradient
    is accepted by this function.
    """

    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
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

    _, singular_values, vh = torch.linalg.svd(
        inputs,
        full_matrices=False,
    )
    if singular_values.numel() == 0:
        raise RuntimeError("prompt activation SVD produced no singular values")
    threshold = float(singular_values[0].item()) * rank_rtol
    numerical_rank = int(torch.count_nonzero(singular_values > threshold).item())
    compiled_rank = min(
        int(max_rank),
        numerical_rank,
        module.exact.in_features,
        inputs.shape[0],
    )
    if compiled_rank <= 0:
        raise RuntimeError("prompt activation matrix has zero numerical rank")

    basis = vh[:compiled_rank].T.contiguous()
    coordinates = inputs @ basis
    solution = torch.linalg.lstsq(coordinates, outputs).solution
    image = solution.T.contiguous()

    input_reconstruction = coordinates @ basis.T
    output_reconstruction = coordinates @ image.T

    module.atlas.input_basis = basis.to(module.atlas.basis_dtype)
    module.atlas.output_image = image.to(module.atlas.basis_dtype)

    return SessionBasisStats(
        input_vectors=int(inputs.shape[0]),
        requested_rank=int(max_rank),
        numerical_rank=numerical_rank,
        compiled_rank=compiled_rank,
        input_features=module.exact.in_features,
        output_features=module.exact.out_features,
        input_reconstruction_relative_error=_relative_error(
            inputs,
            input_reconstruction,
        ),
        output_reconstruction_relative_error=_relative_error(
            outputs,
            output_reconstruction,
        ),
        capsule_bytes=module.atlas.capsule_bytes,
    )


def compile_session_response_bases(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    captured_inputs: Mapping[str, torch.Tensor],
    captured_outputs: Mapping[str, torch.Tensor],
    max_rank: int,
    rank_rtol: float = 1e-6,
) -> dict[str, SessionBasisStats]:
    if not replacements:
        raise ValueError("at least one replacement module is required")
    result: dict[str, SessionBasisStats] = {}
    for name, module in replacements.items():
        if name not in captured_inputs or name not in captured_outputs:
            raise RuntimeError(f"missing exact prompt capture for {name}")
        result[name] = compile_session_response_basis(
            module,
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            max_rank=max_rank,
            rank_rtol=rank_rtol,
        )
    return result
