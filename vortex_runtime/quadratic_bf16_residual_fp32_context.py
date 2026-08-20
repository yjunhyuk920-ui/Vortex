from __future__ import annotations

import torch

from vortex_runtime import quadratic_bf16_residual_generator as base


def select_context_basis_fp32(
    x: torch.Tensor,
    context_bits: int,
    pool_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Select predicates using the exact FP32 representation deployed at query time.

    The original EXP-087A compiler ranked predicate features in FP64 and only
    afterwards stored the mean and basis as FP32. A query recomputed signs with
    those FP32 tensors. Near-zero projections could therefore change sign, so
    the compiled labels and the executable program did not necessarily use the
    same finite-word input. This function keeps the frozen 4-byte sidecar ABI
    and candidate pool, but performs selection and fitting with the exact FP32
    tensors that the runtime stores and evaluates.
    """

    x = base._require_bf16_2d(x, name="context build input")
    d = int(context_bits)
    if d < 1 or int(pool_size) < d:
        raise base.QuadraticResidualError("invalid context basis dimensions")

    mean64 = x.double().mean(dim=0)
    pool64 = base._candidate_direction_pool(x, mean64, int(pool_size), int(seed))

    # These are the checkpoint-static tensors charged by the frozen 4-byte ABI.
    mean = mean64.to(torch.float32).contiguous()
    pool = pool64.to(torch.float32).contiguous()
    pool_bits = ((x.float() - mean) @ pool.T >= 0).to(torch.uint8)

    selected: list[int] = []
    best_rank = 0
    remaining = list(range(int(pool.shape[0])))
    for _ in range(d):
        choice: int | None = None
        choice_rank = -1
        for candidate in remaining:
            columns = selected + [candidate]
            rank = base.gf2_rank(base.quadratic_features(pool_bits[:, columns]))
            if rank > choice_rank or (
                rank == choice_rank and (choice is None or candidate < choice)
            ):
                choice = candidate
                choice_rank = rank
        if choice is None:
            raise base.QuadraticResidualError("context basis selection failed")
        selected.append(choice)
        remaining.remove(choice)
        best_rank = int(choice_rank)

    basis = pool[selected].T.contiguous()
    bits = pool_bits[:, selected].contiguous()
    replay_bits = ((x.float() - mean) @ basis >= 0).to(torch.uint8)
    if not torch.equal(bits, replay_bits):
        raise base.QuadraticResidualError(
            "compiled and deployed FP32 predicate semantics diverged"
        )
    replay_rank = base.gf2_rank(base.quadratic_features(replay_bits))
    if replay_rank != best_rank:
        raise base.QuadraticResidualError("deployed feature-rank replay mismatch")
    return mean, basis, bits, best_rank


def install_fp32_context_semantics() -> None:
    """Patch only the predicate compiler entry point used by fit_program()."""

    base.select_context_basis = select_context_basis_fp32
