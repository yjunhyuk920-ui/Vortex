from __future__ import annotations

from scripts import run_partitioned_signed_dual_cone as runner
from vortex_runtime.quantized_partition_metadata import (
    compile_quantized_partitioned_kernel,
    quantized_partition_metadata_budget,
)


def _compile(
    *,
    gate_weight,
    up_weight,
    down_weight,
    bits: int,
    block_size: int,
):
    return compile_quantized_partitioned_kernel(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        bits=bits,
        block_size=block_size,
        metadata_bits=8,
    )


def _budget(
    *,
    target,
    block_size: int,
    metadata_bits: int,
    metadata_limit_gib: float,
):
    return quantized_partition_metadata_budget(
        target=target,
        block_size=block_size,
        norm_bits=metadata_bits,
        scale_bits=16,
        metadata_limit_gib=metadata_limit_gib,
    )


runner.compile_partitioned_signed_dual_kernel = _compile
runner.partitioned_cone_metadata_budget = _budget


if __name__ == "__main__":
    runner.main()
