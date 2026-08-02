from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_mlp_neuron_dictionary_point as baseline
from vortex_runtime.mlp_functional_skeleton import (
    FunctionalSkeletonStats,
    compile_swiglu_functional_skeleton,
)
from vortex_runtime.mlp_gauge_dictionary import UpDownGaugeStats


def _output_path(arguments: list[str]) -> Path:
    try:
        index = arguments.index("--output")
    except ValueError as error:
        raise SystemExit("--output is required") from error
    if index + 1 >= len(arguments):
        raise SystemExit("--output requires a path")
    return Path(arguments[index + 1])


def _option(arguments: list[str], name: str, default: int) -> int:
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{name} requires a value")
    return int(arguments[index + 1])


def main() -> None:
    arguments = sys.argv[1:]
    output = _output_path(arguments)
    probe_count = _option(arguments, "--probe-count", 256)
    heldout_probe_count = _option(arguments, "--heldout-probe-count", 128)
    ridge_millionths = _option(arguments, "--ridge-millionths", 10)

    # Remove wrapper-only options before the baseline parser sees them.
    cleaned: list[str] = []
    skip = 0
    wrapper_options = {
        "--probe-count",
        "--heldout-probe-count",
        "--ridge-millionths",
    }
    for argument in arguments:
        if skip:
            skip -= 1
            continue
        if argument in wrapper_options:
            skip = 1
            continue
        cleaned.append(argument)
    sys.argv = [sys.argv[0], *cleaned]

    skeleton_rows: list[FunctionalSkeletonStats] = []
    gauge_rows: list[UpDownGaugeStats] = []

    def compile_wrapper(**kwargs):
        compiled, skeleton_stats, gauge_stats = (
            compile_swiglu_functional_skeleton(
                gate_proj=kwargs["gate_proj"],
                up_proj=kwargs["up_proj"],
                down_proj=kwargs["down_proj"],
                prototypes=kwargs["prototypes"],
                probe_count=probe_count,
                heldout_probe_count=heldout_probe_count,
                factor_bits=kwargs["factor_bits"],
                ridge=ridge_millionths / 1_000_000.0,
                seed=kwargs["seed"],
            )
        )
        skeleton_rows.append(skeleton_stats)
        gauge_rows.append(gauge_stats)
        # The baseline aggregator expects the dictionary fit interface. The
        # function-space errors are preserved separately below.
        from vortex_runtime.mlp_neuron_dictionary import MLPDictionaryFitStats

        proxy = MLPDictionaryFitStats(
            neurons=skeleton_stats.neurons,
            prototypes=skeleton_stats.prototypes_selected,
            projection_dim=probe_count,
            iterations=1,
            factor_bits=skeleton_stats.factor_bits,
            gate_up_relative_l2_error=(
                skeleton_stats.heldout_activation_relative_l2_error
            ),
            gate_relative_l2_error=(
                skeleton_stats.heldout_activation_relative_l2_error
            ),
            up_relative_l2_error=(
                skeleton_stats.heldout_activation_relative_l2_error
            ),
            minimum_cluster_size=0,
            maximum_cluster_size=0,
            mean_cluster_size=(
                skeleton_stats.neurons
                / skeleton_stats.prototypes_selected
            ),
            empty_clusters=0,
            factor_elements=skeleton_stats.factor_elements,
            factor_bytes=skeleton_stats.factor_bytes,
        )
        return compiled, proxy

    baseline.compile_swiglu_dictionary = compile_wrapper
    baseline.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    total_neurons = sum(row.neurons for row in skeleton_rows)
    payload["evidence_level"] = (
        "E2 executable SwiGLU functional skeleton frontier"
    )
    payload["functional_skeleton"] = {
        "layers": len(skeleton_rows),
        "probe_count": probe_count,
        "heldout_probe_count": heldout_probe_count,
        "ridge": ridge_millionths / 1_000_000.0,
        "neuron_weighted_probe_activation_error": sum(
            row.neurons * row.probe_activation_relative_l2_error
            for row in skeleton_rows
        ) / total_neurons,
        "neuron_weighted_heldout_activation_error": sum(
            row.neurons * row.heldout_activation_relative_l2_error
            for row in skeleton_rows
        ) / total_neurons,
        "neuron_weighted_probe_output_error": sum(
            row.neurons * row.probe_output_relative_l2_error
            for row in skeleton_rows
        ) / total_neurons,
        "neuron_weighted_heldout_output_error": sum(
            row.neurons * row.heldout_output_relative_l2_error
            for row in skeleton_rows
        ) / total_neurons,
        "maximum_heldout_output_error": max(
            row.heldout_output_relative_l2_error
            for row in skeleton_rows
        ),
        "maximum_selected_condition_number": max(
            row.selected_response_condition_number
            for row in skeleton_rows
        ),
        "maximum_coefficient_absolute_value": max(
            row.coefficient_maximum_absolute_value
            for row in skeleton_rows
        ),
        "factor_bytes": sum(row.factor_bytes for row in skeleton_rows),
        "per_layer": [row.to_dict() for row in skeleton_rows],
        "contract": (
            "Actual checkpoint neuron functions are selected using deterministic "
            "synthetic RMS-one probes. Linear interpolation coefficients are "
            "absorbed into the down projection. No prompts, labels, gradients "
            "or learned adapters are used."
        ),
    }
    payload["gauge_normalization"] = {
        "maximum_exact_function_relative_l2_error": max(
            row.exact_function_relative_l2_error for row in gauge_rows
        ),
        "zero_norm_neurons": sum(row.zero_norm_neurons for row in gauge_rows),
    }
    payload["decision"] = (
        "advance SwiGLU functional skeleton"
        if payload["qualifies"]
        else "reject tested SwiGLU functional skeleton point"
    )
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "functional_skeleton": payload["functional_skeleton"],
                "gauge_normalization": payload["gauge_normalization"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
