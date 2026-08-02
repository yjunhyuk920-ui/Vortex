from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_mlp_neuron_dictionary_point as baseline
from vortex_runtime.mlp_gauge_dictionary import (
    UpDownGaugeStats,
    compile_gauge_normalized_swiglu_dictionary,
)


def _output_path(arguments: list[str]) -> Path:
    try:
        index = arguments.index("--output")
    except ValueError as error:
        raise SystemExit("--output is required for gauge result annotation") from error
    if index + 1 >= len(arguments):
        raise SystemExit("--output requires a path")
    return Path(arguments[index + 1])


def main() -> None:
    gauge_rows: list[UpDownGaugeStats] = []

    def compile_wrapper(**kwargs):
        compiled, fit_stats, gauge_stats = (
            compile_gauge_normalized_swiglu_dictionary(**kwargs)
        )
        gauge_rows.append(gauge_stats)
        return compiled, fit_stats

    baseline.compile_swiglu_dictionary = compile_wrapper
    output = _output_path(sys.argv[1:])
    baseline.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["evidence_level"] = (
        "E2 executable exact-gauge-normalized MLP neuron dictionary"
    )
    payload["gauge_normalization"] = {
        "layers": len(gauge_rows),
        "maximum_exact_function_relative_l2_error": max(
            row.exact_function_relative_l2_error for row in gauge_rows
        ),
        "mean_up_norm": sum(row.mean_up_norm for row in gauge_rows)
        / len(gauge_rows),
        "minimum_up_norm": min(row.minimum_up_norm for row in gauge_rows),
        "maximum_up_norm": max(row.maximum_up_norm for row in gauge_rows),
        "zero_norm_neurons": sum(row.zero_norm_neurons for row in gauge_rows),
        "contract": (
            "Each up row is divided by its positive L2 norm and the matching "
            "down column is multiplied by the same norm before clustering. "
            "This gauge transform is exactly function preserving."
        ),
    }
    payload["decision"] = (
        "advance gauge-normalized neuron dictionary"
        if payload["qualifies"]
        else "reject tested gauge-normalized neuron dictionary point"
    )
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload["gauge_normalization"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
