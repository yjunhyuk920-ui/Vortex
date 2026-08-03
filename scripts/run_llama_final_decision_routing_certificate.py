from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.llama_final_decision_routing import (
    MicroRoutingConfig,
    enumerate_micro_family,
    target_routing_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Experiment 042 end-to-end Llama final-token "
            "routing information certificate."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/llama_final_decision_routing_bound.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MicroRoutingConfig()
    enumeration = enumerate_micro_family(config)
    projection = target_routing_projection()

    certificate_passes = bool(
        enumeration.passes
        and projection.exceeds_resident_limit
        and projection.vocabulary_pass
        and projection.hidden_layout_pass
        and projection.intermediate_pass
        and projection.loader_capacity_pass
    )

    payload = {
        "experiment": "042_llama_final_decision_routing_bound",
        "evidence_level": (
            "E1 formal and executable end-to-end final-token "
            "information certificate"
        ),
        "micro_model": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "kv_dimension": config.kv_dimension,
            "total_layers": config.total_layers,
            "loader_layers": config.loader_layers,
            "variable_layers": config.variable_layers,
            "groups_per_layer": config.groups_per_layer,
            "neurons_per_group": config.neurons_per_group,
            "payload_coordinates": config.payload_coordinates,
            "code_levels": config.code_levels,
            "legal_prompt_tokens": 4,
            "checkpoint_coefficients": (
                config.checkpoint_coefficients
            ),
            "checkpoint_information_bits": (
                config.checkpoint_information_bits
            ),
        },
        "exhaustive_family": enumeration.to_dict(),
        "target_projection": projection.to_dict(),
        "scope_separation": {
            "actual_llama_style_rmsnorm_gqa_residual_swiglu_lm_head": True,
            "final_vocabulary_winner_decodes_every_q4_code": (
                enumeration.exact_code_recovery
            ),
            "two_layer_additivity_exhaustively_proven": (
                enumeration.observed_functions
                == enumeration.expected_functions
                == 256
            ),
            "metadata_aware_final_token_bound_proven_for_constructed_family": (
                certificate_passes
            ),
            "arbitrary_checkpoint_universality_contradicted_at_8gib": (
                certificate_passes
            ),
            "released_405b_checkpoint_maximum_complexity_proven": False,
            "real_405b_execution_performed": False,
            "measured_gpu_wall_clock_proven": False,
        },
        "conclusion": {
            "certificate_passes": certificate_passes,
            "micro_functions": enumeration.observed_functions,
            "micro_minimum_winner_margin": (
                enumeration.minimum_winner_margin
            ),
            "projected_metadata_gib": projection.metadata_gib,
            "resident_limit_gib": projection.resident_limit_gib,
            "excess_gib": (
                projection.metadata_gib
                - projection.resident_limit_gib
            ),
            "fixed_target_status": (
                "contradicted for arbitrary exact-decision Q4 "
                "Llama-style checkpoints under an 8 GiB complete "
                "checkpoint-information allowance"
                if certificate_passes
                else "certificate did not close the target"
            ),
        },
        "next_obligation": (
            "separate the now-contradicted arbitrary worst-case target "
            "from a narrower empirical runtime target for released "
            "checkpoints, then charge quality relaxation, host traffic, "
            "and measured wall clock explicitly"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
