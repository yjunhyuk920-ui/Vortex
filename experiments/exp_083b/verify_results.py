#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vortex_runtime.causal_residual_atlas_legal_execution import (
    bound_legal_projection_candidate,
    compile_stored_pair_certificate,
    propagate_last_down_to_final_hidden,
    vector_norm_upper,
)
from vortex_runtime.causal_residual_atlas_legal_gate import (
    array_frobenius_norm_upper,
    certify_top1_from_row_norms,
    matrix_row_norm_uppers,
    native_linear_per_row_rounding_bounds,
    select_max_residual_energy_page,
    summarize_legal_pair_outward_gate,
    verified_cholesky_spectral_upper_bound,
)
from vortex_runtime.causal_residual_atlas_oracle import (
    canonical_sha256,
    target_to_candidate_kls,
)


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
            )
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError("missing checksums.sha256")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise ValueError(f"duplicate checksum row: {relative}")
        expected[relative] = digest
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(expected) != actual_files:
        raise ValueError("checksum file-set mismatch")
    for relative, digest in expected.items():
        if sha256_file(output / relative) != digest:
            raise ValueError(f"checksum mismatch for {relative}")


def load_checkpoint_arrays(
    model_dir: Path,
    *,
    layer_index: int,
) -> tuple[Any, Any, Any]:
    from safetensors import safe_open

    weight_path = model_dir / "model.safetensors-00001-of-00001.safetensors"
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        down = handle.get_tensor(
            f"model.language_model.layers.{layer_index}.mlp.down_proj.weight"
        )
        embedding = handle.get_tensor("model.language_model.embed_tokens.weight")
        norm_weight = handle.get_tensor("model.language_model.norm.weight")
    return down, embedding, norm_weight


def static_lm_row_norms(weight: Any, *, chunk_rows: int = 4096) -> np.ndarray:
    result = np.empty(int(weight.shape[0]), dtype=np.float64)
    for start in range(0, int(weight.shape[0]), chunk_rows):
        stop = min(int(weight.shape[0]), start + chunk_rows)
        chunk = weight[start:stop].float().numpy().astype(np.float64)
        result[start:stop] = matrix_row_norm_uppers(chunk)
    return result


def assert_array_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(
        left, right
    ):
        raise ValueError(f"array mismatch: {name}")


def assert_dict_equal(name: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    if left != right:
        raise ValueError(f"dictionary mismatch: {name}")


def recompute_native_candidate(
    *,
    stored_basis: np.ndarray,
    stored_image: np.ndarray,
    current_input: np.ndarray,
    residual_branch: np.ndarray,
    selected_weight_page: np.ndarray,
    selected_page: int,
    page_columns: int,
    norm_weight: Any,
    embedding: Any,
) -> dict[str, np.ndarray]:
    """Replay tensor operations only; this is not a Transformer forward."""

    import torch
    import torch.nn.functional as functional

    basis = torch.from_numpy(stored_basis.astype(np.float32)).to(torch.bfloat16)
    image = torch.from_numpy(stored_image.astype(np.float32)).to(torch.bfloat16)
    current = torch.from_numpy(current_input.astype(np.float32)).to(torch.bfloat16)
    branch = torch.from_numpy(residual_branch.astype(np.float32)).to(torch.bfloat16)
    page = torch.from_numpy(selected_weight_page.astype(np.float32)).to(
        torch.bfloat16
    )
    with torch.inference_mode():
        coordinates = basis.T @ current
        basis_application = basis @ coordinates
        residual = current - basis_application
        start = selected_page * page_columns
        page_input = residual[start : start + page.shape[1]]
        image_application = image @ coordinates
        page_application = page @ page_input
        candidate_down = image_application + page_application
        candidate_pre = branch + candidate_down
        normalized = candidate_pre.float() * torch.rsqrt(
            candidate_pre.float().pow(2).mean() + 1e-6
        )
        candidate_hidden = (
            normalized * (1.0 + norm_weight.float())
        ).to(torch.bfloat16)
        candidate_logits = functional.linear(candidate_hidden, embedding)

    def array(value: Any) -> np.ndarray:
        return value.float().numpy().astype(np.float64, copy=False)

    return {
        "coordinates": array(coordinates),
        "basis_application": array(basis_application),
        "input_residual": array(residual),
        "stored_image_application": array(image_application),
        "selected_page_application": array(page_application),
        "candidate_down": array(candidate_down),
        "candidate_pre_norm": array(candidate_pre),
        "candidate_hidden": array(candidate_hidden),
        "candidate_logits": array(candidate_logits),
    }


def deterministic_core(
    *,
    registered_inputs: dict[str, Any],
    static_certificate: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "registered_inputs": registered_inputs,
        "static_certificate": static_certificate,
        "pair_rows": pair_rows,
        "certificate_rows": certificate_rows,
        "token_rows": token_rows,
        "control_rows": control_rows,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args()
    model_dir = Path(arguments.model_dir).resolve()
    output = Path(arguments.output_dir).resolve()
    verify_checksums(output)

    summary = load(output / "summary.json")
    config = json.loads(
        (output / "artifacts/contract.txt").read_text(encoding="utf-8")
    )
    pair_rows = load_rows(output / "raw/pair_rows.jsonl")
    certificate_rows = load_rows(output / "raw/certificate_rows.jsonl")
    token_rows = load_rows(output / "raw/token_rows.jsonl")
    control_rows = load(output / "raw/control_rows.json")
    stored_static = load(output / "raw/static_certificate.json")
    stored_gate = load(output / "processed/gate.json")
    layer_index = int(config["gate"]["layer_index"])

    from experiments.exp_076.run_experiment import validate_checkpoint_manifest

    exp076_config = load(ROOT / config["checkpoint"]["exp076_manifest"])
    checkpoint_rows = validate_checkpoint_manifest(model_dir, exp076_config)
    if checkpoint_rows != load(output / "raw/checkpoint_manifest.json"):
        raise ValueError("checkpoint manifest differs from stored evidence")
    down_tensor, embedding, norm_weight = load_checkpoint_arrays(
        model_dir,
        layer_index=layer_index,
    )
    down_weight = down_tensor.float().numpy().astype(np.float64)
    spectral = verified_cholesky_spectral_upper_bound(down_weight)
    lm_row_norms = static_lm_row_norms(embedding)
    gain = 1.0 + norm_weight.float().numpy().astype(np.float64)
    recomputed_static = {
        "spectral": spectral.to_dict(),
        "lm_head_row_norms_sha256": sha256_array(lm_row_norms),
        "lm_head_rows": int(lm_row_norms.size),
        "lm_head_minimum_row_norm_upper": float(np.min(lm_row_norms)),
        "lm_head_maximum_row_norm_upper": float(np.max(lm_row_norms)),
        "final_norm_gain_abs_max": float(np.max(np.abs(gain))),
        "final_norm_epsilon": 1e-6,
        "registered_spectral_compile_operations": int(
            config["registered_accounting"]["spectral_compile_operations"]
        ),
    }
    assert_dict_equal("static certificate", recomputed_static, stored_static)
    with np.load(output / "raw/static_arrays.npz", allow_pickle=False) as arrays:
        assert_array_equal(
            "static LM row norms",
            arrays["lm_head_row_norm_uppers"],
            lm_row_norms,
        )
        assert_array_equal("static final norm gain", arrays["final_norm_gain"], gain)

    certificate_by_id = {
        str(row["prompt_id"]): row for row in certificate_rows
    }
    token_by_id = {str(row["prompt_id"]): row for row in token_rows}
    if len(certificate_by_id) != len(certificate_rows):
        raise ValueError("duplicate certificate prompt IDs")
    verified_prompt_ids: list[str] = []
    for pair_row in pair_rows:
        prompt_id = str(pair_row["prompt_id"])
        array_path = output / str(pair_row["array_file"])
        if not array_path.is_file():
            # A certified rank failure may stop before a candidate array exists.
            if int(pair_row["rank"]) >= int(config["gate"]["rank"]):
                raise ValueError(f"missing candidate arrays for {prompt_id}")
            continue
        with np.load(array_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        pair = compile_stored_pair_certificate(
            arrays["prefix_inputs"],
            arrays["prefix_images"],
            maximum_rank=int(config["gate"]["rank"]),
            operator_norm_upper=spectral.operator_norm_upper,
            frobenius_norm_upper=spectral.frobenius_norm_upper,
        )
        assert_array_equal("stored basis", pair.stored_basis, arrays["stored_basis"])
        assert_array_equal("stored image", pair.stored_image, arrays["stored_image"])
        assert_array_equal(
            "coefficient map",
            pair.coefficient_map,
            arrays["coefficient_map"],
        )
        if sha256_array(arrays["prefix_inputs"]) != pair_row["prefix_input_sha256"]:
            raise ValueError(f"prefix input SHA mismatch for {prompt_id}")
        if sha256_array(arrays["prefix_images"]) != pair_row["prefix_image_sha256"]:
            raise ValueError(f"prefix image SHA mismatch for {prompt_id}")
        scalar_pair_fields = {
            "rank": pair.rank,
            "accepted_prefix_indices": list(pair.accepted_prefix_indices),
            "accepted_residual_norm_lowers": list(
                pair.accepted_residual_norm_lowers
            ),
            "accepted_threshold_uppers": list(pair.accepted_threshold_uppers),
            "prefix_image_error_frobenius": pair.prefix_image_error_frobenius,
            "coefficient_operator_upper": pair.coefficient_operator_upper,
            "stored_basis_relation_error_operator": (
                pair.stored_basis_relation_error_operator
            ),
            "stored_image_relation_error_operator": (
                pair.stored_image_relation_error_operator
            ),
            "pair_image_defect_operator": pair.pair_image_defect_operator,
        }
        for key, value in scalar_pair_fields.items():
            if pair_row[key] != value:
                raise ValueError(f"pair metadata mismatch for {prompt_id}: {key}")

        certificate_row = certificate_by_id[prompt_id]
        selected = select_max_residual_energy_page(
            arrays["input_residual"],
            page_columns=int(config["gate"]["page_columns"]),
        )
        if selected != int(certificate_row["selected_page"]):
            raise ValueError(f"target-free page mismatch for {prompt_id}")
        start = selected * int(config["gate"]["page_columns"])
        expected_page = down_weight[
            :, start : start + int(config["gate"]["page_columns"])
        ]
        assert_array_equal(
            "selected checkpoint page",
            arrays["selected_weight_page"],
            expected_page,
        )
        replay = recompute_native_candidate(
            stored_basis=pair.stored_basis,
            stored_image=pair.stored_image,
            current_input=arrays["current_input"],
            residual_branch=arrays["residual_branch"],
            selected_weight_page=expected_page,
            selected_page=selected,
            page_columns=int(config["gate"]["page_columns"]),
            norm_weight=norm_weight,
            embedding=embedding,
        )
        for key, value in replay.items():
            assert_array_equal(f"candidate replay {key}", value, arrays[key])

        projection = bound_legal_projection_candidate(
            stored_basis=pair.stored_basis,
            stored_image=pair.stored_image,
            pair_image_defect=pair.pair_image_defect_operator,
            operator_norm_upper=spectral.operator_norm_upper,
            frobenius_norm_upper=spectral.frobenius_norm_upper,
            current_input=arrays["current_input"],
            coordinates=arrays["coordinates"],
            basis_application=arrays["basis_application"],
            residual=arrays["input_residual"],
            selected_page=selected,
            page_columns=int(config["gate"]["page_columns"]),
            selected_weight_page=expected_page,
            stored_image_application=arrays["stored_image_application"],
            selected_page_application=arrays["selected_page_application"],
        )
        final = propagate_last_down_to_final_hidden(
            residual_branch=arrays["residual_branch"],
            candidate_down=arrays["candidate_down"],
            candidate_pre_norm=arrays["candidate_pre_norm"],
            projection_radius=projection.projection_output_radius,
            gain_abs_max=float(np.max(np.abs(gain))),
            epsilon=1e-6,
        )
        assert_dict_equal(
            f"projection bounds {prompt_id}",
            projection.to_dict(),
            certificate_row["projection_bounds"],
        )
        assert_dict_equal(
            f"final bounds {prompt_id}",
            final.to_dict(),
            certificate_row["final_bounds"],
        )
        candidate_hidden_norm = vector_norm_upper(arrays["candidate_hidden"])
        target_hidden_norm = math.nextafter(
            candidate_hidden_norm + final.final_hidden_radius,
            math.inf,
        )
        candidate_rounding = native_linear_per_row_rounding_bounds(
            lm_row_norms,
            input_norm_upper=candidate_hidden_norm,
            columns=int(config["gate"]["hidden_size"]),
        )
        target_rounding = native_linear_per_row_rounding_bounds(
            lm_row_norms,
            input_norm_upper=target_hidden_norm,
            columns=int(config["gate"]["hidden_size"]),
        )
        assert_array_equal(
            "candidate logit rounding",
            arrays["candidate_logit_rounding"],
            candidate_rounding,
        )
        assert_array_equal(
            "target logit rounding",
            arrays["target_logit_rounding"],
            target_rounding,
        )
        two_path = np.nextafter(candidate_rounding + target_rounding, np.inf)
        certificate = certify_top1_from_row_norms(
            arrays["candidate_logits"],
            hidden_radius=final.final_hidden_radius,
            row_norm_uppers=lm_row_norms,
            per_logit_rounding_uppers=two_path,
        )
        if bool(certificate.certified) != bool(
            certificate_row["certificate_resolved"]
        ) or int(certificate.winner) != int(certificate_row["certified_winner"]):
            raise ValueError(f"top1 certificate mismatch for {prompt_id}")
        if certificate.minimum_margin_lower != certificate_row[
            "minimum_margin_lower"
        ]:
            raise ValueError(f"certificate margin mismatch for {prompt_id}")
        if not bool(certificate_row["verdict_frozen_before_dense_completion"]):
            raise ValueError(f"missing verdict barrier for {prompt_id}")
        if int(certificate_row["verdict_frozen_monotonic_ns"]) >= int(
            certificate_row["dense_completion_finished_monotonic_ns"]
        ):
            raise ValueError(f"verdict/dense ordering mismatch for {prompt_id}")
        for key in (
            "native_down",
            "native_pre_norm",
            "native_hidden",
            "native_logits",
        ):
            assert_array_equal(
                f"dense replay {key}",
                arrays[key],
                arrays[f"replay_{key}"],
            )
        actual_projection = float(
            np.linalg.norm(arrays["candidate_down"] - arrays["native_down"])
        )
        actual_hidden = float(
            np.linalg.norm(arrays["candidate_hidden"] - arrays["native_hidden"])
        )
        if actual_projection > projection.projection_output_radius:
            raise ValueError(f"projection radius false for {prompt_id}")
        if actual_hidden > final.final_hidden_radius:
            raise ValueError(f"hidden radius false for {prompt_id}")
        logit_errors = np.abs(
            arrays["candidate_logits"] - arrays["native_logits"]
        )
        if not np.all(logit_errors <= arrays["logit_error_uppers"]):
            raise ValueError(f"logit radius false for {prompt_id}")
        divergence = float(
            target_to_candidate_kls(
                arrays["native_logits"],
                arrays["candidate_logits"][None, :],
            )[0]
        )
        token = token_by_id[prompt_id]
        if divergence != float(token["target_to_candidate_kl"]):
            raise ValueError(f"KL mismatch for {prompt_id}")
        native_winner = int(np.argmax(arrays["native_logits"]))
        false_accept = bool(
            certificate.certified and certificate.winner != native_winner
        )
        if false_accept != bool(token["false_accept"]):
            raise ValueError(f"false-accept mismatch for {prompt_id}")
        verified_prompt_ids.append(prompt_id)

    observed_control_failures = [
        f"{row['prompt_id']}:{row['control']}"
        for row in control_rows
        if not bool(row.get("pass"))
    ]
    stored_control_failures = list(stored_gate["control_failures"])
    # false_accept is represented in the token row rather than as a control row.
    if sorted(observed_control_failures) != sorted(
        value for value in stored_control_failures if not value.endswith(":false_accept")
    ):
        raise ValueError("control failure reconstruction mismatch")
    recomputed_gate = summarize_legal_pair_outward_gate(
        token_rows,
        expected_token_states=int(config["gate"]["expected_token_states"]),
        expected_families=tuple(config["registered_inputs"]["required_families"]),
        prompts_per_family=int(config["registered_inputs"]["prompts_per_family"]),
        maximum_mean_kl=float(config["gate"]["maximum_mean_kl"]),
        maximum_p95_kl=float(config["gate"]["maximum_p95_kl"]),
        control_failures=stored_control_failures,
        leakage_failures=tuple(stored_gate["leakage_failures"]),
        malformed_state_count=int(stored_gate["malformed_state_count"]),
        execution_complete=bool(stored_gate["execution_complete"]),
    )
    assert_dict_equal("Gate aggregation", recomputed_gate, stored_gate)
    if summary["authoritative_decision"] != stored_gate["decision"]:
        raise ValueError("summary decision differs from Gate")
    core = deterministic_core(
        registered_inputs=summary["REGISTERED_EXTERNAL_INPUTS"],
        static_certificate=stored_static,
        pair_rows=pair_rows,
        certificate_rows=certificate_rows,
        token_rows=token_rows,
        control_rows=control_rows,
        gate=stored_gate,
    )
    stored_core = load(output / "processed/deterministic_core.json")
    assert_dict_equal("deterministic core", core, stored_core)
    core_hash = canonical_sha256(core)
    if core_hash != summary["MEASURED"]["deterministic_core_sha256"]:
        raise ValueError("deterministic core SHA mismatch")

    report = {
        "experiment": "EXP-083B",
        "verification": "PASS",
        "authoritative_decision": stored_gate["decision"],
        "verified_prompt_ids": verified_prompt_ids,
        "deterministic_core_sha256": core_hash,
        "model_forward_calls": 0,
        "checks": [
            "bundle file checksums",
            "pinned checkpoint manifest",
            "outward Gram verified-Cholesky spectral proof",
            "full tied-LM-head outward row norms",
            "pair-only MGS and BF16 stored relations",
            "target-free maximum residual-energy page",
            "candidate tensor replay without Transformer forward",
            "projection, residual-add, RMSNorm, and LM-head radii",
            "strict top1 certificate and verdict-before-dense barrier",
            "dense replay, KL, false-accept, Gate, and deterministic core",
        ],
    }
    if arguments.write_report:
        dump(output / "artifacts/independent_verification.json", report)
        write_checksums(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
