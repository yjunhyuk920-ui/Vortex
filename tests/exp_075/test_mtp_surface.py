from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from vortex_runtime.mtp_surface import (
    MtpSurfaceAuditError,
    audit_checkpoint_surface,
    audit_runtime_surface,
    surface_decision,
    validate_metadata_sources,
)


EXPECTED_KEYS = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def checkpoint_fixture(keys: list[str] = EXPECTED_KEYS):
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }
    index = {
        "metadata": {"total_size": 1_746_882_752},
        "weight_map": {key: "model.safetensors" for key in keys},
    }
    card = "MTP: trained with multi-steps\nNEXTN\n"
    return config, index, card


def audit(keys: list[str] = EXPECTED_KEYS):
    config, index, card = checkpoint_fixture(keys)
    return audit_checkpoint_surface(
        model_config=config,
        weight_index=index,
        model_card=card,
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_model_type="qwen3_5_text",
        expected_mtp_layers=1,
        expected_mtp_keys=EXPECTED_KEYS,
        required_card_markers=["MTP: trained with multi-steps", "NEXTN"],
    )


def test_complete_checkpoint_surface_passes() -> None:
    result = audit()
    assert result["passed"] is True
    assert result["mtp_tensor_count"] == 15
    assert result["declared_weight_bytes"] == 1_746_882_752


def test_missing_mtp_tensor_rejects_surface() -> None:
    result = audit(EXPECTED_KEYS[:-1])
    assert result["passed"] is False
    assert result["missing_mtp_tensor_keys"] == [EXPECTED_KEYS[-1]]


def test_unexpected_mtp_tensor_rejects_exact_registered_set() -> None:
    result = audit(EXPECTED_KEYS + ["mtp.unregistered.weight"])
    assert result["passed"] is False
    assert result["unexpected_mtp_tensor_keys"] == ["mtp.unregistered.weight"]


def test_runtime_surface_requires_every_registered_marker() -> None:
    passed = audit_runtime_surface(
        source_text={"spec": "alpha beta", "loader": "gamma delta"},
        required_markers={"spec": ["alpha", "beta"], "loader": ["gamma"]},
    )
    failed = audit_runtime_surface(
        source_text={"spec": "alpha", "loader": "gamma"},
        required_markers={"spec": ["alpha", "beta"], "loader": ["gamma"]},
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["files"][0]["missing_markers"] == ["beta"]


def test_missing_runtime_file_fails_closed() -> None:
    with pytest.raises(MtpSurfaceAuditError):
        audit_runtime_surface(source_text={}, required_markers={"missing": ["x"]})


def test_weight_payload_urls_are_prohibited() -> None:
    with pytest.raises(MtpSurfaceAuditError):
        validate_metadata_sources(
            [
                {
                    "name": "weights",
                    "output_name": "weights.safetensors",
                    "url": "https://example.test/model.safetensors",
                }
            ]
        )


def test_decision_ladder_is_fail_closed() -> None:
    assert surface_decision(
        checkpoint_passed=True, runtime_passed=True, controls_passed=True
    ).startswith("PROMOTE_TO_PINNED")
    assert surface_decision(
        checkpoint_passed=False, runtime_passed=True, controls_passed=True
    ) == "REJECT_NATIVE_MTP_SURFACE_UNAVAILABLE"
    assert surface_decision(
        checkpoint_passed=True, runtime_passed=False, controls_passed=True
    ) == "REVISE_RUNTIME_MTP_SURFACE_NOT_EXPOSED"
    assert surface_decision(
        checkpoint_passed=True, runtime_passed=True, controls_passed=False
    ) == "INVALID_METADATA_AUDIT_CONTROL_FAILURE"


def test_experiment_replays_from_offline_sources(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "experiments/exp_075/config.json").read_text())
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    model_config, index, _ = checkpoint_fixture()
    runtime_blobs = {
        name: "\n".join(markers) + "\n"
        for name, markers in config["runtime"]["required_markers"].items()
    }
    payloads = {
        "model_config": json.dumps(model_config),
        "weight_index": json.dumps(index),
        "model_card": "\n".join(config["checkpoint"]["required_model_card_markers"]),
        **runtime_blobs,
    }
    for source in config["sources"]:
        (source_dir / source["output_name"]).write_text(
            payloads[source["name"]], encoding="utf-8"
        )
    output = tmp_path / "exp_075_test_bundle"
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/exp_075/run_experiment.py"),
            "--config",
            str(root / "experiments/exp_075/config.json"),
            "--output-dir",
            str(output),
            "--source-dir",
            str(source_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["authoritative_decision"].startswith("PROMOTE_TO_PINNED")
    assert summary["MEASURED"]["mtp_tensor_count"] == 15
    assert summary["claim_boundary"]["weight_payload_download"] == "NOT_PERFORMED"
    assert (output / "checksums.sha256").is_file()
