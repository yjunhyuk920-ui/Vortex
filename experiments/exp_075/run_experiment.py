#!/usr/bin/env python3
"""Run EXP-075 public metadata-only native-MTP surface audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vortex_runtime.mtp_surface import (
    audit_checkpoint_surface,
    audit_runtime_surface,
    canonical_sha256,
    require_finite_metrics,
    sha256_bytes,
    surface_decision,
    validate_metadata_sources,
)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def fetch_source(source: dict[str, Any], source_dir: Path | None) -> tuple[bytes, dict[str, Any]]:
    if source_dir is not None:
        path = source_dir / str(source["output_name"])
        payload = path.read_bytes()
        final_url = path.resolve().as_uri()
        status = "OFFLINE_FIXTURE"
        content_type = "application/octet-stream"
    else:
        request = urllib.request.Request(
            str(source["url"]), headers={"User-Agent": "VORTEX-EXP-075/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            limit = int(source["max_bytes"])
            payload = response.read(limit + 1)
            final_url = response.geturl()
            status = int(response.status)
            content_type = str(response.headers.get("Content-Type", ""))
    if len(payload) > int(source["max_bytes"]):
        raise ValueError(f"source exceeds byte limit: {source['name']}")
    payload.decode("utf-8")
    return payload, {
        "name": source["name"],
        "requested_url": source["url"],
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "output_name": source["output_name"],
    }


def write_checksums(output: Path) -> None:
    rows: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_075/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_075_candidate"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read registered filenames from an offline fixture directory.",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    validate_metadata_sources(config["sources"])

    output = arguments.output_dir.resolve()
    if output == ROOT.resolve() or not (
        output.name == "exp_075" or output.name.startswith("exp_075_")
    ):
        raise ValueError("output directory must be a dedicated exp_075 or exp_075_* path")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    started = time.perf_counter_ns()
    blobs: dict[str, bytes] = {}
    manifest_rows: list[dict[str, Any]] = []
    for source in config["sources"]:
        payload, row = fetch_source(source, arguments.source_dir)
        blobs[str(source["name"])] = payload
        manifest_rows.append(row)
        raw_path = output / "raw/sources" / str(source["output_name"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)

    checkpoint = config["checkpoint"]
    checkpoint_audit = audit_checkpoint_surface(
        model_config=json.loads(blobs["model_config"]),
        weight_index=json.loads(blobs["weight_index"]),
        model_card=blobs["model_card"].decode("utf-8"),
        expected_architecture=str(checkpoint["expected_architecture"]),
        expected_model_type=str(checkpoint["expected_text_model_type"]),
        expected_mtp_layers=int(checkpoint["expected_mtp_hidden_layers"]),
        expected_mtp_keys=checkpoint["expected_mtp_tensor_keys"],
        required_card_markers=checkpoint["required_model_card_markers"],
    )
    runtime_text = {
        name: payload.decode("utf-8")
        for name, payload in blobs.items()
        if name.startswith("vllm_")
    }
    runtime_audit = audit_runtime_surface(
        source_text=runtime_text,
        required_markers=config["runtime"]["required_markers"],
    )
    controls = [
        {
            "control": "metadata_source_guard",
            "passed": True,
            "detail": "all source URLs passed the no-weight-payload guard",
        },
        {
            "control": "pinned_checkpoint_revision_in_all_hf_urls",
            "passed": all(
                checkpoint["revision"] in row["requested_url"]
                for row in manifest_rows
                if "huggingface.co" in row["requested_url"]
            ),
        },
        {
            "control": "pinned_runtime_revision_in_all_vllm_urls",
            "passed": all(
                config["runtime"]["revision"] in row["requested_url"]
                for row in manifest_rows
                if "vllm-project/vllm" in row["requested_url"]
            ),
        },
        {
            "control": "source_hashes_unique",
            "passed": len({row["sha256"] for row in manifest_rows})
            == len(manifest_rows),
        },
        {
            "control": "deterministic_checkpoint_replay",
            "passed": checkpoint_audit
            == audit_checkpoint_surface(
                model_config=json.loads(blobs["model_config"]),
                weight_index=json.loads(blobs["weight_index"]),
                model_card=blobs["model_card"].decode("utf-8"),
                expected_architecture=str(checkpoint["expected_architecture"]),
                expected_model_type=str(checkpoint["expected_text_model_type"]),
                expected_mtp_layers=int(checkpoint["expected_mtp_hidden_layers"]),
                expected_mtp_keys=checkpoint["expected_mtp_tensor_keys"],
                required_card_markers=checkpoint["required_model_card_markers"],
            ),
        },
    ]
    controls_passed = all(bool(row["passed"]) for row in controls)
    decision = surface_decision(
        checkpoint_passed=bool(checkpoint_audit["passed"]),
        runtime_passed=bool(runtime_audit["passed"]),
        controls_passed=controls_passed,
    )
    deterministic_core = {
        "checkpoint": checkpoint_audit,
        "runtime": runtime_audit,
        "controls": controls,
        "source_hashes": {
            row["name"]: row["sha256"] for row in manifest_rows
        },
        "decision": decision,
    }
    core_hash = canonical_sha256(deterministic_core)
    peak_traced_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    measured = {
        "source_count": len(manifest_rows),
        "source_bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "mtp_tensor_count": checkpoint_audit["mtp_tensor_count"],
        "runtime_file_count": len(runtime_audit["files"]),
        "control_count": len(controls),
        "control_failures": sum(not bool(row["passed"]) for row in controls),
        "deterministic_core_sha256": core_hash,
        "peak_traced_bytes": peak_traced_bytes,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    derived = {
        "decision": decision,
        "checkpoint_surface_pass": checkpoint_audit["passed"],
        "runtime_surface_pass": runtime_audit["passed"],
        "declared_weight_bytes": checkpoint_audit["declared_weight_bytes"],
        "declared_weight_gib": checkpoint_audit["declared_weight_gib"],
        "mtp_num_hidden_layers": checkpoint_audit["mtp_num_hidden_layers"],
        "selected_next_model": checkpoint["model_id"],
        "selected_next_revision": checkpoint["revision"],
    }
    summary = {
        "experiment": "EXP-075",
        "name": config["name"],
        "phase": ["A", "B-static-public-metadata"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "checkpoint": checkpoint,
            "runtime": {
                "repository": config["runtime"]["repository"],
                "revision": config["runtime"]["revision"],
            },
        },
        "MEASURED": measured,
        "DERIVED": derived,
        "UNVERIFIED": [
            "MTP proposal token correctness or acceptance distribution",
            "causal accepted-prefix p50 and p95",
            "more-than-one-step recursive MTP usefulness",
            "quantized checkpoint preservation of all MTP tensors",
            "runtime execution on the local or target GPU",
            "Gated DeltaNet rollback and rejection-state correctness",
            "proposal LM-head verification fallback compute and memory traffic",
            "physical latency VRAM RAM SSD PCIe H2D power and thermal behavior",
            "expert route locality on 35B 122B or any MoE checkpoint",
            "transfer to arbitrary dense 405B checkpoints",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint_metadata": "MTP_TENSORS_INDEXED_AT_PINNED_OFFICIAL_REVISION",
            "vllm_source": "STATIC_MAPPING_AND_LOADER_SURFACE_PRESENT_AT_PINNED_REVISION",
            "weight_payload_download": "NOT_PERFORMED",
            "model_execution": "NOT_PERFORMED",
            "quantization_path": "NOT_VALIDATED",
            "target_server": "NO_COMMAND_EXECUTED",
            "dense_405b": "NOT_VALIDATED_BY_QWEN_SPECIFIC_SURFACE",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_mode": "offline-fixture" if arguments.source_dir else "network-pinned",
        },
    }
    require_finite_metrics(summary)
    dump(output / "raw/source_manifest.json", manifest_rows)
    dump(output / "raw/control_rows.json", controls)
    dump(output / "processed/surface_audit.json", deterministic_core)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {"experiment": "EXP-075", "decision": decision, "measured": measured},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not controls_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
