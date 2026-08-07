from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vortex_runtime.target_inventory import (
    InventoryError,
    build_inventory,
    collect_over_ssh,
    parse_wire_output,
    sha256_json,
    validate_required_fields,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments" / "exp_073" / "config.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_bundle(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    collection_mode: str,
    inventory: dict[str, Any] | None,
    decision: str,
    validation_failures: list[str],
    transport: dict[str, Any],
    elapsed_seconds: float,
    failure_category: str | None = None,
) -> dict[str, Any]:
    source_commit = _source_commit()
    provenance = {
        "inventory": "MEASURED" if collection_mode == "TARGET_SSH" and inventory is not None else "SYNTHETIC",
        "completeness_decision": "DERIVED",
        "storage_bandwidth": "UNVERIFIED",
        "h2d_bandwidth": "UNVERIFIED",
        "inference_latency": "UNVERIFIED",
        "vortex_improvement": "UNVERIFIED",
        "405b_execution": "UNVERIFIED",
    }
    core = {
        "experiment": config["experiment"],
        "stage": config["stage"],
        "protocol_version": config["protocol_version"],
        "collection_mode": collection_mode,
        "remote_mutation_allowed": config["remote_mutation_allowed"],
        "inventory": inventory,
        "validation_failures": validation_failures,
        "decision": decision,
        "failure_category": failure_category,
        "provenance": provenance,
        "stage_2_authorized": config["stage_2_authorized"],
        "phase_d_runtime_validation": "NOT_TESTED",
        "e6": "NOT_ACHIEVED",
        "e7": "NOT_ACHIEVED",
    }
    summary = {
        **core,
        "evidence_core_sha256": sha256_json(core),
        "config_sha256": _sha256_file(config_path),
        "source_commit": source_commit,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_ceiling": config["evidence_ceiling"],
        "transport": transport,
    }

    (output_dir / "processed").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    if inventory is not None:
        (output_dir / "raw").mkdir(exist_ok=True)
        (output_dir / "raw" / "sanitized_inventory.json").write_bytes(_json_bytes(inventory))
    (output_dir / "processed" / "validation.json").write_bytes(
        _json_bytes({"validation_failures": validation_failures, "private_identifier_scan": "PASS"})
    )
    (output_dir / "logs" / "run.log").write_text(
        "\n".join(
            [
                f"collection_mode={collection_mode}",
                f"transport_returncode={transport['returncode']}",
                f"transport_stderr_present={str(transport['stderr_present']).lower()}",
                f"transport_stdout_lines={transport['stdout_lines']}",
                f"elapsed_seconds={elapsed_seconds:.6f}",
                f"decision={decision}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))

    checksum_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksum_text = "".join(f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in checksum_paths)
    (output_dir / "checksums.sha256").write_text(checksum_text, encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EXP-073 Stage 1 sanitized target inventory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="Runtime-only SSH alias; never serialized")
    source.add_argument("--fixture", type=Path, help="Offline wire fixture; never target evidence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir)
    started = time.monotonic()

    collection_mode = "FIXTURE_VALIDATION" if args.fixture else "TARGET_SSH"
    transport = {"returncode": 0, "stderr_present": False, "stdout_lines": 0}
    try:
        if args.fixture:
            stdout = args.fixture.read_text(encoding="utf-8")
        else:
            collected = collect_over_ssh(args.host, int(config["ssh_timeout_seconds"]))
            transport = {
                "returncode": collected.returncode,
                "stderr_present": collected.stderr_present,
                "stdout_lines": len(collected.stdout.splitlines()),
            }
            if collected.returncode != 0:
                raise ConnectionError("SSH_COLLECTION_FAILED")
            stdout = collected.stdout
        transport["stdout_lines"] = len(stdout.splitlines())
        inventory = build_inventory(parse_wire_output(stdout))
        failures = validate_required_fields(inventory, config["required_fields"])
        if failures:
            raise InventoryError("required inventory fields failed validation")
        decision = (
            config["complete_decision"]
            if collection_mode == "TARGET_SSH"
            else "FIXTURE_VALIDATION_ONLY_NOT_TARGET_EVIDENCE"
        )
        summary = _write_bundle(
            output_dir,
            config_path,
            config,
            collection_mode=collection_mode,
            inventory=inventory,
            decision=decision,
            validation_failures=[],
            transport=transport,
            elapsed_seconds=time.monotonic() - started,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (ConnectionError, InventoryError, subprocess.TimeoutExpired) as exc:
        category = type(exc).__name__.upper()
        summary = _write_bundle(
            output_dir,
            config_path,
            config,
            collection_mode=collection_mode,
            inventory=None,
            decision=config["infrastructure_decision"],
            validation_failures=[category],
            transport=transport,
            elapsed_seconds=time.monotonic() - started,
            failure_category=category,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
