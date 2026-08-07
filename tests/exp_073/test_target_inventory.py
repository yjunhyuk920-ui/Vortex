from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vortex_runtime.target_inventory import (
    InventoryError,
    REMOTE_READ_ONLY_PROGRAM,
    build_inventory,
    collect_over_ssh,
    parse_wire_output,
    sha256_json,
    validate_required_fields,
)
from experiments.exp_073.run_stage1 import _write_text_lf


FIXTURE = Path(__file__).parent / "fixtures" / "valid_wire.txt"
CONFIG = Path(__file__).parents[2] / "experiments" / "exp_073" / "config.json"


def valid_raw() -> dict[str, str]:
    return parse_wire_output(FIXTURE.read_text(encoding="utf-8"))


def test_valid_fixture_builds_sanitized_inventory() -> None:
    inventory = build_inventory(valid_raw())
    assert inventory["gpu"]["models"] == ["NVIDIA Example GPU"]
    assert inventory["gpu"]["cuda_driver_api_versions"] == ["12.2"]
    assert inventory["gpu"]["total_vram_mib"] == [8192]
    assert inventory["storage"]["block_device_count"] == 2
    assert inventory["storage"]["block_total_bytes"] == 3_000_603_820_032
    assert inventory["storage"]["transport_counts"] == {"sata": 1, "usb": 1}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert validate_required_fields(inventory, config["required_fields"]) == []


def test_unknown_and_duplicate_wire_keys_fail_closed() -> None:
    fixture = FIXTURE.read_text(encoding="utf-8")
    with pytest.raises(InventoryError, match="unknown key"):
        parse_wire_output(fixture + "hostname\tprivate-host\n")
    with pytest.raises(InventoryError, match="duplicates"):
        parse_wire_output(fixture + "gpu.count\t1\n")


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform.os_pretty_name", "10.20.30.40"),
        ("cpu.model_name", "private-user@private-host"),
        ("runtimes.python3_version", "/home/private-user/runtime"),
        ("gpu.models", "-----BEGIN PRIVATE KEY-----"),
        ("platform.kernel_release", "123e4567-e89b-12d3-a456-426614174000"),
        ("runtimes.ollama_version", "C:\\private\\runtime.exe"),
    ],
)
def test_private_identifier_patterns_fail_closed(field: str, value: str) -> None:
    raw = valid_raw()
    raw[field] = value
    with pytest.raises(InventoryError, match="forbidden identifier"):
        build_inventory(raw)


def test_missing_optional_tools_are_not_installed_or_promoted() -> None:
    raw = valid_raw()
    raw["runtimes.fio_version"] = "bash: fio: command not found"
    raw["runtimes.nvcc_version"] = "NOT_AVAILABLE"
    inventory = build_inventory(raw)
    assert inventory["runtimes"]["fio_version"] == "NOT_AVAILABLE"
    assert inventory["runtimes"]["nvcc_version"] == "NOT_AVAILABLE"


def test_gpu_count_and_storage_rows_must_be_consistent() -> None:
    raw = valid_raw()
    raw["gpu.count"] = "2"
    with pytest.raises(InventoryError, match="does not match gpu.count|incomplete"):
        build_inventory(raw)
    raw = valid_raw()
    raw["storage.block_rows"] = "0,100,disk,nvme;bad-row;"
    with pytest.raises(InventoryError, match="invalid row"):
        build_inventory(raw)


def test_fixture_replay_has_deterministic_core_hash() -> None:
    inventory = build_inventory(valid_raw())
    core = {"protocol": 1, "inventory": inventory, "decision": "FIXTURE_VALIDATION_ONLY_NOT_TARGET_EVIDENCE"}
    assert sha256_json(core) == sha256_json(json.loads(json.dumps(core)))


def test_ssh_alias_is_runtime_only_and_program_has_no_mutating_tools() -> None:
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(command, 0, FIXTURE.read_bytes(), b"")

    alias = "private-target-alias"
    collected = collect_over_ssh(alias, 20, runner=fake_runner)
    inventory = build_inventory(parse_wire_output(collected.stdout))
    serialized = json.dumps(inventory, sort_keys=True)
    assert alias in captured["command"]
    assert alias not in serialized
    assert isinstance(captured["input"], bytes)
    assert b"\r" not in captured["input"]
    assert "StrictHostKeyChecking=yes" in captured["command"]
    assert "UpdateHostKeys=no" in captured["command"]
    forbidden = (
        "apt ",
        "apt-get",
        "pip install",
        "systemctl restart",
        "ollama run",
        "fio --name",
        "fio --filename",
        "touch ",
        "tee ",
    )
    assert not any(token in REMOTE_READ_ONLY_PROGRAM for token in forbidden)


def test_nonzero_ssh_status_is_exposed_without_stderr_content() -> None:
    def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 255, b"", b"private-target: permission denied")

    collected = collect_over_ssh("private-target-alias", 20, runner=fake_runner)
    assert collected.returncode == 255
    assert collected.stderr_present is True
    assert collected.stdout == ""


def test_evidence_text_writer_is_lf_only(tmp_path: Path) -> None:
    output = tmp_path / "evidence.txt"
    _write_text_lf(output, "first\nsecond\n")
    assert output.read_bytes() == b"first\nsecond\n"
