from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


NOT_AVAILABLE = "NOT_AVAILABLE"

REMOTE_READ_ONLY_PROGRAM = r'''set -u
export LC_ALL=C

emit() {
  key="$1"
  value="${2:-}"
  value="$(printf '%s' "$value" | tr '\r\n\t' '   ')"
  if [ -z "$value" ]; then value="NOT_AVAILABLE"; fi
  printf '%s\t%s\n' "$key" "$value"
}

gpu_query() {
  field="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  nvidia-smi --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | paste -sd '|' -
}

gpu_query_available() {
  value="$(gpu_query "$1")"
  case "$value" in
    ""|*"N/A"*|*"Not Supported"*) printf '%s' "false" ;;
    *) printf '%s' "true" ;;
  esac
}

os_name="$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | head -n 1 | sed 's/^"//;s/"$//')"
cpu_model="$(lscpu 2>/dev/null | awk -F: '/^Model name:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
mem_total="$(awk '/^MemTotal:/ {printf "%.0f", $2 * 1024; exit}' /proc/meminfo 2>/dev/null)"
mem_available="$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024; exit}' /proc/meminfo 2>/dev/null)"

emit platform.os_pretty_name "$os_name"
emit platform.kernel_release "$(uname -r 2>/dev/null || true)"
emit platform.machine_architecture "$(uname -m 2>/dev/null || true)"
emit cpu.model_name "$cpu_model"
emit cpu.logical_cpu_count "$cpu_count"
emit memory.total_bytes "$mem_total"
emit memory.available_bytes "$mem_available"

gpu_models="$(gpu_query name)"
if [ -n "$gpu_models" ]; then
  gpu_count="$(printf '%s' "$gpu_models" | awk -F'|' '{print NF}')"
else
  gpu_count="0"
fi
emit gpu.count "$gpu_count"
emit gpu.models "$gpu_models"
emit gpu.driver_versions "$(gpu_query driver_version)"
cuda_driver_api_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
emit gpu.cuda_driver_api_versions "$cuda_driver_api_version"
emit gpu.compute_capabilities "$(gpu_query compute_cap)"
emit gpu.total_vram_mib "$(gpu_query memory.total)"
emit gpu.free_vram_mib "$(gpu_query memory.free)"
emit gpu.pcie_current_generation "$(gpu_query pcie.link.gen.current)"
emit gpu.pcie_current_width "$(gpu_query pcie.link.width.current)"
emit gpu.pcie_max_generation "$(gpu_query pcie.link.gen.max)"
emit gpu.pcie_max_width "$(gpu_query pcie.link.width.max)"

root_row="$(df -B1 -P -T / 2>/dev/null | awk 'NR == 2 {print $2 "|" $3 "|" $5}')"
block_rows="$(lsblk -b -d -n -o ROTA,SIZE,TYPE,TRAN 2>/dev/null \
  | awk '$3 == "disk" {transport=$4; if (transport == "") transport="unknown"; printf "%s,%s,%s,%s;", $1, $2, $3, transport}')"
emit storage.root_row "$root_row"
emit storage.block_rows "$block_rows"

python_version="$(python3 --version 2>&1 | head -n 1 || true)"
ollama_version="$(ollama --version 2>&1 | tail -n 1 || true)"
fio_version="$(fio --version 2>&1 | head -n 1 || true)"
nvcc_version="$(nvcc --version 2>/dev/null | awk '/release/ {line=$0} END {print line}' || true)"
if command -v systemctl >/dev/null 2>&1; then
  ollama_state="$(systemctl is-active ollama 2>/dev/null || true)"
else
  ollama_state="NOT_AVAILABLE"
fi
emit runtimes.python3_version "$python_version"
emit runtimes.ollama_version "$ollama_version"
emit runtimes.fio_version "$fio_version"
emit runtimes.nvcc_version "$nvcc_version"
emit runtimes.ollama_service_state "$ollama_state"
emit telemetry.gpu_power_available "$(gpu_query_available power.draw)"
emit telemetry.gpu_temperature_available "$(gpu_query_available temperature.gpu)"
emit telemetry.gpu_clock_available "$(gpu_query_available clocks.current.graphics)"
'''


WIRE_KEYS = frozenset(
    {
        "platform.os_pretty_name",
        "platform.kernel_release",
        "platform.machine_architecture",
        "cpu.model_name",
        "cpu.logical_cpu_count",
        "memory.total_bytes",
        "memory.available_bytes",
        "gpu.count",
        "gpu.models",
        "gpu.driver_versions",
        "gpu.cuda_driver_api_versions",
        "gpu.compute_capabilities",
        "gpu.total_vram_mib",
        "gpu.free_vram_mib",
        "gpu.pcie_current_generation",
        "gpu.pcie_current_width",
        "gpu.pcie_max_generation",
        "gpu.pcie_max_width",
        "storage.root_row",
        "storage.block_rows",
        "runtimes.python3_version",
        "runtimes.ollama_version",
        "runtimes.fio_version",
        "runtimes.nvcc_version",
        "runtimes.ollama_service_state",
        "telemetry.gpu_power_available",
        "telemetry.gpu_temperature_available",
        "telemetry.gpu_clock_available",
    }
)

_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6 = re.compile(r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f]{0,4}(?![0-9a-f:])")
_USER_AT_HOST = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+")
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")
_UNIX_PATH = re.compile(r"(?:^|\s)/(?:[^\s]+)")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\")
_KEY_MARKER = re.compile(r"(?i)(?:BEGIN [A-Z ]*PRIVATE KEY|ssh-(?:rsa|ed25519)|private[_ -]?key)")
_TRANSPORT = re.compile(r"^[A-Za-z0-9_.-]+$")


class InventoryError(ValueError):
    pass


@dataclass(frozen=True)
class SSHCollection:
    stdout: str
    returncode: int
    stderr_present: bool


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_wire_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line:
            continue
        if "\t" not in line:
            raise InventoryError(f"wire line {line_number} has no tab separator")
        key, value = line.split("\t", 1)
        if key not in WIRE_KEYS:
            raise InventoryError(f"wire line {line_number} has unknown key")
        if key in parsed:
            raise InventoryError(f"wire line {line_number} duplicates a key")
        if any(ord(char) < 32 for char in value):
            raise InventoryError(f"wire line {line_number} contains a control character")
        parsed[key] = value.strip() or NOT_AVAILABLE
    missing = sorted(WIRE_KEYS - parsed.keys())
    if missing:
        raise InventoryError(f"wire output is missing {len(missing)} keys")
    return parsed


def _safe_public_string(value: str) -> str:
    value = " ".join(value.strip().split()) or NOT_AVAILABLE
    checks = (_IPV4, _IPV6, _USER_AT_HOST, _UUID, _UNIX_PATH, _WINDOWS_PATH, _KEY_MARKER)
    if any(pattern.search(value) for pattern in checks):
        raise InventoryError("a public inventory string contains a forbidden identifier pattern")
    return value


def _not_available(value: str) -> bool:
    normalized = value.strip()
    return normalized in {"", NOT_AVAILABLE, "N/A", "[Not Supported]"}


def _integer(value: str, field: str, *, minimum: int = 0) -> int:
    if _not_available(value):
        raise InventoryError(f"{field} is unavailable")
    try:
        number = int(value.strip())
    except ValueError as exc:
        raise InventoryError(f"{field} is not an integer") from exc
    if number < minimum:
        raise InventoryError(f"{field} is below its minimum")
    return number


def _list(value: str, field: str, converter: Callable[[str], Any] = _safe_public_string) -> list[Any] | str:
    if _not_available(value):
        return NOT_AVAILABLE
    items: list[Any] = []
    for item in value.split("|"):
        item = item.strip()
        if _not_available(item):
            items.append(NOT_AVAILABLE)
        else:
            try:
                items.append(converter(item))
            except InventoryError:
                raise
            except ValueError as exc:
                raise InventoryError(f"{field} contains an invalid item") from exc
    return items


def _integer_list(value: str, field: str) -> list[int | str] | str:
    return _list(value, field, lambda item: _integer(item, field))


def _boolean(value: str, field: str) -> bool | str:
    if _not_available(value):
        return NOT_AVAILABLE
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise InventoryError(f"{field} is not a boolean")
    return normalized == "true"


def _runtime_string(value: str) -> str:
    if _not_available(value):
        return NOT_AVAILABLE
    lower = value.lower()
    missing_markers = ("command not found", "not recognized", "no such file")
    if any(marker in lower for marker in missing_markers):
        return NOT_AVAILABLE
    return _safe_public_string(value)


def _parse_storage(raw: Mapping[str, str]) -> dict[str, Any]:
    root_parts = raw["storage.root_row"].split("|")
    if len(root_parts) != 3:
        raise InventoryError("storage.root_row has the wrong shape")
    root_type = _safe_public_string(root_parts[0])
    root_total = _integer(root_parts[1], "storage.root_total_bytes", minimum=1)
    root_available = _integer(root_parts[2], "storage.root_available_bytes")
    if root_available > root_total:
        raise InventoryError("root available bytes exceed total bytes")

    count = 0
    total = 0
    rotational = 0
    transports: dict[str, int] = {}
    rows = [row for row in raw["storage.block_rows"].split(";") if row]
    for row in rows:
        parts = row.split(",")
        if len(parts) != 4 or parts[2] != "disk":
            raise InventoryError("storage.block_rows has an invalid row")
        rota = _integer(parts[0], "storage.block_rows.rotational")
        size = _integer(parts[1], "storage.block_rows.size", minimum=1)
        transport = parts[3].strip() or "unknown"
        if not _TRANSPORT.fullmatch(transport):
            raise InventoryError("storage transport has invalid characters")
        count += 1
        total += size
        rotational += int(rota != 0)
        transports[transport] = transports.get(transport, 0) + 1
    if count == 0:
        raise InventoryError("no block devices were reported")

    return {
        "root_filesystem_type": root_type,
        "root_total_bytes": root_total,
        "root_available_bytes": root_available,
        "block_device_count": count,
        "block_total_bytes": total,
        "rotational_device_count": rotational,
        "transport_counts": dict(sorted(transports.items())),
    }


def build_inventory(raw: Mapping[str, str]) -> dict[str, Any]:
    unknown = set(raw) - WIRE_KEYS
    missing = WIRE_KEYS - set(raw)
    if unknown or missing:
        raise InventoryError("wire mapping does not match the allowlist")

    gpu_count = _integer(raw["gpu.count"], "gpu.count")
    inventory: dict[str, Any] = {
        "platform": {
            "os_pretty_name": _safe_public_string(raw["platform.os_pretty_name"]),
            "kernel_release": _safe_public_string(raw["platform.kernel_release"]),
            "machine_architecture": _safe_public_string(raw["platform.machine_architecture"]),
        },
        "cpu": {
            "model_name": _safe_public_string(raw["cpu.model_name"]),
            "logical_cpu_count": _integer(raw["cpu.logical_cpu_count"], "cpu.logical_cpu_count", minimum=1),
        },
        "memory": {
            "total_bytes": _integer(raw["memory.total_bytes"], "memory.total_bytes", minimum=1),
            "available_bytes": _integer(raw["memory.available_bytes"], "memory.available_bytes"),
        },
        "gpu": {
            "count": gpu_count,
            "models": _list(raw["gpu.models"], "gpu.models"),
            "driver_versions": _list(raw["gpu.driver_versions"], "gpu.driver_versions"),
            "cuda_driver_api_versions": _list(raw["gpu.cuda_driver_api_versions"], "gpu.cuda_driver_api_versions"),
            "compute_capabilities": _list(raw["gpu.compute_capabilities"], "gpu.compute_capabilities"),
            "total_vram_mib": _integer_list(raw["gpu.total_vram_mib"], "gpu.total_vram_mib"),
            "free_vram_mib": _integer_list(raw["gpu.free_vram_mib"], "gpu.free_vram_mib"),
            "pcie_current_generation": _integer_list(raw["gpu.pcie_current_generation"], "gpu.pcie_current_generation"),
            "pcie_current_width": _integer_list(raw["gpu.pcie_current_width"], "gpu.pcie_current_width"),
            "pcie_max_generation": _integer_list(raw["gpu.pcie_max_generation"], "gpu.pcie_max_generation"),
            "pcie_max_width": _integer_list(raw["gpu.pcie_max_width"], "gpu.pcie_max_width"),
        },
        "storage": _parse_storage(raw),
        "runtimes": {
            "python3_version": _runtime_string(raw["runtimes.python3_version"]),
            "ollama_version": _runtime_string(raw["runtimes.ollama_version"]),
            "fio_version": _runtime_string(raw["runtimes.fio_version"]),
            "nvcc_version": _runtime_string(raw["runtimes.nvcc_version"]),
            "ollama_service_state": _runtime_string(raw["runtimes.ollama_service_state"]),
        },
        "telemetry": {
            "gpu_power_available": _boolean(raw["telemetry.gpu_power_available"], "telemetry.gpu_power_available"),
            "gpu_temperature_available": _boolean(raw["telemetry.gpu_temperature_available"], "telemetry.gpu_temperature_available"),
            "gpu_clock_available": _boolean(raw["telemetry.gpu_clock_available"], "telemetry.gpu_clock_available"),
        },
    }

    if inventory["memory"]["available_bytes"] > inventory["memory"]["total_bytes"]:
        raise InventoryError("available memory exceeds total memory")
    if gpu_count < 1:
        raise InventoryError("no NVIDIA GPU was reported")
    required_gpu_lists = ("models", "driver_versions", "total_vram_mib", "free_vram_mib")
    for field in required_gpu_lists:
        values = inventory["gpu"][field]
        if not isinstance(values, list) or len(values) != gpu_count or NOT_AVAILABLE in values:
            raise InventoryError(f"gpu.{field} is incomplete")
    for field, values in inventory["gpu"].items():
        if field == "count" or values == NOT_AVAILABLE:
            continue
        if isinstance(values, list) and len(values) != gpu_count:
            raise InventoryError(f"gpu.{field} does not match gpu.count")
    for free, total in zip(inventory["gpu"]["free_vram_mib"], inventory["gpu"]["total_vram_mib"], strict=True):
        if free > total:
            raise InventoryError("free VRAM exceeds total VRAM")
    return inventory


def validate_required_fields(inventory: Mapping[str, Any], required_fields: Sequence[str]) -> list[str]:
    failures: list[str] = []
    for dotted in required_fields:
        value: Any = inventory
        try:
            for component in dotted.split("."):
                value = value[component]
        except (KeyError, TypeError):
            failures.append(dotted)
            continue
        if value in (None, "", NOT_AVAILABLE, []):
            failures.append(dotted)
    return failures


def collect_over_ssh(
    host_alias: str,
    timeout_seconds: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> SSHCollection:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", host_alias):
        raise InventoryError("SSH alias has invalid characters")
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "LogLevel=ERROR",
        "-o",
        f"ConnectTimeout={timeout_seconds}",
        host_alias,
        "bash --noprofile --norc -s",
    ]
    completed = runner(
        command,
        input=REMOTE_READ_ONLY_PROGRAM,
        text=True,
        capture_output=True,
        timeout=timeout_seconds + 10,
        check=False,
    )
    return SSHCollection(
        stdout=completed.stdout,
        returncode=completed.returncode,
        stderr_present=bool(completed.stderr.strip()),
    )
