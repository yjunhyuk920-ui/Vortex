#!/usr/bin/env python3
"""Capture target-hardware prerequisites without pretending to run Phase D."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def run(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as exc:
        return {"command": command, "error": repr(exc)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    output_dir = Path("results/exp_047/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_text = os.environ.get("VORTEX_MODEL_PATH")
    revision = os.environ.get("VORTEX_MODEL_REVISION")
    manifest_text = os.environ.get("VORTEX_CHECKPOINT_MANIFEST")

    report: dict[str, object] = {
        "timestamp_unix": time.time(),
        "phase": "D preflight only",
        "phase_d_status": "NOT TESTED",
        "python": sys.version,
        "platform": platform.platform(),
        "model_path": model_path_text,
        "model_revision": revision,
        "checkpoint_manifest": manifest_text,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "commands": [],
        "errors": [],
    }

    if shutil.which("nvidia-smi"):
        report["commands"].append(
            run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
    else:
        report["errors"].append("nvidia-smi is unavailable")

    if not model_path_text:
        report["errors"].append("VORTEX_MODEL_PATH is required")
    else:
        model_path = Path(model_path_text)
        if not model_path.exists():
            report["errors"].append("VORTEX_MODEL_PATH does not exist")
        else:
            report["model_path_bytes"] = sum(
                item.stat().st_size for item in model_path.rglob("*") if item.is_file()
            )

    if not revision:
        report["errors"].append("VORTEX_MODEL_REVISION is required")

    if not manifest_text:
        report["errors"].append("VORTEX_CHECKPOINT_MANIFEST is required")
    else:
        manifest = Path(manifest_text)
        if not manifest.is_file():
            report["errors"].append("checkpoint manifest does not exist")
        else:
            report["checkpoint_manifest_sha256"] = sha256_file(manifest)

    report_path = output_dir / "future_gpu_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
