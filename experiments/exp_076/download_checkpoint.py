#!/usr/bin/env python3
"""Download only the EXP-076 preregistered checkpoint files and verify them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload, usedforsecurity=False
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_076/config.json"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / ".deps/exp076-model"
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    model_dir = arguments.model_dir.resolve()
    allowed_root = (ROOT / ".deps").resolve()
    if model_dir == allowed_root or allowed_root not in model_dir.parents:
        raise ValueError("model directory must be a dedicated child of .deps")
    required = config["checkpoint"]["required_files"]
    snapshot_download(
        repo_id=config["checkpoint"]["model_id"],
        revision=config["checkpoint"]["revision"],
        local_dir=model_dir,
        allow_patterns=[row["path"] for row in required],
        max_workers=4,
    )
    rows = []
    for expected in required:
        path = model_dir / expected["path"]
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"missing or wrong-sized checkpoint file: {path}")
        sha256 = sha256_file(path)
        if expected.get("sha256") and sha256 != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        if not expected.get("sha256") and git_blob_oid(path) != expected["repository_oid"]:
            raise RuntimeError(f"repository OID mismatch: {path}")
        rows.append({"path": expected["path"], "bytes": path.stat().st_size, "sha256": sha256})
    print(json.dumps({"model_dir": str(model_dir), "files": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
